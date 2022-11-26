import logging
import os
import random
import math
from tkinter import N

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from datasets import my_collate
from torch.optim import Adam
from transformers import AdamW
from transformers import BertTokenizer
from utils_gm.torch_utils.lagrangian_optimization import LagrangianOptimization

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# def get_input_from_batch(args, batch):
#
#     inputs = {  'post_tokens_id': batch[0], 'embeddings': batch[1],
#                 'label1': batch[2], 'label2': batch[3], 'label3': batch[4], 'label4': batch[5]}  # B * 50 * 70
#     return inputs
def get_input_from_batch(args, batch):

    inputs = {  'post_tokens_id': batch[0]}  # B * 50 * 70
    label1 = batch[1]
    label2 = batch[2]
    label3 = batch[3]
    label4 = batch[4]
    return inputs, label1, label2, label3, label4

def get_collate_fn(args):

    return my_collate


def get_optimizer(args, model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    total_params_num = sum(x.numel() for x in model.parameters())
    bert_params_num = sum(x.numel() for x in model.pretrain_models.parameters())
    gm_params_num = args.gcn_num_layers * sum(x.numel() for x in model.multi_dgcn.dgcn1.A[0].parameters()) * 4
    gm_params_num += args.gcn_num_layers * sum(x.numel() for x in model.multi_dgcn.dgcn1.A[1].parameters()) * 4
    other_params_num = total_params_num - bert_params_num - gm_params_num
    logger.info('parameters of plm, graphmask and other: %d, %d, %d', bert_params_num, gm_params_num, other_params_num)
    bert_params = list(map(id, model.pretrain_models.parameters()))
    #
    gm_params_id = []
    for l in range(args.gcn_num_layers):
        gm_params_id += list(map(id, model.multi_dgcn.dgcn1.A[l].parameters()))
        gm_params_id += list(map(id, model.multi_dgcn.dgcn2.A[l].parameters()))
        gm_params_id += list(map(id, model.multi_dgcn.dgcn3.A[l].parameters()))
        gm_params_id += list(map(id, model.multi_dgcn.dgcn4.A[l].parameters()))

    #gm_params = list(map(id, model.multi_dgcn.parameters()))
    other_params = filter(lambda p: id(p) not in bert_params + gm_params_id, model.parameters())
    gm_params = filter(lambda p: id(p) in gm_params_id, model.parameters())
    optimizer_grouped_parameters = [
        {'params':other_params, 'lr': args.other_learning_rate},
        {'params':model.pretrain_models.parameters(), 'lr':args.learning_rate},
        {'params':gm_params, 'lr': args.gm_learning_rate}
    ]
    
    model_optimizer = Adam(optimizer_grouped_parameters,
                      eps=args.adam_epsilon)
    optimizer = LagrangianOptimization(model_optimizer, args.device, gradient_accumulation_steps=args.gradient_accumulation_steps, alpha_optimizer_lr=args.alpha_learning_rate, max_grad_norm=args.max_grad_norm, max_alpha=args.max_alpha) if args.l0 else model_optimizer #L0

    return optimizer

def write_tb_writer(tb_writer: SummaryWriter, args):
    tb_writer.add_text ('seed', str(args.seed))
    tb_writer.add_text('plm_lr', str(args.learning_rate))
    tb_writer.add_text('ohter_lr', str(args.other_learning_rate))
    tb_writer.add_text('a_lr', str(args.alpha_learning_rate))
    tb_writer.add_text('gm_lr', str(args.gm_learning_rate))
    tb_writer.add_text('gcn_layers', str(args.gcn_num_layers))
    tb_writer.add_text('final_hidden_size', str(args.final_hidden_size))
    tb_writer.add_text('bsz', str(args.all_gpu_train_batch_size))
    tb_writer.add_text('max_a', str(args.max_alpha))

def train(args, train_dataset, model, eval_dataset, test_dataset):
    '''Train the model'''
    tb_writer = SummaryWriter()
    write_tb_writer(tb_writer, args)
    print('-----------training-----------')
    args.train_batch_size = args.all_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn(args)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    optimizer = get_optimizer(args, model)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.all_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_ave_f1 = 0.0
    best_test_ave_f1 = 0.0
    best_epoch = 0
    best_other_result = None
    best_test_other_result = None
    all_eval_results = []
    all_test_resylts = []
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    for train_iter, _ in enumerate(train_iterator):
        preds1 = None
        preds2 = None
        preds3 = None
        preds4 = None
        out_label_ids1 = None
        out_label_ids2 = None
        out_label_ids3 = None
        out_label_ids4 = None
        layer_sparsity1 = None
        layer_sparsity2 = None
        layer_sparsity3 = None
        layer_sparsity4 = None
        node_sparsity1 = None
        node_sparsity2 = None
        node_sparsity3 = None
        node_sparsity4 = None
        rs1 = None
        rs2 = None
        rs3 = None
        rs4 = None
        results1, results2, results3, results4 = {}, {}, {}, {}
        step_loss = 0.0
        nb_steps = 0
        # epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        for step, batch in enumerate(train_dataloader):
            nb_steps += 1
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs, label1, label2, label3, label4 = get_input_from_batch(args, batch)
            logit1, logit2, logit3, logit4, l0_attr, retain_scores = model(**inputs)
            l0_attr1, l0_attr2, l0_attr3, l0_attr4 = l0_attr
            if retain_scores is not None: #unused
                retain_scores1, retain_scores2, retain_scores3, retain_scores4 = retain_scores 
            
            if layer_sparsity1 is None:
                layer_sparsity1 = l0_attr1[2]
                layer_sparsity2 = l0_attr2[2]
                layer_sparsity3 = l0_attr3[2]
                layer_sparsity4 = l0_attr4[2]
                node_sparsity1 = l0_attr1[1]
                node_sparsity2 = l0_attr2[1]
                node_sparsity3 = l0_attr3[1]
                node_sparsity4 = l0_attr4[1]
                
            else:
                layer_sparsity1 = torch.cat((layer_sparsity1, l0_attr1[2]), dim=0)
                layer_sparsity2 = torch.cat((layer_sparsity2, l0_attr2[2]), dim=0)
                layer_sparsity3 = torch.cat((layer_sparsity3, l0_attr3[2]), dim=0)
                layer_sparsity4 = torch.cat((layer_sparsity4, l0_attr4[2]), dim=0)
                node_sparsity1 = torch.cat((node_sparsity1, l0_attr1[1]), dim=0)
                node_sparsity2 = torch.cat((node_sparsity2, l0_attr2[1]), dim=0)
                node_sparsity3 = torch.cat((node_sparsity3, l0_attr3[1]), dim=0)
                node_sparsity4 = torch.cat((node_sparsity4, l0_attr4[1]), dim=0)
                
            loss1 = F.cross_entropy(logit1, label1, reduction='sum')
            loss2 = F.cross_entropy(logit2, label2, reduction='sum')
            loss3 = F.cross_entropy(logit3, label3, reduction='sum')
            loss4 = F.cross_entropy(logit4, label4, reduction='sum')

            if preds1 is None:
                preds1 = logit1.detach().cpu().numpy()
                preds2 = logit2.detach().cpu().numpy()
                preds3 = logit3.detach().cpu().numpy()
                preds4 = logit4.detach().cpu().numpy()
                out_label_ids1 = label1.detach().cpu().numpy()
                out_label_ids2 = label2.detach().cpu().numpy()
                out_label_ids3 = label3.detach().cpu().numpy()
                out_label_ids4 = label4.detach().cpu().numpy()
            else:
                preds1 = np.append(preds1, logit1.detach().cpu().numpy(), axis=0)
                preds2 = np.append(preds2, logit2.detach().cpu().numpy(), axis=0)
                preds3 = np.append(preds3, logit3.detach().cpu().numpy(), axis=0)
                preds4 = np.append(preds4, logit4.detach().cpu().numpy(), axis=0)
                out_label_ids1 = np.append(
                    out_label_ids1, label1.detach().cpu().numpy(), axis=0)
                out_label_ids2 = np.append(
                    out_label_ids2, label2.detach().cpu().numpy(), axis=0)
                out_label_ids3 = np.append(
                    out_label_ids3, label3.detach().cpu().numpy(), axis=0)
                out_label_ids4 = np.append(
                    out_label_ids4, label4.detach().cpu().numpy(), axis=0)

            batch_samples = label1.size(0)

            loss = (loss1  + loss2 + loss3 + loss4)/ (4.0 * batch_samples)
            step_loss += loss.item()
            total_l0loss = (l0_attr1[0]+l0_attr2[0]+l0_attr3[0]+l0_attr4[0]) / (4.0 * batch_samples)  

            if math.isnan(loss.item()):
                import pdb; pdb.set_trace() 

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                total_l0loss = total_l0loss / args.gradient_accumulation_steps

            if args.l0:
                lagr_lambda =  optimizer.update(total_l0loss, loss, model)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.l0:
                    tb_writer.add_scalar('lambda', lagr_lambda, global_step)
                else:
                    optimizer.step()
                    model.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    results, eval_loss, ave_f1, other_result = evaluate(args, eval_dataset, model)
                    if ave_f1 > best_ave_f1:
                        # Save model checkpoint
                        torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_f1_dggcn_vis.pth'))
                        best_epoch = train_iter
                        best_ave_f1 = ave_f1
                        best_other_result = other_result
                        test_results, test_loss, test_ave_f1, test_other_result = test(args, test_dataset, model)
                        best_test_ave_f1 = test_ave_f1
                        best_test_other_result = test_other_result
                        all_test_resylts.append(test_results)
                        for key, value in test_results[0].items():
                            tb_writer.add_scalar(
                                'test1_{}'.format(key), value, global_step)
                        for key, value in test_results[1].items():
                            tb_writer.add_scalar(
                                'test2_{}'.format(key), value, global_step)
                        for key, value in test_results[2].items():
                            tb_writer.add_scalar(
                                'test3_{}'.format(key), value, global_step)
                        for key, value in test_results[3].items():
                            tb_writer.add_scalar(
                                'test4_{}'.format(key), value, global_step)
                        tb_writer.add_scalar('test_loss', test_loss, global_step)
                        tb_writer.add_scalar('test_ave_f1', test_ave_f1, global_step)

                    all_eval_results.append(results)
                    for key, value in results[0].items():
                        tb_writer.add_scalar(
                            'eval1_{}'.format(key), value, global_step)
                    for key, value in results[1].items():
                        tb_writer.add_scalar(
                            'eval2_{}'.format(key), value, global_step)
                    for key, value in results[2].items():
                        tb_writer.add_scalar(
                            'eval3_{}'.format(key), value, global_step)
                    for key, value in results[3].items():
                        tb_writer.add_scalar(
                            'eval4_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('eval_loss', eval_loss, global_step)
                    tb_writer.add_scalar('eval_ave_f1', ave_f1, global_step)
                    tb_writer.add_scalar(
                        'train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

        avg_g_spar1 = layer_sparsity1.mean(dim=0)
        avg_g_spar2 = layer_sparsity2.mean(dim=0)
        avg_g_spar3 = layer_sparsity3.mean(dim=0)
        avg_g_spar4 = layer_sparsity4.mean(dim=0)
        avg_c_spar1 = node_sparsity1.mean(dim=0)
        avg_c_spar2 = node_sparsity2.mean(dim=0)
        avg_c_spar3 = node_sparsity3.mean(dim=0)
        avg_c_spar4 = node_sparsity4.mean(dim=0)

        logger.info("sparsity of graph and representing node: ", avg_g_spar1, avg_g_spar2, avg_g_spar3,avg_g_spar4, avg_c_spar1,avg_c_spar2, avg_c_spar3, avg_c_spar4)

        f1_rs, f2_rs, f3_rs, f4_rs = {}, {}, {}, {}

        for l in range(avg_g_spar1.shape[0]):
            avg_g_spar = {'g_spar1':avg_g_spar1[l], 'g_spar2':avg_g_spar2[l], 'g_spar3':avg_g_spar3[l], 'g_spar4':avg_g_spar4[l]}
            avg_c_spar = {'c_spar1':avg_c_spar1[l], 'c_spar2':avg_c_spar2[l], 'c_spar3':avg_c_spar3[l], 'c_spar4':avg_c_spar4[l]}
            #import pdb; pdb.set_trace()
            tb_writer.add_scalars("graph_sparsity"+str(l), avg_g_spar, train_iter)
            tb_writer.add_scalars("cnode_sparsity"+str(l), avg_c_spar, train_iter)

        
        preds1 = np.argmax(preds1, axis=1)
        preds2 = np.argmax(preds2, axis=1)
        preds3 = np.argmax(preds3, axis=1)
        preds4 = np.argmax(preds4, axis=1)

        result1 = compute_metrics(preds1, out_label_ids1)
        result2 = compute_metrics(preds2, out_label_ids2)
        result3 = compute_metrics(preds3, out_label_ids3)
        result4 = compute_metrics(preds4, out_label_ids4)
        results1.update(result1)
        results2.update(result2)
        results3.update(result3)
        results4.update(result4)

        for key, value in results1.items():
            tb_writer.add_scalar(
                'train1_{}'.format(key), value, global_step)
        for key, value in results2.items():
            tb_writer.add_scalar(
                'train2_{}'.format(key), value, global_step)
        for key, value in results3.items():
            tb_writer.add_scalar(
                'train3_{}'.format(key), value, global_step)
        for key, value in results4.items():
            tb_writer.add_scalar(
                'train4_{}'.format(key), value, global_step)

        output_eval_file = os.path.join(args.output_dir, 'train_results.txt')
        file_mode = 'w' if not os.path.exists(output_eval_file) else 'a+'

        with open(output_eval_file, file_mode) as writer:
            logger.info('***** Train results *****')
            logger.info("  train loss: %s", str(step_loss/(nb_steps)))
            for key in sorted(result1.keys()):
                logger.info("1:  %s = %s", key, str(result1[key]))
                writer.write("1:  %s = %s\n" % (key, str(result1[key])))
                writer.write('-----------------\n')
            for key in sorted(result2.keys()):
                logger.info("2:  %s = %s", key, str(result2[key]))
                writer.write("2:  %s = %s\n" % (key, str(result2[key])))
                writer.write('-----------------\n')
            for key in sorted(result3.keys()):
                logger.info("3:  %s = %s", key, str(result3[key]))
                writer.write("3:  %s = %s\n" % (key, str(result3[key])))
                writer.write('------------------\n')
            for key in sorted(result4.keys()):
                logger.info("4:  %s = %s", key, str(result4[key]))
                writer.write("4:  %s = %s\n" % (key, str(result4[key])))
                writer.write('------------------\n')
            writer.write('\n')

        best_record_file = os.path.join(args.output_dir, 'best_eval_results.txt')
        with open(best_record_file, file_mode) as writer:
            logger.info('***** Best results *****')
            logger.info("best_epoch= %s", str(best_epoch))
            logger.info("best_ave_f1= %s", str(best_ave_f1))
            writer.write(("best_ave_f1= %s\n" % str(best_ave_f1)))
            for key in sorted(best_other_result.keys()):
                logger.info("%s = %s", key, str(best_other_result[key]))
                writer.write("%s = %s\n" % (key, str(best_other_result[key])))
                writer.write('-----------------\n')
            writer.write('\n')
        # tb_writer.close()
        best_test_record_file = os.path.join(args.output_dir, 'best_test_results.txt')
        with open(best_test_record_file, file_mode) as writer:
            logger.info('***** Test results *****')
            logger.info("test_ave_f1= %s", str(best_test_ave_f1))
            writer.write(("test_ave_f1= %s\n" % str(best_test_ave_f1)))
            for key in sorted(best_test_other_result.keys()):
                logger.info("%s = %s", key, str(best_test_other_result[key]))
                writer.write("%s = %s\n" % (key, str(best_test_other_result[key])))
                writer.write('-----------------\n')
            writer.write('\n')
    tb_writer.close()
    return global_step, tr_loss/global_step, all_eval_results


def evaluate(args, eval_dataset, model):
    results1, results2, results3, results4 = {}, {}, {}, {}

    args.eval_batch_size = args.all_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds1 = None
    preds2 = None
    preds3 = None
    preds4 = None
    out_label_ids1 = None
    out_label_ids2 = None
    out_label_ids3 = None
    out_label_ids4 = None

    for batch in eval_dataloader:
    # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, label1, label2, label3, label4 = get_input_from_batch(args, batch)

            logit1, logit2, logit3, logit4, _, _ = model(**inputs)
            tmp_eval_loss1 = F.cross_entropy(logit1, label1, reduction='sum')
            tmp_eval_loss2 = F.cross_entropy(logit2, label2, reduction='sum')
            tmp_eval_loss3 = F.cross_entropy(logit3, label3, reduction='sum')
            tmp_eval_loss4 = F.cross_entropy(logit4, label4, reduction='sum')

            batch_samples = label1.size(0)

            tmp_eval_loss = (tmp_eval_loss1 + tmp_eval_loss2 + tmp_eval_loss3 + tmp_eval_loss4) / (4.0 * batch_samples)

            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1
        if preds1 is None:
            preds1 = logit1.detach().cpu().numpy()
            preds2 = logit2.detach().cpu().numpy()
            preds3 = logit3.detach().cpu().numpy()
            preds4 = logit4.detach().cpu().numpy()
            out_label_ids1 = label1.detach().cpu().numpy()
            out_label_ids2 = label2.detach().cpu().numpy()
            out_label_ids3 = label3.detach().cpu().numpy()
            out_label_ids4 = label4.detach().cpu().numpy()
        else:
            preds1 = np.append(preds1, logit1.detach().cpu().numpy(), axis=0)
            preds2 = np.append(preds2, logit2.detach().cpu().numpy(), axis=0)
            preds3 = np.append(preds3, logit3.detach().cpu().numpy(), axis=0)
            preds4 = np.append(preds4, logit4.detach().cpu().numpy(), axis=0)
            out_label_ids1 = np.append(
                out_label_ids1, label1.detach().cpu().numpy(), axis=0)
            out_label_ids2 = np.append(
                out_label_ids2, label2.detach().cpu().numpy(), axis=0)
            out_label_ids3 = np.append(
                out_label_ids3, label3.detach().cpu().numpy(), axis=0)
            out_label_ids4 = np.append(
                out_label_ids4, label4.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds1 = np.argmax(preds1, axis=1)
    preds2 = np.argmax(preds2, axis=1)
    preds3 = np.argmax(preds3, axis=1)
    preds4 = np.argmax(preds4, axis=1)

    result1 = compute_metrics(preds1, out_label_ids1)
    result2 = compute_metrics(preds2, out_label_ids2)
    result3 = compute_metrics(preds3, out_label_ids3)
    result4 = compute_metrics(preds4, out_label_ids4)

    ave_f1 = (result1['f1'] + result2['f1']  + result3['f1'] + result4['f1'] ) / 4.0
    other_result = {'acc_1':result1['acc'], 'f1_1':result1['f1'],
                    'acc_2':result2['acc'], 'f1_2':result2['f1'],
                    'acc_3':result3['acc'], 'f1_3':result3['f1'],
                    'acc_4':result4['acc'], 'f1_4':result4['f1']}

    results1.update(result1)
    results2.update(result2)
    results3.update(result3)
    results4.update(result4)

    output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
    file_mode = 'w' if not os.path.exists(output_eval_file) else 'a+'
    with open(output_eval_file, file_mode) as writer:
        logger.info('***** Eval results *****')
        logger.info("  eval loss: %s", str(eval_loss))
        count = 0
        for key in sorted(result1.keys()):
            logger.info("1:  %s = %s", key, str(result1[key]))
            writer.write("1:  %s = %s\n" % (key, str(result1[key])))
            writer.write('-----------------\n')
        for key in sorted(result2.keys()):
            logger.info("2:  %s = %s", key, str(result2[key]))
            writer.write("2:  %s = %s\n" % (key, str(result2[key])))
            writer.write('-----------------\n')
        for key in sorted(result3.keys()):
            logger.info("3:  %s = %s", key, str(result3[key]))
            writer.write("3:  %s = %s\n" % (key, str(result3[key])))
            writer.write('------------------\n')
        for key in sorted(result4.keys()):
            logger.info("4:  %s = %s", key, str(result4[key]))
            writer.write("4:  %s = %s\n" % (key, str(result4[key])))
            writer.write('------------------\n')
        logger.info("ave_f1= %s", str(ave_f1))
        writer.write(("ave_f1= %s\n" % str(ave_f1)))
        writer.write('\n')
    return [results1, results2, results3, results4], eval_loss, ave_f1, other_result

def test(args, eval_dataset, model):
    results1, results2, results3, results4 = {}, {}, {}, {}

    args.eval_batch_size = args.all_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    # Eval
    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds1 = None
    preds2 = None
    preds3 = None
    preds4 = None
    out_label_ids1 = None
    out_label_ids2 = None
    out_label_ids3 = None
    out_label_ids4 = None

    for batch in eval_dataloader:
    # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, label1, label2, label3, label4 = get_input_from_batch(args, batch)

            logit1, logit2, logit3, logit4, _, _ = model(**inputs)

            tmp_eval_loss1 = F.cross_entropy(logit1, label1, reduction='sum')
            tmp_eval_loss2 = F.cross_entropy(logit2, label2, reduction='sum')
            tmp_eval_loss3 = F.cross_entropy(logit3, label3, reduction='sum')
            tmp_eval_loss4 = F.cross_entropy(logit4, label4, reduction='sum')

            batch_samples = label1.size(0)

            tmp_eval_loss = (tmp_eval_loss1 + tmp_eval_loss2 + tmp_eval_loss3 + tmp_eval_loss4) / (4.0 * batch_samples)

            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1
        if preds1 is None:
            preds1 = logit1.detach().cpu().numpy()
            preds2 = logit2.detach().cpu().numpy()
            preds3 = logit3.detach().cpu().numpy()
            preds4 = logit4.detach().cpu().numpy()
            out_label_ids1 = label1.detach().cpu().numpy()
            out_label_ids2 = label2.detach().cpu().numpy()
            out_label_ids3 = label3.detach().cpu().numpy()
            out_label_ids4 = label4.detach().cpu().numpy()
        else:
            preds1 = np.append(preds1, logit1.detach().cpu().numpy(), axis=0)
            preds2 = np.append(preds2, logit2.detach().cpu().numpy(), axis=0)
            preds3 = np.append(preds3, logit3.detach().cpu().numpy(), axis=0)
            preds4 = np.append(preds4, logit4.detach().cpu().numpy(), axis=0)
            out_label_ids1 = np.append(
                out_label_ids1, label1.detach().cpu().numpy(), axis=0)
            out_label_ids2 = np.append(
                out_label_ids2, label2.detach().cpu().numpy(), axis=0)
            out_label_ids3 = np.append(
                out_label_ids3, label3.detach().cpu().numpy(), axis=0)
            out_label_ids4 = np.append(
                out_label_ids4, label4.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds1 = np.argmax(preds1, axis=1)
    preds2 = np.argmax(preds2, axis=1)
    preds3 = np.argmax(preds3, axis=1)
    preds4 = np.argmax(preds4, axis=1)

    result1 = compute_metrics(preds1, out_label_ids1)
    result2 = compute_metrics(preds2, out_label_ids2)
    result3 = compute_metrics(preds3, out_label_ids3)
    result4 = compute_metrics(preds4, out_label_ids4)

    ave_f1 = (result1['f1'] + result2['f1']  + result3['f1'] + result4['f1'] ) / 4.0
    other_result = {'acc_1':result1['acc'], 'f1_1':result1['f1'],
                    'acc_2':result2['acc'], 'f1_2':result2['f1'],
                    'acc_3':result3['acc'], 'f1_3':result3['f1'],
                    'acc_4':result4['acc'], 'f1_4':result4['f1']}

    results1.update(result1)
    results2.update(result2)
    results3.update(result3)
    results4.update(result4)

    output_eval_file = os.path.join(args.output_dir, 'test_results.txt')
    file_mode = 'w' if not os.path.exists(output_eval_file) else 'a+'

    with open(output_eval_file, file_mode) as writer:
        logger.info('***** Test results *****')
        logger.info("  test loss: %s", str(eval_loss))
        count = 0
        for key in sorted(result1.keys()):
            logger.info("1:  %s = %s", key, str(result1[key]))
            writer.write("1:  %s = %s\n" % (key, str(result1[key])))
            writer.write('-----------------\n')
        for key in sorted(result2.keys()):
            logger.info("2:  %s = %s", key, str(result2[key]))
            writer.write("2:  %s = %s\n" % (key, str(result2[key])))
            writer.write('-----------------\n')
        for key in sorted(result3.keys()):
            logger.info("3:  %s = %s", key, str(result3[key]))
            writer.write("3:  %s = %s\n" % (key, str(result3[key])))
            writer.write('------------------\n')
        for key in sorted(result4.keys()):
            logger.info("4:  %s = %s", key, str(result4[key]))
            writer.write("4:  %s = %s\n" % (key, str(result4[key])))
            writer.write('------------------\n')
        logger.info("test_ave_f1= %s", str(ave_f1))
        writer.write(("test_ave_f1= %s\n" % str(ave_f1)))
        writer.write('\n')
    return [results1, results2, results3, results4], eval_loss, ave_f1, other_result


def get_labels(args, model, test_dataset):

    args.train_batch_size = args.all_gpu_train_batch_size
    args.eval_batch_size = args.all_gpu_eval_batch_size
    collate_fn = get_collate_fn(args)

    test_dataloader = DataLoader(test_dataset,
                                  batch_size=args.eval_batch_size,
                                  collate_fn=collate_fn)
    preds1 = None
    preds2 = None
    preds3 = None
    preds4 = None
    out_label_ids1 = None
    out_label_ids2 = None
    out_label_ids3 = None
    out_label_ids4 = None

    for batch in test_dataloader:
        # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, label1, label2, label3, label4 = get_input_from_batch(args, batch)

            logit1, logit2, logit3, logit4 = model(**inputs)

        if preds1 is None:
            preds1 = logit1.detach().cpu().numpy()
            preds2 = logit2.detach().cpu().numpy()
            preds3 = logit3.detach().cpu().numpy()
            preds4 = logit4.detach().cpu().numpy()
            out_label_ids1 = label1.detach().cpu().numpy()
            out_label_ids2 = label2.detach().cpu().numpy()
            out_label_ids3 = label3.detach().cpu().numpy()
            out_label_ids4 = label4.detach().cpu().numpy()
        else:
            preds1 = np.append(preds1, logit1.detach().cpu().numpy(), axis=0)
            preds2 = np.append(preds2, logit2.detach().cpu().numpy(), axis=0)
            preds3 = np.append(preds3, logit3.detach().cpu().numpy(), axis=0)
            preds4 = np.append(preds4, logit4.detach().cpu().numpy(), axis=0)
            out_label_ids1 = np.append(
                out_label_ids1, label1.detach().cpu().numpy(), axis=0)
            out_label_ids2 = np.append(
                out_label_ids2, label2.detach().cpu().numpy(), axis=0)
            out_label_ids3 = np.append(
                out_label_ids3, label3.detach().cpu().numpy(), axis=0)
            out_label_ids4 = np.append(
                out_label_ids4, label4.detach().cpu().numpy(), axis=0)

    preds1 = np.argmax(preds1, axis=1)
    preds2 = np.argmax(preds2, axis=1)
    preds3 = np.argmax(preds3, axis=1)
    preds4 = np.argmax(preds4, axis=1)

    return preds1,preds2, preds3, preds4, out_label_ids1, out_label_ids2, out_label_ids3, out_label_ids4

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1
    }

def compute_metrics(preds, labels):
    return acc_and_f1(preds, labels)
