import pickle
import os
import random

def data_split(file, result_path, top):
    with open(file, 'rb') as f:
        pandora = pickle.load(f)
    annotations = pandora['annotations']
    posts_text = pandora['posts_text']
    posts_num =  pandora['posts_num']
    user_num = len(annotations)
    user_ids = list(range(user_num))
    random.shuffle(user_ids)

    train_ids, eval_ids, test_ids = user_ids[:int(0.6 * user_num)], user_ids[int(0.6 * user_num):int(0.8 * user_num)], user_ids[int(0.8 * user_num):]
    train_text, train_num, train_annotations = [keep_top(posts_text[i], top) for i in train_ids],[min(posts_num[i], top) for i in train_ids],[annotations[i] for i in train_ids]
    eval_text, eval_num, eval_annotations = [keep_top(posts_text[i], top) for i in eval_ids],[min(posts_num[i], top) for i in eval_ids],[annotations[i] for i in eval_ids]
    test_text, test_num, test_annotations = [keep_top(posts_text[i], top) for i in test_ids],[min(posts_num[i], top)for i in test_ids],[annotations[i] for i in test_ids]

    save_data(train_text, train_num, train_annotations, result_path, 'train')
    save_data(eval_text, eval_num, eval_annotations, result_path, 'eval')
    save_data(test_text, test_num, test_annotations, result_path, 'test')

def keep_top(data, k, recent = True):
    if len(data) <= k:
        return data
    else:
        if recent:
            return data[-k:]
        else:
            return data[:k]

def save_data(posts_text, posts_num, annotations, result_path, option):    
    with open(os.path.join(result_path, option + '.pkl'), 'wb') as f:
        data = {'posts_text':posts_text, 'posts_num':posts_num, 'annotations':annotations}
        pickle.dump(data, f)

if __name__ == '__main__':
    top = 2000
    random.seed(0)
    file = 'pandora.pkl'
    result_path = 'pandora'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    data_split(file, result_path, top)
