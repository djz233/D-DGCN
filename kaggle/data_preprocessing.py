import csv
import re
import pickle
import tqdm
from collections import Counter
import random
import os

token = ''  # '' or '<mask>'
MBTIs = ('INTJ', 'INTP', 'INFP', 'ENTP', 'ISTP', 'ISFP', 'ESTJ', 'ISTJ', 'ESTP', 'ISFJ', 'ENFP', 'ESFP', 'ESFJ', 'ENFJ', 'INFJ', 'ENTJ')
def find_all_MBTIs(post, mbti):
    return [(match.start(), match.end()) for match in re.finditer(mbti, post)]

def save_kaggle(data_kaggle, path):
    '''
    '''
    kaggle = {'annotations':[], 'posts_text':[], 'posts_num':[]}
    for data in data_kaggle:
        raw_posts = patten_posts.findall(data[1])[0].split('|||')
        filter_posts = []
        for text in raw_posts:
            #delete http:/ 
            filter_text = ' '.join(filter(lambda x : x[:4] != 'http', text.split(' ')))
            if filter_text != '':
                #delete and MBTI
                for MBTI in MBTIs:
                    mbti_idx_list = find_all_MBTIs(filter_text.lower(), MBTI.lower())
                    delete_idx = 0
                    for start, end in mbti_idx_list:
                        filter_text = filter_text[:start - delete_idx] + token + filter_text[end - delete_idx:]
                        delete_idx += end - start + len(token)

                filter_posts.append(filter_text)
    
        kaggle['annotations'].append(data[0])
        kaggle['posts_text'].append(filter_posts)
        kaggle['posts_num'].append(len(filter_posts))
    print(kaggle['posts_text'][:1])
    print(len(kaggle['annotations']))
    print(Counter(kaggle['posts_num']))

    with open(os.path.join('kaggle', path + '.pkl'), 'wb') as f:
        pickle.dump(kaggle, f)

def data_preprocessing():
    '''
    '''
    with open('MBTI_kaggle.csv', 'r', encoding = 'utf-8') as f:
        f_csv = list(csv.reader(f))[1:]
        random.shuffle(f_csv)
        train_len = int(0.6 * len(f_csv))
        eval_len = int(0.2 * len(f_csv)) 
        train_kaggle, eval_kaggle, test_kaggle = f_csv[:train_len], f_csv[train_len:train_len + eval_len], f_csv[train_len + eval_len:] 
        save_kaggle(train_kaggle, 'train')
        save_kaggle(eval_kaggle, 'eval')
        save_kaggle(test_kaggle, 'test')

if __name__ == '__main__':
    
    random.seed(0)
    #patten_type = re.compile(r'([IESNTFJP]{4}),')
    patten_posts = re.compile(r'\"{0,1}\'{0,1}(.*)\'{0,1}\"{0,1}')
    data_preprocessing()
