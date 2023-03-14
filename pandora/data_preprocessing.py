import csv
import re
import pickle
import tqdm
from collections import Counter
import random
import os
import pandas as pd
import numpy as np

MBTIs = ('INTJ', 'INTP', 'INFP', 'ENTP', 'ISTP', 'ISFP', 'ESTJ', 'ISTJ', 'ESTP', 'ISFJ', 'ENFP', 'ESFP', 'ESFJ', 'ENFJ', 'INFJ', 'ENTJ','TYPE_MENTION', 'COGFUNC_MENTION')
def get_user_label():
    data = pd.read_csv('pandora_comments/author_profiles.csv')
    user = list(data['author'])
    label = list(data['mbti'])
    user_dict = {}
    for i,t in enumerate(label):
        if str(t).upper() not in MBTIs:
            continue
        else:
            user_dict[user[i]]= t
    return user_dict

user_dict = get_user_label()

data = pd.read_csv('pandora_comments/all_comments_since_2015.csv')
user = list(data['author'])
text = list(data['body'])

token = ''  # '' or '<mask>'

def find_all_MBTIs(post, mbti):
    return [(match.start(), match.end()) for match in re.finditer(mbti, post)]

reg_link = re.compile('http\S+', flags=re.MULTILINE)

posts = {'annotations':[], 'posts_text':[], 'posts_num':[], 'max_len':[], 'users':[]}
# recorded_user = []
for i,t in enumerate(user):
    if t in user_dict.keys():
        if t not in posts['users']: # new user
            posts['users'].append(t)
            posts['posts_text'].append([])
            posts['annotations'].append(user_dict[t].upper())
            posts['max_len'].append(0)
            posts['posts_num'].append(0)

        user_id = posts['users'].index(t)

        filter_text = reg_link.sub('', text[i])
        if filter_text != '':
            # delete and MBTI
            for MBTI in MBTIs:
                mbti_idx_list = find_all_MBTIs(filter_text.lower(), MBTI.lower())  
                delete_idx = 0
                for start, end in mbti_idx_list:
                    filter_text = filter_text[:start - delete_idx] + token + filter_text[end - delete_idx:]
                    delete_idx += end - start + len(token)
            post_len = len(filter_text.split(' '))
            if post_len > 5:
                posts['posts_text'][user_id].append(filter_text)
                if posts['max_len'][user_id] < post_len:
                    posts['max_len'][user_id] = post_len
                posts['posts_num'][user_id] += 1
            else:
                continue

label_lookup = {'E': 0, 'I': 1, 'S': 0, 'N':1, 'T': 0, 'F': 1, 'J': 0, 'P':1}
types = posts['annotations']

label0, label1, label2, label3 = [],[],[],[]
for type in types:
    label0.append(label_lookup[list(type)[0]])
    label1.append(label_lookup[list(type)[1]])
    label2.append(label_lookup[list(type)[2]])
    label3.append(label_lookup[list(type)[3]])

save_data= pd.DataFrame(posts['annotations'],columns=['annotations'])
save_data['posts_num'] = posts['posts_num']
save_data['max_len'] = posts['max_len']
save_data['I_E'] = label0
save_data['N_S'] = label1
save_data['F_T'] = label2
save_data['P_J'] = label3
save_data.to_excel('data_statistic.xlsx')

with open('pandora.pkl', 'wb') as f:
    pickle.dump(posts, f)












