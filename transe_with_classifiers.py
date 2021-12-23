#!/usr/bin/env python
# coding: utf-8
# ===use jupyter===

# # 图嵌入模型+分类器结果
# # diseasekg-N3

# In[1]:


from tqdm import tqdm
import pickle as pkl
import json
import numpy as np
import pandas as pd


# ## 读取、存储pkl文件

# In[2]:


def to_pkl(fpath, data_lst):
    with open(fpath, 'wb') as fin:
        pkl.dump(data_lst, fin)

def from_pkl(fpath):
    with open(fpath, 'rb') as fout:
        return pkl.load(fout)


# In[ ]:


# pos_triples, neg_triples, conf_train2id, conf_test2id, conf_valid2id, gold_train2id, gold_test2id, gold_valid2id, ent_vocab, rel_vocab, ent_type, rel, diakg, ent2type = from_pkl('./diakg/dataset/diakg-dataset.pkl')


# ## 接入openKE，生成embedding

# In[3]:


def to_txt_embed(file, data, display=True):
    with open(file, 'w', encoding='utf-8') as f:
        for i in data:
            if type(i) == int:
                f.write(str(i))
                f.write('\n')
            else:
                f.write('{} {} {}'.format(i[0], i[1], i[2]))
                f.write('\n')
    if display:
        print('已导出至: {}'.format(file))

def to_txt_dct_embed(file, data, data_nums, display=True):
    with open(file, 'w', encoding='utf-8') as f:
        f.write(str(data_nums))
        f.write('\n')
        for k, v in data.items():
            f.write('{}\t{}'.format(k, v))
            f.write('\n')
    if display:
        print('已导出至: {}'.format(file))


# In[4]:


embedding_path = '../OpenKE/diseasekg_new/'

def to_embedding_txt(embedding_path, gold_train2id, gold_test2id):
    train2id = [len(gold_train2id)] + gold_train2id
    test2id = [len(gold_test2id)] + gold_test2id
    #valid2id = [len(gold_valid2id)] + gold_valid2id
    ent2id = ent_vocab['ent2id']
    rel2id = rel_vocab['rel2id']

    to_txt_embed(embedding_path+'train2id.txt', train2id)
    to_txt_embed(embedding_path+'test2id.txt', test2id)
    #to_txt_embed(embedding_path+'valid2id.txt', valid2id)

    to_txt_dct_embed(embedding_path+'entity2id.txt', ent2id, len(ent2id))
    to_txt_dct_embed(embedding_path+'relation2id.txt', rel2id, len(rel2id))


# In[5]:


def loadJson(file):
    with open(file, 'r') as f:
        return json.load(f)

def loadVector(file):
    if file.split('/')[-1] == 'rescal.json':
        tmp = loadJson(file)
        ent_param = tmp['ent_embeddings.weight']
        rel_param = [i[:100] for i in tmp['rel_matrices.weight']]
        return ent_param, rel_param
    if file.split('/')[-1] == 'complEx.json':
        tmp = loadJson(file)
        ent_param = list(map(lambda x,y: (x+y).tolist(), np.array(tmp['ent_re_embeddings.weight']), np.array(tmp['ent_im_embeddings.weight'])))
        rel_param = list(map(lambda x,y: (x+y).tolist(), np.array(tmp['rel_re_embeddings.weight']), np.array(tmp['rel_im_embeddings.weight'])))
        return ent_param, rel_param
    if file.split('/')[-1] == 'analogy.json':
        tmp = loadJson(file)
        ent_param = list(map(lambda x,y: (x+y).tolist()[:100], np.array(tmp['ent_re_embeddings.weight']), np.array(tmp['ent_im_embeddings.weight'])))
        rel_param = list(map(lambda x,y: (x+y).tolist()[:100], np.array(tmp['rel_re_embeddings.weight']), np.array(tmp['rel_im_embeddings.weight'])))
        return ent_param, rel_param
    ent_param = loadJson(file)['ent_embeddings.weight']
    rel_param = loadJson(file)['rel_embeddings.weight']
    return ent_param, rel_param

def toVectorFile(path, mapping, data):
    with open(path, 'w', encoding='utf-8') as f:
        for i in tqdm(range(len(data))):
            f.write('{}\t{}\n'.format(mapping[i], data[i]))


# In[6]:


def load_vec(fname, vocab, k=100):
    '''

    :param fname: word2vec的文件名称
    :param vocab: 词典（包括实体词典和关系词典）
    :param k: 设置默认维度
    :return: k: w2v维度 W: w2v矩阵
    '''
    f = open(fname)
    w2v={}
    W = np.zeros(shape=(vocab.__len__() + 2, k))
    unknowtoken = 0
    error_count = 0
    for line in f:
        try:
            values = line.split('\t')
            word = values[0]
            values[1] = values[1][1:-2].split(',')
            coefs = np.asarray(values[1], dtype='float32')
            w2v[word] = coefs
        except:
            error_count += 1
    f.close()
    w2v["**UNK**"] = np.random.uniform(-0.25, 0.25, k)

    for word in vocab:
        if not w2v.__contains__(word):
            w2v[word] = w2v["**UNK**"]
            unknowtoken +=1
            W[vocab[word]] = w2v[word]
        else:
            W[vocab[word]] = w2v[word]

    print('!!!!!! UnKnown tokens in w2v', unknowtoken)
    print('error_count: {}'.format(error_count))
    return k, W


fpath = './diseasekg_dataset/diseasekg_data_1.pkl'

conf_train2id, conf_test2id, conf_valid2id, gold_train2id, gold_test2id, gold_valid2id, ent_vocab, rel_vocab, all_triples,pos_triples,neg_triples = from_pkl(fpath)


# In[8]:


ent_embedd_path = '../kg_embedding/diseasekg_new/transe_ent_diseasekg_new.txt'
rel_embedd_path = '../kg_embedding/diseasekg_new/transe_rel_diseasekg_new.txt'

kg_ent_vec = load_vec(ent_embedd_path, ent_vocab['ent2id'])[1]
kg_rel_vec = load_vec(rel_embedd_path, rel_vocab['rel2id'])[1]


# ## 分类器

# In[9]:


x_train = [list(kg_ent_vec[i[0]]) + list(kg_rel_vec[i[2]]) + list(kg_ent_vec[i[1]]) for i in conf_train2id]
y_train = [i[3] for i in conf_train2id]

x_test = [list(kg_ent_vec[i[0]]) + list(kg_rel_vec[i[2]]) + list(kg_ent_vec[i[1]]) for i in conf_test2id]
y_test = [i[3] for i in conf_test2id]

x_valid = [list(kg_ent_vec[i[0]]) + list(kg_rel_vec[i[2]]) + list(kg_ent_vec[i[1]]) for i in conf_valid2id]
y_valid = [i[3] for i in conf_valid2id]


# In[10]:


len(x_train), len(x_test), len(x_valid)



# In[12]:


from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, auc


# In[13]:


from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,RandomForestRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import xgboost
from xgboost import XGBClassifier

# 机器学习分类器
# rfc = RandomForestClassifier(n_estimators=180)
# bc = BaggingClassifier(n_estimators =25)
# knc = KNeighborsClassifier(n_neighbors =7,weights ='distance')
# svc = SVC(C=20,kernel = 'rbf',gamma='auto')
# qda = QuadraticDiscriminantAnalysis()
# xgb = XGBClassifier()


# In[ ]:


from sklearn.model_selection import GridSearchCV


def grid(clf, x_prove, y_prove, tuned_parameter):
    rfc = GridSearchCV(estimator=clf, param_grid=tuned_parameter, cv=5, n_jobs=1, scoring='accuracy')
    rfc.fit(x_prove, y_prove)

    print(rfc.best_params_)


# In[ ]:


grid(rfc, x_valid, y_valid, [{'max_features': [2, 'auto', 'log2'], 'n_estimators': [100, 200, 1000]}])


# In[ ]:


grid(bc, x_valid, y_valid, [{'n_estimators': [10, 50, 100, 200, 1000]}])


# In[ ]:


grid(knc, x_valid, y_valid, [{'n_neighbors': [10, 50, 100, 200, 1000], 'weights': ['distance']}])


# In[ ]:


grid(svc, x_valid, y_valid, [{'kernel': ['rbf', 'sigmoid'], 'gamma': ['auto'], 'C': [10, 20, 30, 50, 100, 200]}])


# In[ ]:


grid(rfc, x_valid, y_valid, [{'n_estimators': [100, 200, 300, 400, 500, 1000, 1100, 1200, 1300, 1400]}])


# In[14]:


# 机器学习分类器
rfc = RandomForestClassifier(max_features=2, n_estimators=1000)
knc = KNeighborsClassifier(n_neighbors =50,weights ='distance')
svc = SVC(C=200,kernel = 'rbf',gamma='auto', probability=True)
xgb = XGBClassifier(n_estimators=200, probability=True)


# In[15]:


from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

cls_cnt = 1
ml_cls_res = defaultdict(list)
cls_pred = []
cls_pred_probs = []

for cls in [rfc, knc, svc, xgb]:
    cls.fit(x_train, y_train)
    y_pred = cls.predict(x_test)
    predictions = [round(value) for value in y_pred]
    acc1 = accuracy_score(y_test, predictions)
    neg_pre_right_cnt, neg_pre_right_idx = 0, [] # 噪声样本预测正确数量、idx
    neg_pre_wrong_cnt, neg_pre_wrong_idx = 0, [] # 噪声样本预测错误数量、idx
    pos_pre_right_cnt, pos_pre_right_idx = 0, []
    pos_pre_wrong_cnt, pos_pre_wrong_idx = 0, []
    cls_pred.append(predictions)
    y_pred_probs = cls.predict_proba(x_test)
    cls_pred_probs.append(y_pred_probs)
    for i in tqdm(range(len(predictions))):
        if y_test[i] == 0 and predictions[i] == 0:
            neg_pre_right_cnt += 1
            neg_pre_right_idx.append(i)
        if y_test[i] == 0 and predictions[i] == 1:
            neg_pre_wrong_cnt += 1
            neg_pre_wrong_idx.append(i)
        if y_test[i] == 1 and predictions[i] == 1:
            pos_pre_right_cnt += 1
            pos_pre_right_idx.append(i)
        if y_test[i] == 1 and predictions[i] == 0:
            pos_pre_wrong_cnt += 1
            pos_pre_wrong_idx.append(i)
    ml_cls_res[cls_cnt].append(neg_pre_right_idx)
    ml_cls_res[cls_cnt].append(neg_pre_wrong_idx)
    ml_cls_res[cls_cnt].append(pos_pre_right_idx)
    ml_cls_res[cls_cnt].append(pos_pre_wrong_idx)
    cls_cnt += 1
    acc2 = neg_pre_right_cnt / (neg_pre_right_cnt + neg_pre_wrong_cnt)
    acc3 = pos_pre_right_cnt / (pos_pre_right_cnt + pos_pre_wrong_cnt)
    print('acc_1(pos+neg): {} acc_2(only neg): {} acc_3(only pos): {}'.format(acc1, acc2, acc3))
    print(np.mean(precision_recall_fscore_support(y_test, predictions, labels = [0, 1]), axis=1))
    print('-----')


# In[16]:


to_pkl('./diseasekg_new_noise_50_rfc_bc_knc_svc_qda_xgb_pre_lst.pkl', [cls_pred, cls_pred_probs])


# In[17]:


def my_frr(y_true_lst, m_pred_lst):
    tp, fn = 0, 0
    for i in range(len(y_true_lst)):
        if y_true_lst[i] == 1 and m_pred_lst[i] == 1:
            tp += 1
        if y_true_lst[i] == 1 and m_pred_lst[i] == 0:
            fn += 1
    return fn / (tp + fn)


# In[18]:


for i in range(len(cls_pred)):
    print(my_frr(y_test, cls_pred[i]))


# In[19]:


len(cls_pred_probs)


# In[ ]:




