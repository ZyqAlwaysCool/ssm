#!/usr/bin/env python
# coding: utf-8
# == use jupyter ==

# In[1]:


import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import json
import pickle
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
#from bert4keras.snippets import open
from keras.layers import Dropout, Dense


# In[2]:


dataset_path = './diseasekg_dataset/diseasekg_data_noise_30.pkl'


# In[3]:


def get_data_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


# In[4]:


import pickle as pkl

def to_pkl(fpath, data_lst):
    with open(fpath, 'wb') as fin:
        pkl.dump(data_lst, fin)

def from_pkl(fpath):
    with open(fpath, 'rb') as fout:
        return pkl.load(fout)


# In[5]:


rel2nlp = dict(推荐食谱='的推荐食谱是', 
               忌吃='忌吃', 
               宜吃='宜吃', 
               属于='属于', 
               所属科室='所属科室是', 
               常用药品='的常用药品是', 
               生产药品='的生产药品是', 
               好评药品='的好评药品是', 
               诊断检查='的诊断检查方式是', 
               症状='的症状是', 
               并发症='的并发症是', 
               治疗方法='的治疗方法是')


# In[6]:


dataset_entities = './diseasekg_dataset/diseaseKG_entities.json'
dataset_relations = './diseasekg_dataset/diseaseKG_relations.json'
disease_ent = get_data_from_json(dataset_entities)
disease_rel = get_data_from_json(dataset_relations)


# In[7]:

# In[8]:


ent_type


# In[9]:


conf_train2id, conf_test2id, conf_valid2id, gold_train2id, gold_test2id, gold_valid2id, ent_vocab, rel_vocab, all_triples, pos_triples, neg_triples = from_pkl(dataset_path)


# In[10]:


len(conf_train2id), len(conf_test2id), len(conf_valid2id)


# In[11]:


train = []
valid = []
test = []

for ct in conf_train2id:
    tmp = []
    tmp.append(ent_vocab['id2ent'][ct[0]] + rel2nlp[rel_vocab['id2rel'][ct[2]]] + ent_vocab['id2ent'][ct[1]])
    if ct[3] == -1:
        tmp.append(0)
    else:
        tmp.append(ct[3])
    train.append(tmp)

for cv in conf_valid2id:
    tmp = []
    tmp.append(ent_vocab['id2ent'][cv[0]] + rel2nlp[rel_vocab['id2rel'][cv[2]]] + ent_vocab['id2ent'][cv[1]])
    if cv[3] == -1:
        tmp.append(0)
    else:
        tmp.append(cv[3])
    valid.append(tmp)

for ct in conf_test2id:
    tmp = []
    tmp.append(ent_vocab['id2ent'][ct[0]] + rel2nlp[rel_vocab['id2rel'][ct[2]]] + ent_vocab['id2ent'][ct[1]])
    if ct[3] == -1:
        tmp.append(0)
    else:
        tmp.append(ct[3])
    test.append(tmp)


# ## 训练模型

# In[13]:


set_gelu('tanh')  # 切换gelu版本

maxlen = 128
batch_size = 64
config_path = '../bert_params/publish/bert_config.json'
checkpoint_path = '../bert_params/publish/bert_model.ckpt'
dict_path = '../bert_params/publish/vocab.txt'


def load_data(data):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    for d in data:
        D.append((d[0], int(d[1])))
    return D


# 加载数据集
train_data = load_data(train)
valid_data = load_data(valid)
test_data = load_data(test)


# In[14]:


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


# In[15]:


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text1, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# In[16]:


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    return_keras_model=False,
)

output = Dropout(rate=0.1)(bert.model.output)
output = Dense(
    units=2, activation='softmax', kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5), 
    metrics=['accuracy'],
)


# In[17]:


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


# In[18]:


weights_name = '../bert_weights/best_model_diseasekg_new_sentence_1_noise_30.weights'


# In[19]:


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights(weights_name)
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


# In[20]:


import tensorflow as tf

tf.test.is_gpu_available()


# In[21]:


evaluator = Evaluator()

model.fit(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=20,
    callbacks=[evaluator]
)

model.load_weights(weights_name)
print(u'final test acc: %05f\n' % (evaluate(test_generator)))


# In[22]:


model.load_weights(weights_name)
print(u'final test acc: %05f\n' % (evaluate(test_generator)))


# In[23]:


from tqdm import tqdm


# In[24]:


def evaluate2(data):
    total, right = 0., 0.
    neg_wrong_lst = []
    pos_wrong_lst = []
    y_pred_lst = []
    y_true_lst = []
    for x_true, y_true in tqdm(data):
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
        for i in range(len(y_true)):
            y_pred_lst.append(model.predict(x_true)[i])
            y_true_lst.append(y_true[i])
            if y_true[i] == 0 and y_pred[i] == 1:
                neg_wrong_lst.append(tokenizer.decode(x_true[0][i]))
            if y_true[i] == 1 and y_pred[i] == 0:
                pos_wrong_lst.append(tokenizer.decode(x_true[0][i]))
    return right / total, neg_wrong_lst, y_pred_lst, y_true_lst, pos_wrong_lst


# In[25]:


acc2, neg_wrong_ls, y_pred_lst, y_true_lst, pos_wrong_lst = evaluate2(test_generator)


# In[26]:


print(acc2)


# In[27]:


len(neg_wrong_ls), len(pos_wrong_lst)


# In[28]:


to_pkl('../融合模型中各模型预测结果/bert_diseasekg_new_test_pred_true_noise_30.pkl', 
       [y_pred_lst, y_true_lst])


# In[ ]:




