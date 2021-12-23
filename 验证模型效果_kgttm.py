#!/usr/bin/env python
# coding: utf-8
# ===use jupyter===

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[2]:


import pickle
import numpy as np
from keras.layers.core import Dropout,Flatten
from keras.layers.merge import concatenate
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, SimpleRNN, RepeatVector,add,subtract,dot
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras import regularizers
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import keras


# In[3]:


def to_pickle(file, data, display=False):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
        if display:
            print('已导出至：{}'.format(file))

def from_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f) 


# In[4]:


def get_model_origin(entvocabsize, relvocabsize, ent2vec, rel2vec, input_path_lenth,
                          ent_emd_dim, rel_emd_dim):
    
    ent_h_input = Input(shape=(1,), dtype='int32')
    ent_h_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=1,
                                   mask_zero=False, trainable=False, weights=[ent2vec])(ent_h_input)

    ent_t_input = Input(shape=(1,), dtype='int32')
    ent_t_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=1,
                                   mask_zero=False, trainable=False, weights=[ent2vec])(ent_t_input)

    rel_r_input = Input(shape=(1,), dtype='int32')
    rel_r_embedding = Embedding(input_dim=relvocabsize+2, output_dim=rel_emd_dim, input_length=1,
                                   mask_zero=False, trainable=False, weights=[rel2vec])(rel_r_input)


    path_h_input = Input(shape=(input_path_lenth,), dtype='int32')
    path_h_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path_h_input)

    path_t_input = Input(shape=(input_path_lenth,), dtype='int32')
    path_t_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path_t_input)

    path_r_input = Input(shape=(input_path_lenth,), dtype='int32')
    path_r_embedding = Embedding(input_dim=relvocabsize+2, output_dim=rel_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[rel2vec])(path_r_input)

    path2_h_input = Input(shape=(input_path_lenth,), dtype='int32')
    path2_h_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path2_h_input)

    path2_t_input = Input(shape=(input_path_lenth,), dtype='int32')
    path2_t_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path2_t_input)

    path2_r_input = Input(shape=(input_path_lenth,), dtype='int32')
    path2_r_embedding = Embedding(input_dim=relvocabsize+2, output_dim=rel_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[rel2vec])(path2_r_input)

    path3_h_input = Input(shape=(input_path_lenth,), dtype='int32')
    path3_h_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path3_h_input)

    path3_t_input = Input(shape=(input_path_lenth,), dtype='int32')
    path3_t_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path3_t_input)

    path3_r_input = Input(shape=(input_path_lenth,), dtype='int32')
    path3_r_embedding = Embedding(input_dim=relvocabsize+2, output_dim=rel_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[rel2vec])(path3_r_input)

    ent_h_embedding = Flatten()(ent_h_embedding)
    ent_t_embedding = Flatten()(ent_t_embedding)
    rel_r_embedding = Flatten()(rel_r_embedding)
    ent_h_embedding = RepeatVector(input_path_lenth)(ent_h_embedding)
    ent_t_embedding = RepeatVector(input_path_lenth)(ent_t_embedding)
    rel_r_embedding = RepeatVector(input_path_lenth)(rel_r_embedding)

    path_embedding = concatenate([ent_h_embedding, rel_r_embedding, ent_t_embedding,
                                  path_h_embedding,
                                  path_r_embedding,
                                  path_t_embedding], axis=-1)
    path_embedding = Dropout(0.5)(path_embedding)

    path_LSTM = SimpleRNN(100, return_sequences=False)(path_embedding)
    path_LSTM = BatchNormalization()(path_LSTM)
    
    path_LSTM = Dropout(0.5)(path_LSTM)

    path_value = Dense(1, activation='sigmoid')(path_LSTM)

    # -----------------
    path2_embedding = concatenate([ent_h_embedding, rel_r_embedding, ent_t_embedding,
                                  path2_h_embedding,
                                  path2_r_embedding,
                                  path2_t_embedding], axis=-1)
    path2_embedding = Dropout(0.5)(path2_embedding)

    path2_LSTM = SimpleRNN(100, return_sequences=False)(path2_embedding)
    path2_LSTM = BatchNormalization()(path2_LSTM)
    path2_LSTM = Dropout(0.5)(path2_LSTM)
    path2_value = Dense(1, activation='sigmoid')(path2_LSTM)
    # ------------------
    path3_embedding = concatenate([ent_h_embedding, rel_r_embedding, ent_t_embedding,
                                  path3_h_embedding,
                                  path3_r_embedding,
                                  path3_t_embedding], axis=-1)
    path3_embedding = Dropout(0.5)(path3_embedding)

    path3_LSTM = SimpleRNN(100, return_sequences=False)(path3_embedding)
    path3_LSTM = BatchNormalization()(path3_LSTM)
    path3_LSTM = Dropout(0.5)(path3_LSTM)
    path3_value = Dense(1, activation='sigmoid')(path3_LSTM)
    # ------------------


    TransE_input = Input(shape=(1,), dtype='float32')
    
    # 实体评估器特征输入 原有是6个特征
    RRank_input = Input(shape=(6,), dtype='float32')
    
    # 1、实体评估器特征加入特征向量中心性特征，加入头尾实体特征向量中心性
#     RRank_input = Input(shape=(8,), dtype='float32')

    RRank_hinden = Dense(100, activation='tanh')(RRank_input)
    RRank_hinden = Dropout(0.5)(RRank_hinden)
    RRank_value = Dense(1, activation='sigmoid')(RRank_hinden)

    BP_input = concatenate([
        path_value, path2_value, path3_value,
        TransE_input,
        RRank_value
    ], axis=-1)


    BP_hidden = Dense(50)(BP_input)
    BP_hidden = Dropout(0.5)(BP_hidden)
    model = Dense(2, activation='softmax')(BP_hidden)
    # model = Dense(1, activation='tanh')(BP_hidden)

    Models = Model([
        ent_h_input, ent_t_input, rel_r_input,
        path_h_input, path_t_input, path_r_input,
        path2_h_input, path2_t_input, path2_r_input,
        path3_h_input, path3_t_input, path3_r_input,
        TransE_input,
        RRank_input
    ], model)

    Models.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return Models


# In[5]:


def get_model_1(entvocabsize, relvocabsize, ent2vec, rel2vec, input_path_lenth,
                          ent_emd_dim, rel_emd_dim):
    
    ent_h_input = Input(shape=(1,), dtype='int32')
    ent_h_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=1,
                                   mask_zero=False, trainable=False, weights=[ent2vec])(ent_h_input)

    ent_t_input = Input(shape=(1,), dtype='int32')
    ent_t_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=1,
                                   mask_zero=False, trainable=False, weights=[ent2vec])(ent_t_input)

    rel_r_input = Input(shape=(1,), dtype='int32')
    rel_r_embedding = Embedding(input_dim=relvocabsize+2, output_dim=rel_emd_dim, input_length=1,
                                   mask_zero=False, trainable=False, weights=[rel2vec])(rel_r_input)


    path_h_input = Input(shape=(input_path_lenth,), dtype='int32')
    path_h_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path_h_input)

    path_t_input = Input(shape=(input_path_lenth,), dtype='int32')
    path_t_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path_t_input)

    path_r_input = Input(shape=(input_path_lenth,), dtype='int32')
    path_r_embedding = Embedding(input_dim=relvocabsize+2, output_dim=rel_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[rel2vec])(path_r_input)

    path2_h_input = Input(shape=(input_path_lenth,), dtype='int32')
    path2_h_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path2_h_input)

    path2_t_input = Input(shape=(input_path_lenth,), dtype='int32')
    path2_t_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path2_t_input)

    path2_r_input = Input(shape=(input_path_lenth,), dtype='int32')
    path2_r_embedding = Embedding(input_dim=relvocabsize+2, output_dim=rel_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[rel2vec])(path2_r_input)

    path3_h_input = Input(shape=(input_path_lenth,), dtype='int32')
    path3_h_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path3_h_input)

    path3_t_input = Input(shape=(input_path_lenth,), dtype='int32')
    path3_t_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path3_t_input)

    path3_r_input = Input(shape=(input_path_lenth,), dtype='int32')
    path3_r_embedding = Embedding(input_dim=relvocabsize+2, output_dim=rel_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[rel2vec])(path3_r_input)

    ent_h_embedding = Flatten()(ent_h_embedding)
    ent_t_embedding = Flatten()(ent_t_embedding)
    rel_r_embedding = Flatten()(rel_r_embedding)
    ent_h_embedding = RepeatVector(input_path_lenth)(ent_h_embedding)
    ent_t_embedding = RepeatVector(input_path_lenth)(ent_t_embedding)
    rel_r_embedding = RepeatVector(input_path_lenth)(rel_r_embedding)

    path_embedding = concatenate([ent_h_embedding, rel_r_embedding, ent_t_embedding,
                                  path_h_embedding,
                                  path_r_embedding,
                                  path_t_embedding], axis=-1)
    path_embedding = Dropout(0.5)(path_embedding)

    path_LSTM = SimpleRNN(100, return_sequences=False)(path_embedding)
    path_LSTM = BatchNormalization()(path_LSTM)
    
    path_LSTM = Dropout(0.5)(path_LSTM)

    path_value = Dense(1, activation='sigmoid')(path_LSTM)

    # -----------------
    path2_embedding = concatenate([ent_h_embedding, rel_r_embedding, ent_t_embedding,
                                  path2_h_embedding,
                                  path2_r_embedding,
                                  path2_t_embedding], axis=-1)
    path2_embedding = Dropout(0.5)(path2_embedding)

    path2_LSTM = SimpleRNN(100, return_sequences=False)(path2_embedding)
    path2_LSTM = BatchNormalization()(path2_LSTM)
    path2_LSTM = Dropout(0.5)(path2_LSTM)
    path2_value = Dense(1, activation='sigmoid')(path2_LSTM)
    # ------------------
    path3_embedding = concatenate([ent_h_embedding, rel_r_embedding, ent_t_embedding,
                                  path3_h_embedding,
                                  path3_r_embedding,
                                  path3_t_embedding], axis=-1)
    path3_embedding = Dropout(0.5)(path3_embedding)

    path3_LSTM = SimpleRNN(100, return_sequences=False)(path3_embedding)
    path3_LSTM = BatchNormalization()(path3_LSTM)
    path3_LSTM = Dropout(0.5)(path3_LSTM)
    path3_value = Dense(1, activation='sigmoid')(path3_LSTM)
    # ------------------


    TransE_input = Input(shape=(1,), dtype='float32')
    
    # 实体评估器特征输入 原有是6个特征
    # RRank_input = Input(shape=(6,), dtype='float32')
    
    # 实体评估器特征扩充至14种
    RRank_input = Input(shape=(12,), dtype='float32')

    RRank_hinden = Dense(100, activation='tanh')(RRank_input)
    RRank_hinden = Dropout(0.5)(RRank_hinden)
    RRank_value = Dense(1, activation='sigmoid')(RRank_hinden)

    BP_input = concatenate([
        path_value, path2_value, path3_value,
        TransE_input,
        RRank_value
    ], axis=-1)


    BP_hidden = Dense(50)(BP_input)
    BP_hidden = Dropout(0.5)(BP_hidden)
    model = Dense(2, activation='softmax')(BP_hidden)
    # model = Dense(1, activation='tanh')(BP_hidden)

    Models = Model([
        ent_h_input, ent_t_input, rel_r_input,
        path_h_input, path_t_input, path_r_input,
        path2_h_input, path2_t_input, path2_r_input,
        path3_h_input, path3_t_input, path3_r_input,
        TransE_input,
        RRank_input
    ], model)

    Models.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return Models


# In[6]:


def get_model_2(entvocabsize, relvocabsize, ent2vec, rel2vec, input_path_lenth,
                          ent_emd_dim, rel_emd_dim):
    
    ent_h_input = Input(shape=(1,), dtype='int32')
    ent_h_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=1, name="h_emb",
                                   mask_zero=False, trainable=False, weights=[ent2vec])(ent_h_input)

    ent_t_input = Input(shape=(1,), dtype='int32')
    ent_t_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=1, name="t_emb",
                                   mask_zero=False, trainable=False, weights=[ent2vec])(ent_t_input)

    rel_r_input = Input(shape=(1,), dtype='int32')
    rel_r_embedding = Embedding(input_dim=relvocabsize+2, output_dim=rel_emd_dim, input_length=1, name="r_emb",
                                   mask_zero=False, trainable=False, weights=[rel2vec])(rel_r_input)
    
    meta_path_input = Input(shape=(3,), dtype='int32')
    meta_path_embedder = Embedding(input_dim=19, output_dim=ent_emd_dim, input_length=3, name="meta_emb")
    

    path_h_input = Input(shape=(input_path_lenth,), dtype='int32')
    path_h_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path_h_input)

    path_t_input = Input(shape=(input_path_lenth,), dtype='int32')
    path_t_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path_t_input)

    path_r_input = Input(shape=(input_path_lenth,), dtype='int32')
    path_r_embedding = Embedding(input_dim=relvocabsize+2, output_dim=rel_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[rel2vec])(path_r_input)

    path2_h_input = Input(shape=(input_path_lenth,), dtype='int32')
    path2_h_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path2_h_input)

    path2_t_input = Input(shape=(input_path_lenth,), dtype='int32')
    path2_t_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path2_t_input)

    path2_r_input = Input(shape=(input_path_lenth,), dtype='int32')
    path2_r_embedding = Embedding(input_dim=relvocabsize+2, output_dim=rel_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[rel2vec])(path2_r_input)

    path3_h_input = Input(shape=(input_path_lenth,), dtype='int32')
    path3_h_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path3_h_input)

    path3_t_input = Input(shape=(input_path_lenth,), dtype='int32')
    path3_t_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[ent2vec])(path3_t_input)

    path3_r_input = Input(shape=(input_path_lenth,), dtype='int32')
    path3_r_embedding = Embedding(input_dim=relvocabsize+2, output_dim=rel_emd_dim, input_length=input_path_lenth,
                                   mask_zero=True, trainable=False, weights=[rel2vec])(path3_r_input)
    
    meta_path_embedding = meta_path_embedder(meta_path_input)
    

    def rmp_add(x):
        a, b, c, d = x
        a = a + d[:,0:1]
        b = b + d[:,2:3]
        c = c + d[:,1:2]
        return [a, b, c]
    
    def rmp_mul(x):
        a, b, c, d = x
        a = a * d[:, 0:1]
        b = b * d[:, 2:3]
        c = c * d[:, 1:2]
        return [a, b, c]
    
    def rmp_concat(x):
        a, b, c, d = x
        a = concatenate([a, d[:, 0:1]], axis=-1)
        b = concatenate([b, d[:, 2:3]], axis=-1)
        c = concatenate([c, d[:, 1:2]], axis=-1)
        return [a, b, c]

    ent_h_embedding, ent_t_embedding, rel_r_embedding = keras.layers.Lambda(rmp_concat)([ent_h_embedding, ent_t_embedding, rel_r_embedding, meta_path_embedding])

    
    ent_h_embedding = Flatten()(ent_h_embedding)
    ent_t_embedding = Flatten()(ent_t_embedding)
    rel_r_embedding = Flatten()(rel_r_embedding)
    
    ent_h_embedding = RepeatVector(input_path_lenth)(ent_h_embedding)
    ent_t_embedding = RepeatVector(input_path_lenth)(ent_t_embedding)
    rel_r_embedding = RepeatVector(input_path_lenth)(rel_r_embedding)

    path_embedding = concatenate([ent_h_embedding, rel_r_embedding, ent_t_embedding,
                                  path_h_embedding,
                                  path_r_embedding,
                                  path_t_embedding], axis=-1)
    path_embedding = Dropout(0.5)(path_embedding)

    path_LSTM = SimpleRNN(100, return_sequences=False)(path_embedding)
    path_LSTM = BatchNormalization()(path_LSTM)
    
    path_LSTM = Dropout(0.5)(path_LSTM)

    path_value = Dense(1, activation='sigmoid')(path_LSTM)

    # -----------------
    path2_embedding = concatenate([ent_h_embedding, rel_r_embedding, ent_t_embedding,
                                  path2_h_embedding,
                                  path2_r_embedding,
                                  path2_t_embedding], axis=-1)
    path2_embedding = Dropout(0.5)(path2_embedding)

    path2_LSTM = SimpleRNN(100, return_sequences=False)(path2_embedding)
    path2_LSTM = BatchNormalization()(path2_LSTM)
    path2_LSTM = Dropout(0.5)(path2_LSTM)
    path2_value = Dense(1, activation='sigmoid')(path2_LSTM)
    # ------------------
    path3_embedding = concatenate([ent_h_embedding, rel_r_embedding, ent_t_embedding,
                                  path3_h_embedding,
                                  path3_r_embedding,
                                  path3_t_embedding], axis=-1)
    path3_embedding = Dropout(0.5)(path3_embedding)

    path3_LSTM = SimpleRNN(100, return_sequences=False)(path3_embedding)
    path3_LSTM = BatchNormalization()(path3_LSTM)
    path3_LSTM = Dropout(0.5)(path3_LSTM)
    path3_value = Dense(1, activation='sigmoid')(path3_LSTM)
    # ------------------


    TransE_input = Input(shape=(1,), dtype='float32')
    
    # 实体评估器特征输入 原有是6个特征
    RRank_input = Input(shape=(6,), dtype='float32')

    RRank_hinden = Dense(100, activation='tanh')(RRank_input)
    RRank_hinden = Dropout(0.5)(RRank_hinden)
    RRank_value = Dense(1, activation='sigmoid')(RRank_hinden)

    BP_input = concatenate([
        path_value, path2_value, path3_value,
        TransE_input,
        RRank_value
    ], axis=-1)


    BP_hidden = Dense(50)(BP_input)
    BP_hidden = Dropout(0.5)(BP_hidden)
    model = Dense(2, activation='softmax')(BP_hidden)
    # model = Dense(1, activation='tanh')(BP_hidden)

    Models = Model([
        ent_h_input, ent_t_input, rel_r_input, meta_path_input, 
        path_h_input, path_t_input, path_r_input,
        path2_h_input, path2_t_input, path2_r_input,
        path3_h_input, path3_t_input, path3_r_input,
        TransE_input,
        RRank_input
    ], model)

    Models.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return Models


# In[ ]:





# In[7]:


def get_dataset(dataset_name):
    base_path = '../ttmf_dataset/'
    datafile = base_path + dataset_name + '.pkl'
    return from_pickle(datafile)


# In[8]:


def get_model_param(dataset):
    ent_vocab, ent_idex_word, rel_vocab, rel_idex_word,     entity2vec, entity2vec_dim,     relation2vec, relation2vec_dim,     train_triple, train_confidence,     test_triple, test_confidence,     tcthreshold_dict, train_transE, test_transE,     rrkthreshold_dict, train_rrank, test_rrank,     max_p,     train_path_h, train_path_t, train_path_r,     test_path_h, test_path_t, test_path_r,     train_path2_h, train_path2_t, train_path2_r,    test_path2_h, test_path2_t, test_path2_r,    train_path3_h, train_path3_t, train_path3_r,    test_path3_h, test_path3_t, test_path3_r = dataset
    return len(ent_vocab), len(rel_vocab), entity2vec, relation2vec, max_p, entity2vec_dim, relation2vec_dim

def get_model_param2(dataset):
    ent_vocab, ent_idex_word, rel_vocab, rel_idex_word,     entity2vec, entity2vec_dim,     relation2vec, relation2vec_dim,     train_triple, train_confidence,     test_triple, test_confidence,     tcthreshold_dict, train_transE, test_transE,     rrkthreshold_dict, train_rrank, test_rrank,     max_p,     train_path_h, train_path_t, train_path_r,     test_path_h, test_path_t, test_path_r,     train_path2_h, train_path2_t, train_path2_r,    test_path2_h, test_path2_t, test_path2_r,    train_path3_h, train_path3_t, train_path3_r,    test_path3_h, test_path3_t, test_path3_r, train_meta_path, test_meta_path = dataset
    return len(ent_vocab), len(rel_vocab), entity2vec, relation2vec, max_p, entity2vec_dim, relation2vec_dim


# In[9]:


error_idx = []
ttmf_pred_lst = []
pos_wrong_lst = []
neg_wrong_lst = []

def test_model(model,
               input_test_h, input_test_t, input_test_r,
               test_path_h, test_path_t, test_path_r,
               test_path2_h, test_path2_t, test_path2_r,
               test_path3_h, test_path3_t, test_path3_r,
               test_transE, test_rrank, test_confidence):

    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.
    results = model.predict([
        np.array(input_test_h), np.array(input_test_t), np.array(input_test_r),
        np.array(test_path_h), np.array(test_path_t), np.array(test_path_r),
        np.array(test_path2_h), np.array(test_path2_t), np.array(test_path2_r),
        np.array(test_path3_h), np.array(test_path3_t), np.array(test_path3_r),
        np.array(test_transE),
        np.array(test_rrank)
        ], batch_size=40)

    for i, res in enumerate(results):
        tag = np.argmax(res)
        ttmf_pred_lst.append(res)
        
        if test_confidence[i][1] == 1:
            if tag == 1:
                total_predict_right += 1.0
            else:
                error_idx.append(i)
                pos_wrong_lst.append(i)
        else:
            if tag == 0:
                total_predict_right += 1.0
            else:
                error_idx.append(i)
                neg_wrong_lst.append(i)
    print('total_predict_right', total_predict_right, 'len(test_confidence)', float(len(test_confidence)))
    acc = total_predict_right / float(len(test_confidence))
    print('acc: ' + str(acc))


# In[10]:


error_idx = []

def test_model2(model,
               input_test_h, input_test_t, input_test_r, input_test_meta_path,
               test_path_h, test_path_t, test_path_r,
               test_path2_h, test_path2_t, test_path2_r,
               test_path3_h, test_path3_t, test_path3_r,
               test_transE, test_rrank, test_confidence):

    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.
    results = model.predict([
        np.array(input_test_h), np.array(input_test_t), np.array(input_test_r), np.array(input_test_meta_path), 
        np.array(test_path_h), np.array(test_path_t), np.array(test_path_r),
        np.array(test_path2_h), np.array(test_path2_t), np.array(test_path2_r),
        np.array(test_path3_h), np.array(test_path3_t), np.array(test_path3_r),
        np.array(test_transE),
        np.array(test_rrank)
        ], batch_size=40)

    for i, res in enumerate(results):
        tag = np.argmax(res)
        
        if test_confidence[i][1] == 1:
#             fin1.write(str(res[1]) + '\n')
            if tag == 1:
                total_predict_right += 1.0
            else:
                error_idx.append(i)
        else:
#             fin0.write(str(res[1]) + '\n')
            if tag == 0:
                total_predict_right += 1.0
            else:
                error_idx.append(i)
    print('total_predict_right', total_predict_right, 'len(test_confidence)', float(len(test_confidence)))
    acc = total_predict_right / float(len(test_confidence))
    print('acc: ' + str(acc))


# In[11]:


weights = '../ttmf_dataset/model/'


# In[12]:


dataset_o = get_dataset('transe_diseasekg_new_ttmf_train_valid_noise_30')
ev_o, rv_o, e2v_o, r2v_o, mp_o, e2vd_o, r2vd_o = get_model_param(dataset_o)
model_o = get_model_origin(ev_o, rv_o, e2v_o, r2v_o, mp_o, e2vd_o, r2vd_o)
model_o.load_weights(weights + 'transe_diseasekg_new_ttmf_train_valid_noise_30.h5')


# ## 加载验证数据集

# In[13]:


def load_valid_dataset(dataset_type):
    load_path = '../ttmf_dataset/{}'.format(dataset_type)
    return from_pickle(load_path)


# In[14]:


valid_origin = load_valid_dataset('transe_diseasekg_new_ttmf_test_noise_30.pkl')


# In[15]:


def get_valid_data2(valid_dataset):
    v_triple, v_confidence, v_transE, v_rrank, v_path_h, v_path_t, v_path_r, v_path2_h, v_path2_t, v_path2_r, v_path3_h, v_path3_t, v_path3_r, valid_meta_path = valid_dataset
    input_valid_h = np.zeros((len(v_triple),1)).astype('int32')
    input_valid_t = np.zeros((len(v_triple),1)).astype('int32')
    input_valid_r = np.zeros((len(v_triple),1)).astype('int32')
    for idx, tri in enumerate(v_triple):
        input_valid_h[idx,] = tri[0]
        input_valid_t[idx,] = tri[1]
        input_valid_r[idx,] = tri[2]
    return input_valid_h, input_valid_t, input_valid_r, v_path_h, v_path_t, v_path_r, v_path2_h, v_path2_t, v_path2_r, v_path3_h, v_path3_t, v_path3_r, v_transE, v_rrank, v_confidence, valid_meta_path

def get_valid_data(valid_dataset):
    v_triple, v_confidence, v_transE, v_rrank, v_path_h, v_path_t, v_path_r, v_path2_h, v_path2_t, v_path2_r, v_path3_h, v_path3_t, v_path3_r = valid_dataset
    input_valid_h = np.zeros((len(v_triple),1)).astype('int32')
    input_valid_t = np.zeros((len(v_triple),1)).astype('int32')
    input_valid_r = np.zeros((len(v_triple),1)).astype('int32')
    for idx, tri in enumerate(v_triple):
        input_valid_h[idx,] = tri[0]
        input_valid_t[idx,] = tri[1]
        input_valid_r[idx,] = tri[2]
    return input_valid_h, input_valid_t, input_valid_r, v_path_h, v_path_t, v_path_r, v_path2_h, v_path2_t, v_path2_r, v_path3_h, v_path3_t, v_path3_r, v_transE, v_rrank, v_confidence


# In[16]:


input_valid_h, input_valid_t, input_valid_r, v_path_h, v_path_t, v_path_r, v_path2_h, v_path2_t, v_path2_r, v_path3_h, v_path3_t, v_path3_r, v_transE, v_rrank, v_confidence = get_valid_data(valid_origin)
test_model(model_o, input_valid_h, input_valid_t, input_valid_r, v_path_h, v_path_t, v_path_r, v_path2_h, v_path2_t, v_path2_r, v_path3_h, v_path3_t, v_path3_r, v_transE, v_rrank, v_confidence)


# In[17]:


len(ttmf_pred_lst)


# In[18]:


len(error_idx)


# In[19]:


id2ent, id2rel = dataset_o[1], dataset_o[3]


# In[20]:


len(valid_origin[0])


# In[21]:


# In[22]:

# In[23]:

# In[24]:


with open('../融合模型中各模型预测结果/ttmf_diseasekg_new_pred_noise_30.pkl', 'wb') as f:
    pickle.dump([right_triples, err_triples, ttmf_pred_lst], f)


# In[ ]:





# In[ ]:





# In[ ]:




