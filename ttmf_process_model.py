#!/usr/bin/env python
# coding: utf-8
# ==use jupyter==




import pickle
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import time
import math
from numpy import *
from collections import defaultdict
from pygraph.classes.digraph import digraph
import networkx as nx




class ModelUtils:
    def __init__(self, data_path):
        self.save_path = data_path
        
    def to_pickle(self, data, display=False):
        with open(self.save_path, 'wb') as f:
            pickle.dump(data, f)
            if display:
                print('数据已导出至: {}'.format(self.save_path))
                
    def from_pickle(self, file_name, display=False):
        with open(self.save_path+file_name, 'rb') as f:
            if display:
                print('数据已读取至: {}'.format(self.save_path))
            return pickle.load(f)
    
    @staticmethod
    def replace_func(h_ent2id, t_ent2id, r_ent2id, raw, strategy): # 随机替换h、r、t
        flag = True 
        choice = -1
        if strategy == 'replace_h':
            while(flag):
                choice = np.random.choice(h_ent2id, 1)[0]
                if choice != raw:
                    flag = False
        if strategy == 'replace_r':
            while(flag):
                choice = np.random.choice(r_ent2id, 1)[0]
                if choice != raw:
                    flag = False
        if strategy == 'replace_t':
            while(flag):
                choice = np.random.choice(t_ent2id, 1)[0]
                if choice != raw:
                    flag = False
        return choice
    
    @staticmethod
    def get_all_triples(triples):
        triples_dict = {}
        for triple in tqdm(triples):
            if triple[0] in triples_dict.keys():
                if triple[1] in triples_dict.get(triple[0]).keys():
                    triples_dict.get(triple[0]).get(triple[1]).append(triple[2])
                else:
                    triples_dict.get(triple[0])[triple[1]] = [triple[2]]
            else:
                triples_dict[triple[0]] = {triple[1]: [triple[2]]}
        print('triples dict size: {}'.format(len(triples_dict)))
        return triples_dict
    
    @staticmethod
    def load_vec_txt(fname, vocab, k=100):
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



class EntityEstimatorUtils:
    def get_all_triples(self):
        triples_dict = {}
        for triple in self.triples:
            if triple[0] in triples_dict.keys():
                if triple[1] in triples_dict.get(triple[0]).keys():
                    triples_dict.get(triple[0]).get(triple[1]).append(triple[2])
                else:
                    triples_dict.get(triple[0])[triple[1]] = [triple[2]]
            else:
                triples_dict[triple[0]] = {triple[1]: [triple[2]]}
        print('triples dict size: {}'.format(len(triples_dict)))
        return triples_dict
    
    @staticmethod
    def DFS(triples_dict, dg, node, depth=3):
        node = int(node)
        depth -= 1

        if depth < 0:
            return dg
        if node not in triples_dict.keys():
            return dg
        sequence = triples_dict[node]
        count = 0
        for key in sequence.keys():
            if not dg.has_node(key):
                dg.add_node(key)
            if not dg.has_edge((node, key)):
                dg.add_edge((node, key), wt=len(sequence[key]))  # len(sequence[key])->节点间关系数量
                count += len(sequence[key])
            else:
                continue
            dg = EntityEstimatorUtils.DFS(triples_dict, dg, key, depth)

        for n in dg.neighbors(node):
            dg.set_edge_weight((node, n),wt= float(dg.edge_weight((node, n))/max(count,1)))
        return dg
    
    def get_tri_with_conf(self, train):
        train_triple = []
        train_confidence = []
        for t in tqdm(train):
            train_triple.append(t)
            if t[3] == 1:
                train_confidence.append([0, 1])
            else:
                train_confidence.append([1, 0])
        return train_triple, train_confidence
    
    def rrcThreshold(self, train_triples, entityRank):
        threshold_dict = {}
        rrank_dict = {}
        for tt in tqdm(train_triples):  # [(104,105,45,1), (), ()]

            if tt[1] in entityRank[tt[0]].keys():
                v = (entityRank[tt[0]][tt[1]], tt[3])
            else:
                v = (0.0, tt[3])

            if tt[0] not in rrank_dict.keys():
                rrank_dict[tt[0]] = [v]
            else:
                rrank_dict[tt[0]].append(v)

        for it in tqdm(rrank_dict.keys()):
            threshold_dict[it] = self.getThreshold(rrank_dict[it])

        return threshold_dict
    
    def getThreshold(self, rrank):

        distanceFlagList = rrank
        distanceFlagList = sorted(distanceFlagList, key=lambda sp: sp[0], reverse=True)

        threshold = distanceFlagList[0][0] + 0.01
        maxValue = 0
        currentValue = 0
        for i in range(1, len(distanceFlagList)):
            if distanceFlagList[i - 1][1] == 1:
                currentValue += 1
            else:
                currentValue -= 1

            if currentValue > maxValue:
                threshold = (distanceFlagList[i][0] + distanceFlagList[i - 1][0]) / 2.0
                maxValue = currentValue
        return threshold
    
    def get_features(self, entity_rank, sub_graphs, threshold_dict, ent_num): 
        features = []
        classlabels = []
        results = defaultdict(list)
        for id in tqdm(range(ent_num)):
            core_node = id

            dg_nodes, dg_edges = sub_graphs[core_node]
            dg = digraph()
            for node in dg_nodes:
                dg.add_node(node)
            for edge in dg_edges:
                dg.add_edge((edge[0], edge[1]))

            rudu = {}
            chudu = {}
            for d in dg.nodes():
                rudu[d] = len(dg.incidents(d))
                chudu[d] = len(dg.neighbors(d))

            depdict = {core_node: 0}  # 节点所在深度
            list1 = dg.neighbors(core_node)
            list2 = [0] * len(list1)
            # i = 0
            while list1.__len__()>0:
                node = list1[0]
                # print(node)
                if node not in depdict.keys():

                    depdict[node] = list2[0]+1

                    for node2 in dg.neighbors(node):
                        if node2 not in depdict.keys():
                            list1.append(node2)
                            list2.append(depdict[node])
                del list1[0]
                del list2[0]
            for d in dg.nodes():
                rr = self.get_f(int(core_node), int(d), threshold_dict, entity_rank)
                depth = depdict[d]
                results[int(core_node)].append((d, rr, rudu[core_node], chudu[core_node], rudu[d], chudu[d], depth))
        return results
    
    
    def get_f(self, head, tail, threshold_dict,dict_entityRank):
        if head in threshold_dict.keys():
            threshold = threshold_dict[head]
        else:
            threshold = 0.5
        f = 0.001
        if tail in dict_entityRank[head].keys():
            rankvalue = dict_entityRank[head][tail]
            try:
                f = 1.0 / (1.0 + math.exp(-25 * (rankvalue - threshold)))
            except:
                f = 0.001
        return f
    
    # =====实体评估器中加入特征=====
    # =====特征向量中心性=====
    # =====紧密型中心性=====
    # =====介数中心性=====
    # =====聚类系数=====
    # ===========================
    def get_features_ev(self, entity_rank, sub_graphs, threshold_dict, ent_num): 
        features = []
        classlabels = []
        results = defaultdict(list)

        #for id in tqdm(range(1, ent_num)): #有改动
        for id in tqdm(range(ent_num)):
            core_node = id

             # dg_nodes: [5, 11, 12, 6, 9, 7, 10, 8]
             # dg_edges: [(5, 11, 0.08333333333333333),(5, 5, 0.08333333333333333),(5, 12, 0.08333333333333333)]
            dg_nodes, dg_edges = sub_graphs[core_node]
            dg = digraph()
            G = nx.DiGraph() # 有向图--->DiGraph() 无向图--->Graph()
            for node in dg_nodes:
                dg.add_node(node)
                G.add_node(node)

            for edge in dg_edges:
                dg.add_edge((edge[0], edge[1]))
                G.add_edge(edge[0], edge[1])
        
            rudu = {}
            chudu = {}

            try:
                cent = nx.eigenvector_centrality_numpy(G) # 特征向量中心性
            except:
                '''
                若出现上述无法计算情况，采用下述方法手动计算
                '''
                matrix_shape = len(dg_nodes)
                mapping_i2v, mapping_v2i = dict(), dict()
                relations = defaultdict(list)
                cent = dict()

                for idx, val in enumerate(dg_nodes):
                    mapping_i2v[idx] = val
                    mapping_v2i[val] = idx

                for edge in dg_edges:
                    start_node = edge[0]
                    end_node = edge[1]
                    relations[mapping_v2i[start_node]].append(mapping_v2i[end_node])
                ev_lst = get_ev_value(matrix_shape, relations) # 特征向量中心性列表
                for cidx, cval in enumerate(ev_lst):
                    cent[mapping_i2v[cidx]] = cval.A[0][0]
            
            closeness_cent = nx.closeness_centrality(G)
            betweenness_cent = nx.betweenness_centrality(G)
            cluster = nx.clustering(G)
            
            for d in dg.nodes():
                rudu[d] = len(dg.incidents(d))
                chudu[d] = len(dg.neighbors(d))

            depdict = {core_node: 0}  # 节点所在深度
            list1 = dg.neighbors(core_node)
            list2 = [0] * len(list1)
            # i = 0
            while list1.__len__()>0:
                node = list1[0]
                if node not in depdict.keys():

                    depdict[node] = list2[0]+1

                    for node2 in dg.neighbors(node):
                        if node2 not in depdict.keys():
                            list1.append(node2)
                            list2.append(depdict[node])
                del list1[0]
                del list2[0]


            for d in dg.nodes():
                rr = eu.get_f(int(core_node), int(d), threshold_dict, entity_rank)
                depth = depdict[d]
                # =====实体级特征在这里改=====
                results[int(core_node)].append((d, 
                                                rr, 
                                                rudu[core_node], 
                                                chudu[core_node], 
                                                rudu[d], 
                                                chudu[d], 
                                                depth, 
                                                cent[core_node], 
                                                cent[d], 
                                                closeness_cent[core_node], 
                                                closeness_cent[d], 
                                                #betweenness_cent[core_node], 
                                                #betweenness_cent[d], 
                                                cluster[core_node], 
                                                cluster[d]))
                    
        return results
    
    
class PRIterator:
    __doc__ = '''计算一张图中的PR值'''

    def __init__(self, dg, core_node):
        self.damping_factor = 0.85  # 阻尼系数,即α
        self.max_iterations = 500  # 最大迭代次数
        self.min_delta = 0.00001  # 确定迭代是否结束的参数,即ϵ
        self.core_node = core_node
        self.graph = dg

    def page_rank(self):
        #print('******')
        cout =0
         # 先将图中没有出链的节点改为对所有节点都有出链
        for node in self.graph.nodes():
            if len(self.graph.neighbors(node)) == 0:
                cout +=1
                try:
                    digraph.add_edge(self.graph, (node, node), wt=0.5)
                    digraph.add_edge(self.graph, (node, self.core_node), wt=0.5)
                except:
                    pass

        #print(cout)

        nodes = self.graph.nodes()
        graph_size = len(nodes)

        if graph_size == 0:
            return {}

        # page_rank = dict.fromkeys(nodes, 1.0 / graph_size)  # 给每个节点赋予初始的PR值
        page_rank = dict.fromkeys(nodes, 0.0)  # 给每个节点赋予初始的PR值
        page_rank[self.core_node] = 1.0
        # print(page_rank)
        damping_value = (1.0 - self.damping_factor) / graph_size  # 公式中的(1−α)/N部分
        #print('start iterating...')
        flag = False
        for i in range(self.max_iterations):
            change = 0
            for node in nodes:
                rank = 0
                for incident_page in self.graph.incidents(node):  # 遍历所有“入射”的页面
                    rank += self.damping_factor * page_rank[incident_page] * float(self.graph.edge_weight((incident_page,node)))
                rank += damping_value
                change += abs(page_rank[node] - rank)  # 绝对值
                page_rank[node] = rank


            if change < self.min_delta:  # 跳出循环
                flag = True
                break
        return page_rank


# In[6]:


class EntityEstimator:
    def __init__(self, ent2id, depth, EntityEstimatorUtils):
        self.ent2id_dict = ent2id
        self.graph_depth = depth
        self.eu = EntityEstimatorUtils
        
    def construct_directed_graphs(self, triples_dict):
        print('------【通过词表为每个头实体构建联通子图】------')
        sub_graphs = {}
        for _, ent_id in tqdm(self.ent2id_dict.items()):
            node0 = ent_id
            dg = digraph()
            dg.add_node(node0)
            dg = EntityEstimatorUtils.DFS(triples_dict, dg, node0, depth=self.graph_depth)

            dg_nodes = [node for node in dg.nodes()]
            dg_edges = [(edge[0], edge[1], dg.edge_weight(edge))for edge in dg.edges()]
            sub_graphs[node0] = [dg_nodes, dg_edges]
        return sub_graphs
    
    def iter_resources(self, sub_graphs):
        print('------【获取每个子图中的资源流】------')
        entity_ranks = {}
        for core_node in tqdm(sub_graphs.keys()):
            dg_nodes, dg_edges = sub_graphs[core_node]
            dg = digraph()
            for node in dg_nodes:
                dg.add_node(node)
            for edge in dg_edges:
                dg.add_edge((edge[0], edge[1]), wt=edge[2])

            pr = PRIterator(dg, int(core_node))
            page_ranks = pr.page_rank()
            entity_ranks[core_node] = page_ranks
        return entity_ranks
    
    def get_rr_features(self, conf_train2id, sub_graphs, entity_ranks):
        print('-----【实体评估器生成特征】-----')
        train_triple, train_confidence = self.eu.get_tri_with_conf(conf_train2id)
        threshold_dict = self.eu.rrcThreshold(train_triple, entity_ranks)
        ent_num = len(entity_ranks)
        resource_rank = self.eu.get_features(entity_ranks, sub_graphs, threshold_dict, ent_num)
        return resource_rank
    
    def get_rrank_features(self, entity_dict_features, Examples):
        features = []
        dict_features = {}
        for k in entity_dict_features.keys():
            t = {}
            for j in entity_dict_features[k]:
                t[j[0]] = j[1:]
            dict_features[k] = t

        for id, example in enumerate(Examples):
            if example[1] in dict_features[example[0]].keys():
                features.append(dict_features[example[0]][example[1]])
            else:
                features.append([0.0, 0.0, 0.0, 0.0, 0.0, 10000.0])
                
        return features
    
    # =====1、实体评估器中加入节点特征向量中心性特征=====
    def get_rr_features_ev(self, conf_train2id, sub_graphs, entity_ranks):
        print('-----【实体评估器生成特征_添加特征】-----')
        train_triple, train_confidence = self.eu.get_tri_with_conf(conf_train2id)
        threshold_dict = self.eu.rrcThreshold(train_triple, entity_ranks)
        ent_num = len(entity_ranks)
        resource_rank = self.eu.get_features_ev(entity_ranks, sub_graphs, threshold_dict, ent_num)
        return resource_rank
    
        
    def get_rrank_features_ev(self, entity_dict_features, Examples):
        features = []
        dict_features = {}
        for k in entity_dict_features.keys():
            t = {}
            for j in entity_dict_features[k]:
                t[j[0]] = j[1:]
            dict_features[k] = t

        for id, example in enumerate(Examples):
            if example[1] in dict_features[example[0]].keys():
                features.append(dict_features[example[0]][example[1]])
            else:
                features.append([0.0, 0.0, 0.0, 0.0, 0.0, 10000.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
                
        return features
    # ============================================




class RelationEstimatorUtils:
    @staticmethod
    def tcThreshold(triples, ent2vec, rel2vec):
        threshold_dict = {}
        trans_dict = {}
        for tri in triples:
            s = ent2vec[tri[0]] + rel2vec[tri[2]] - ent2vec[tri[1]]
            transV = np.linalg.norm(s, ord=2)
            if tri[2] not in trans_dict.keys():
                trans_dict[tri[2]] = [(transV, tri[3])]
            else:
                trans_dict[tri[2]].append((transV, tri[3]))

        for it in trans_dict.keys():
            # 算出和该关系有关的三元组的可能阈值？
            threshold_dict[it] = RelationEstimatorUtils.getThreshold(trans_dict[it])
        return threshold_dict
    
    @staticmethod
    def getThreshold(rrank):
        distanceFlagList = rrank
        distanceFlagList = sorted(distanceFlagList, key=lambda sp: sp[0], reverse=False)

        threshold = distanceFlagList[0][0] - 0.01
        maxValue = 0
        currentValue = 0
        for i in range(1, len(distanceFlagList)):
            if distanceFlagList[i - 1][1] == 1:
                currentValue += 1
            else:
                currentValue -= 1

            if currentValue > maxValue:
                threshold = (distanceFlagList[i][0] + distanceFlagList[i - 1][0]) / 2.0
                maxValue = currentValue
        return threshold


class RelationEstimator:
    def __init__(self, ent_vocab, rel_vocab, ent2vec, rel2vec):
        self.ent_vocab = ent_vocab
        self.rel_vocab = rel_vocab
        self.ent2vec = ent2vec
        self.rel2vec = rel2vec
    
    def get_threshold_dict(self, triples):
        return RelationEstimatorUtils.tcThreshold(triples, self.ent2vec, self.rel2vec)
    
    def get_TransConfidence(self, threshold_dict, triples):
        All_conf = 0.0
        confidence_dict = []

        right = 0.0
        for triple in triples:
            if triple[2] in threshold_dict.keys():
                threshold = threshold_dict[triple[2]]
            else:
                # print('threshold is None !!!!!!!!!')
                threshold = 0.0

            s = self.ent2vec[triple[0]] + self.rel2vec[triple[2]] - self.ent2vec[triple[1]]
            transV = np.linalg.norm(s, ord=2)
            f = 1.0 / (1.0 + math.exp(-1 * (threshold - transV)))  # P(E(h,r,t))
            f = (threshold - transV)

            confidence_dict.append(f)

            if transV <= threshold and triple[3] == 1:
                right += 1.0
                All_conf += f

            elif transV > threshold and triple[3] == -1:
                right += 1.0


        print('TransConfidence accuracy ---- ', right / len(triples))

        avg_conf = All_conf / float(len(triples))
        print('avg_confidence ... ', avg_conf, float(len(triples)))

        return confidence_dict  # 存放每一个三元组的实体级置信度


class GraphEstimatorUtils:
    def Rank(self, Paths, Ent2V, Rel2V, h, t, r):
        plist =[]

        for path in Paths:
            SD_r = 0.0
            SD_h = 0.0
            SD_t = 0.0
            for triple in path:
                cosV_h = dot(Ent2V[int(h)], Ent2V[int(triple[1])]) / (linalg.norm(Ent2V[int(h)]) * linalg.norm(Ent2V[int(triple[1])]))
                SD_h +=cosV_h
                cosV_t = dot(Ent2V[int(t)], Ent2V[int(triple[0])]) / (linalg.norm(Ent2V[int(t)]) * linalg.norm(Ent2V[int(triple[0])]))
                SD_t +=cosV_t

                cosV_r = dot(Rel2V[int(r)], Rel2V[int(triple[2])]) / (linalg.norm(Rel2V[int(r)]) * linalg.norm(Rel2V[int(triple[2])]))
                SD_r +=cosV_r
            SD = (SD_r + SD_h + SD_t) / (3 * len(path))
            plist.append((SD, path))

        plist = sorted(plist, key=lambda sp: sp[0], reverse=True)


        return plist

    def searchpath(self, core, startnode, dict, taillist, Paths, pathlist, depth=5):
        depth -= 1

        if depth <= 0:
            return Paths

        if startnode not in dict.keys():
            return Paths

        sequence = dict[startnode]  # {ent1:[rel1, rel2], ent2:[rel3, rel4, rel5], ...}
        count = 0
        for key in sequence.keys():  # 遍历startnode(头实体)对应的尾实体

            if key in taillist:
                continue

            for val in sequence.get(key):  # 获取h->t中的关系数组[rel1, rel2, ...]
                pathlist.append((startnode, key, val))  # [(startnode, tail, rel1), (startnode, tail, rel2), ...]
                taillist.append(key)
                # print('***', pathlist)
                s = tuple(pathlist)
                if '{}_{}'.format(core, key) not in Paths.keys():
                    Paths['{}_{}'.format(core, key)] = [s]
                else:
                    Paths['{}_{}'.format(core, key)].append(s)
                pathlist.remove((startnode, key, val))
                taillist.remove(key)


            for val in sequence.get(key):
                taillist.append(key)
                pathlist.append((startnode, key, val))
                Paths = self.searchpath(core, key, dict, taillist, Paths, pathlist, depth)
                taillist.remove(key)
                pathlist.remove((startnode, key, val))

        return Paths

class GraphEstimator:
    def __init__(self, triples_dict, ent_vocab, rel_vocab, ent2vec, rel2vec, GraphEstimatorUtils):
        self.triples_dict = triples_dict
        self.ent_vocab = ent_vocab
        self.rel_vocab = rel_vocab
        self.ent2vec = ent2vec
        self.rel2vec = rel2vec
        self.line_dict = {}
        self.head_list = []
        self.geu = GraphEstimatorUtils
        
    def generate_head_and_line(self, conf_train2id, conf_test2id):
        print('generate 【head_list】 and 【line_dict】====>conf_train2id, conf_test2id')
        for triples in tqdm([conf_train2id, conf_test2id]):
            for triple in triples:
                if '{}_{}'.format(triple[0], triple[1]) in self.line_dict.keys():
                    if(triple[0], triple[1], triple[2]) not in self.line_dict['{}_{}'.format(triple[0], triple[1])]:
                        self.line_dict['{}_{}'.format(triple[0], triple[1])].append((triple[0], triple[1], triple[2]))
                else:
                    self.line_dict['{}_{}'.format(triple[0], triple[1])] = [(triple[0], triple[1], triple[2])]
                if triple[0] not in self.head_list:
                    self.head_list.append(triple[0])
    
    def get_paths(self):
        # 遍历头实体
        result_dict = defaultdict(list)
        for i in tqdm(self.head_list):
            startnode = i
            Paths = {}
            pathlist = []
            taillist = [startnode]
            Paths = self.geu.searchpath(startnode, startnode, self.triples_dict, taillist, Paths, pathlist, 4)
            for head in Paths.keys():
                if head in self.line_dict.keys():
                    for tri in self.line_dict[head]:
                        if '{}_{}_{}'.format(tri[0], tri[1], tri[2]) in result_dict:
                            continue              
                        Pranklist = self.geu.Rank(Paths[head], self.ent2vec, self.rel2vec, tri[0], tri[1], tri[2])
                        for num, ps in enumerate(Pranklist):
                            if num > 50:
                                break
                            if ps[1] == ((tri[0], tri[1], tri[2]),):
                                continue
                            result_dict['{}_{}_{}'.format(tri[0], tri[1], tri[2])].append(ps)
        return result_dict


def get_path_index(path_result, max_p, train_triple, topk):
    train_path_h = []
    train_path_t = []
    train_path_r = []
    not_find = 0
    for baset in tqdm(train_triple):
        ph = []
        pt = []
        pr = []
        length = 0
        fstr = '{}_{}_{}'.format(baset[0], baset[1], baset[2])
        if fstr in path_result:
            lines = path_result[fstr]
            if len(lines) >= (topk+1):
                tri = lines[topk]
                for path in tri[1:]:
                    for tri in path:
                        ph.append(tri[0])
                        ph.append(tri[1])
                        ph.append(tri[2])
                length = len(tri)-1
        else:
            not_find += 1
            #print(fstr, 'Not find the path file!!!!!')
        for i in range(0, max_p - length):
            ph.append(0)
            pt.append(0)
            pr.append(0)
        train_path_h.append(ph)
        train_path_t.append(pt)
        train_path_r.append(pr)
    return train_path_h, train_path_t, train_path_r


def get_triples_and_confidence(conf_train2id, conf_test2id):
    train, test = [], []
    for i in conf_train2id:
        train.append(i)
    for i in conf_test2id:
        test.append(i)
    def get_confidence(triples):
        confidence = []
        for t in triples:
                if t[-1] == 1:
                    confidence.append([0, 1])
                else:
                    confidence.append([1, 0])
        return confidence
    train_confidence = get_confidence(train)
    test_confidence = get_confidence(test)
    return train, train_confidence, test, test_confidence

def get_ev_value(matrix_shape, relations):
    adjacency_matrix = np.mat(np.zeros([matrix_shape, matrix_shape]))
    for source, targets in relations.items():
        for target in targets:
            adjacency_matrix[source, target] = 1
            adjacency_matrix[source, source] = 1
    lam, vet = np.linalg.eig(adjacency_matrix)
    ev = np.abs(vet[:, np.argmax(lam)])
    return ev


# ## 使用DiseaseKG:基于cnSchma常见疾病信息知识图谱
# * 来源:OpenKG
# * [链接](http://openkg.cn/dataset/disease-information)

# In[14]:


import json
import random


# In[15]:


def get_data_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


datafile_path = './'

labels = set()
ent2label, rel2label = dict(), dict()


# In[14]:


def from_pkl(fpath):
    with open(fpath, 'rb') as f:
        return pickle.load(f)
    
def to_pkl(fpath, data_lst):
    with open(fpath, 'wb') as fin:
        pickle.dump(data_lst, fin)


import pickle

conf_train2id, conf_test2id, conf_valid2id, gold_train2id, gold_test2id, gold_valid2id, ent_vocab, rel_vocab, all_triples, pos_triples, neg_triples = from_pkl('./diseasekg_dataset/diseasekg_data_noise_10.pkl')

# ## 接入openKE

# In[37]:


def to_txt(file, data, display=True):
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

def to_txt_dct(file, data, data_nums, display=True):
    with open(file, 'w', encoding='utf-8') as f:
        f.write(str(data_nums))
        f.write('\n')
        for k, v in data.items():
            f.write('{}\t{}'.format(k, v))
            f.write('\n')
    if display:
        print('已导出至: {}'.format(file))


# In[34]:


file_path = '../OpenKE/diseasekg_new/'

train2id = [len(gold_train2id)] + gold_train2id
test2id = [len(gold_test2id)] + gold_test2id
valid2id = [len(gold_valid2id)] + gold_valid2id
ent2id = ent_vocab['ent2id']
rel2id = rel_vocab['rel2id']

to_txt(file_path+'train2id.txt', train2id)
to_txt(file_path+'test2id.txt', test2id)
to_txt(file_path+'valid2id.txt', valid2id)

to_txt_dct(file_path+'entity2id.txt', ent2id, len(ent2id))
to_txt_dct(file_path+'relation2id.txt', rel2id, len(rel2id))


# In[38]:


openKE_embedding_path = '../embedding_result/'

import json
import numpy as np

def loadJson(file):
    with open(file, 'r') as f:
        return json.load(f)

def loadVector(file):
    ent_param = loadJson(file)['ent_embeddings.weight']
    rel_param = loadJson(file)['rel_embeddings.weight']
    return ent_param, rel_param

def toVectorFile(path, mapping, data):
    with open(path, 'w', encoding='utf-8') as f:
        for i in tqdm(range(len(data))):
            f.write('{}\t{}\n'.format(mapping[i], data[i]))


kg_name = 'diseasekg_new'


# In[29]:


kg_embedding_path = '../kg_embedding/{}/'.format(kg_name)


# In[41]:


transe_ent, transe_rel = loadVector(openKE_embedding_path+'transe_{}.json'.format(kg_name))

toVectorFile(kg_embedding_path+'transe_ent_{}.txt'.format(kg_name), ent_vocab['id2ent'], transe_ent)
toVectorFile(kg_embedding_path+'transe_rel_{}.txt'.format(kg_name), rel_vocab['id2rel'], transe_rel)


# In[43]:


map_labels = list(labels)


# In[44]:


map4labels = {v: k for k, v in enumerate(map_labels)}


# In[46]:


#embedding加入属性信息
transe_ent_attr = []
one_hot = [0.0] * len(labels)
for idx, ent in enumerate(transe_ent):
    index = map4labels[ent2label[ent_vocab['id2ent'][idx]]]
    one_hot[index] = 1.0
    transe_ent_attr.append(ent + one_hot)
    one_hot = [0.0] * len(labels)


# In[47]:


transe_rel_attr = []
one_hot = [0.0] * len(labels)
for idx, rel in enumerate(transe_rel):
    index = map4labels[rel2label[rel_vocab['id2rel'][idx]]]
    one_hot[index] = 1.0
    transe_rel_attr.append(rel + one_hot)
    one_hot = [0.0] * len(labels)


# In[49]:


toVectorFile(kg_embedding_path+'transe_ent_{}_onehot.txt'.format(kg_name), ent_vocab['id2ent'], transe_ent_attr)
toVectorFile(kg_embedding_path+'transe_rel_{}_onehot.txt'.format(kg_name), rel_vocab['id2rel'], transe_rel_attr)


# In[ ]:





# In[ ]:





# In[458]:


transe_ent, transe_rel = loadVector(openKE_embedding_path+'transe.json')
transr_ent, transr_rel = loadVector(openKE_embedding_path+'transr.json')
transh_ent, transh_rel = loadVector(openKE_embedding_path+'transh.json')
transd_ent, transd_rel = loadVector(openKE_embedding_path+'transd.json')


# In[459]:


toVectorFile(datafile_path+'transe-entity-vector.txt', ent_vocab['id2ent'], transe_ent)
toVectorFile(datafile_path+'transe-relation-vector.txt', rel_vocab['id2rel'], transe_rel)
toVectorFile(datafile_path+'transh-entity-vector.txt', ent_vocab['id2ent'], transh_ent)
toVectorFile(datafile_path+'transh-relation-vector.txt', rel_vocab['id2rel'], transh_rel)
toVectorFile(datafile_path+'transr-entity-vector.txt', ent_vocab['id2ent'], transr_ent)
toVectorFile(datafile_path+'transr-relation-vector.txt', rel_vocab['id2rel'], transr_rel)
toVectorFile(datafile_path+'transd-entity-vector.txt', ent_vocab['id2ent'], transd_ent)
toVectorFile(datafile_path+'transd-relation-vector.txt', rel_vocab['id2rel'], transd_rel)


# In[65]:


simple_ent, simple_rel = loadVector(openKE_embedding_path+'simple.json')


# In[66]:

# In[ ]:





# In[ ]:





# In[58]:


# ## pipeline

# In[30]:


eu = EntityEstimatorUtils()
ee = EntityEstimator(ent_vocab['ent2id'], 4, eu)


# In[31]:


sub_graphs = ee.construct_directed_graphs(all_triples)
entity_ranks = ee.iter_resources(sub_graphs)



# In[32]:


resource_ranks = ee.get_rr_features(conf_train2id, sub_graphs, entity_ranks)


# In[44]:


# In[33]:


_, ent2vec = ModelUtils.load_vec_txt(kg_embedding_path+'transe_ent_{}.txt'.format(kg_name), ent_vocab['ent2id'])
_, rel2vec = ModelUtils.load_vec_txt(kg_embedding_path+'transe_rel_{}.txt'.format(kg_name), rel_vocab['rel2id'])

re = RelationEstimator(ent_vocab, rel_vocab, ent2vec, rel2vec)

train_triples = []
for i in conf_train2id:
    train_triples.append(i)
#     train_triples.append(i[0])
#     train_triples.append(i[1])
_ = re.get_TransConfidence(re.get_threshold_dict(train_triples), train_triples)

gu = GraphEstimatorUtils()
ge = GraphEstimator(all_triples, ent_vocab, rel_vocab, ent2vec, rel2vec, gu)
ge.generate_head_and_line(conf_train2id, conf_test2id)
ge_result = ge.get_paths()

train, train_confidence, valid, valid_confidence = get_triples_and_confidence(conf_train2id, conf_valid2id)

#生成数据集
ent2id, id2ent = ent_vocab['ent2id'], ent_vocab['id2ent']
rel2id, id2rel = rel_vocab['rel2id'], rel_vocab['id2rel']


entity2vec, entvec_k = ent2vec, 100
relation2vec, relvec_k = rel2vec, 100

train_triple, train_confidence = train, train_confidence
valid_triple, valid_confidence = valid, valid_confidence
tcthreshold_dict = re.get_threshold_dict(train_triple)
train_transE = re.get_TransConfidence(tcthreshold_dict, train_triple)
valid_transE = re.get_TransConfidence(tcthreshold_dict, valid_triple)
rrkthreshold_dict = {}

train_rrank = ee.get_rrank_features(ee.get_rr_features(conf_train2id, sub_graphs, entity_ranks), train_triple)
valid_rrank = ee.get_rrank_features(ee.get_rr_features(conf_valid2id, sub_graphs, entity_ranks), valid_triple)

#train_rrank = ee.get_rrank_features_ev(ee.get_rr_features_ev(conf_train2id, sub_graphs, entity_ranks), train_triple)
#test_rrank = ee.get_rrank_features_ev(ee.get_rr_features_ev(conf_test2id, sub_graphs, entity_ranks), test_triple)

max_p = 3
train_path_h, train_path_t, train_path_r = get_path_index(ge_result, max_p, train, 0)
valid_path_h, valid_path_t, valid_path_r = get_path_index(ge_result, max_p, valid, 0)
train_path2_h, train_path2_t, train_path2_r = get_path_index(ge_result, max_p, train, 1)
valid_path2_h, valid_path2_t, valid_path2_r = get_path_index(ge_result, max_p, valid, 1)
train_path3_h, train_path3_t, train_path3_r = get_path_index(ge_result, max_p, train, 2)
valid_path3_h, valid_path3_t, valid_path3_r = get_path_index(ge_result, max_p, valid, 2)


# In[ ]:





# In[101]:


train_meta_path = []
valid_meta_path = []

for tt in train_triple:
    train_h, train_t, train_r = tt[0], tt[1], tt[2]
    train_type = [map4labels[ent2label[id2ent[train_h]]], map4labels[ent2label[id2ent[train_t]]], map4labels[rel2label[id2rel[train_r]]]]
    train_meta_path.append(train_type)
    
for tt in valid_triple:
    valid_h, valid_t, valid_r = tt[0], tt[1], tt[2]
    valid_type = [map4labels[ent2label[id2ent[valid_h]]], map4labels[ent2label[id2ent[valid_t]]], map4labels[rel2label[id2rel[valid_r]]]]
    valid_meta_path.append(valid_type)


# In[160]:


# dataset = [ent2id, 
#            id2ent, 
#            rel2id, 
#            id2rel, 
#            entity2vec, 
#            entvec_k, 
#            relation2vec, 
#            relvec_k, 
#            train_triple, 
#            train_confidence, 
#            test_triple, 
#            test_confidence, 
#            tcthreshold_dict, 
#            train_transE, 
#            test_transE,
#            rrkthreshold_dict,
#            train_rrank,
#            test_rrank,
#            max_p,
#            train_path_h, 
#            train_path_t, 
#            train_path_r, 
#            test_path_h, 
#            test_path_t, 
#            test_path_r,
#            train_path2_h, 
#            train_path2_t, 
#            train_path2_r,
#            test_path2_h, 
#            test_path2_t, 
#            test_path2_r,
#            train_path3_h, 
#            train_path3_t, 
#            train_path3_r,
#            test_path3_h, 
#            test_path3_t, 
#            test_path3_r, 
#            train_meta_path, 
#            test_meta_path
#           ]


# In[35]:


dataset = [ent2id, 
           id2ent, 
           rel2id, 
           id2rel, 
           entity2vec, 
           entvec_k, 
           relation2vec, 
           relvec_k, 
           train_triple, 
           train_confidence, 
           valid_triple, 
           valid_confidence, 
           tcthreshold_dict, 
           train_transE, 
           valid_transE,
           rrkthreshold_dict,
           train_rrank,
           valid_rrank,
           max_p,
           train_path_h, 
           train_path_t, 
           train_path_r, 
           valid_path_h, 
           valid_path_t, 
           valid_path_r,
           train_path2_h, 
           train_path2_t, 
           train_path2_r,
           valid_path2_h, 
           valid_path2_t, 
           valid_path2_r,
           train_path3_h, 
           train_path3_t, 
           train_path3_r,
           valid_path3_h, 
           valid_path3_t, 
           valid_path3_r, 
#            train_meta_path, 
#            valid_meta_path
          ]


# In[66]:


def from_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f) 


# In[67]:


import pickle


# In[68]:


len(dataset)


# In[36]:


export_path = '../ttmf_dataset/{}'.format('transe_diseasekg_new_ttmf_train_valid_noise_10.pkl')


mu = ModelUtils(export_path)
mu.to_pickle(dataset, True)


# ## 构建测试集

# In[37]:


test_triples = []
for i in conf_test2id:
    test_triples.append(i)
#     valid_triples.append(i[0])
#     valid_triples.append(i[1])


# In[38]:


def get_triples_and_confidence_v(conf_test2id):
    '''
    获取验证集的三元组及其对应的置信值 
    0代表负例 1代表正例
    '''
    test = []
    for i in conf_test2id:
        test.append(i)
#         valid.append(i[0])
#         valid.append(i[1])
        
    def get_confidence(triples):
        confidence = []
        for t in triples:
                if t[-1] == 1:
                    confidence.append([0, 1])
                else:
                    confidence.append([1, 0])
        return confidence
    test_confidence = get_confidence(test)
    return test, test_confidence


# In[39]:


test, test_confidence = get_triples_and_confidence_v(conf_test2id)
test_triple, test_confidence = test, test_confidence
tcthreshold_dict = re.get_threshold_dict(test_triple)
test_transE = re.get_TransConfidence(tcthreshold_dict, test_triple)

rrkthreshold_dict = {}

test_rrank = ee.get_rrank_features(ee.get_rr_features(conf_test2id, sub_graphs, entity_ranks), test_triple)

#valid_rrank = ee.get_rrank_features_ev(ee.get_rr_features_ev(conf_valid2id, sub_graphs, entity_ranks), valid_triple)

max_p = 3
test_path_h, test_path_t,test_path_r = get_path_index(ge_result, max_p, test, 0)
test_path2_h, test_path2_t, test_path2_r = get_path_index(ge_result, max_p, test, 1)
test_path3_h, test_path3_t, test_path3_r = get_path_index(ge_result, max_p, test, 2)


# In[88]:




# In[40]:


test_dataset = [test_triple, 
                 test_confidence, 
                 test_transE, 
                 test_rrank, 
                 test_path_h, 
                 test_path_t, 
                 test_path_r, 
                 test_path2_h, 
                 test_path2_t, 
                 test_path2_r, 
                 test_path3_h, 
                 test_path3_t, 
                 test_path3_r,
#                  test_meta_path
                ]


# In[41]:


export_path = '../ttmf_dataset/{}'.format('transe_diseasekg_new_ttmf_test_noise_10.pkl')

mu = ModelUtils(export_path)
mu.to_pickle(test_dataset, True)


# In[ ]:




