from torch.utils.data import Dataset
import os
import torch
import pickle
import copy
from .process_utils import  decoder_process, row_process, content_process, make_extended_vocabulary
from dgl import DGLGraph
from .batch_graph import GGNNBatchGraph
from gensim.models.fasttext import FastText
import random
random.seed(1000)
fasttext_model = FastText.load('c_model.pt')

class GraphAttenDataset(Dataset):
    def __init__(self, args, s_vocab, t_vocab, type_):
        self.on_memory = args.on_memory
        #self.dataset_dir = os.path.join('./data', args.dataset)
        self.dataset_dir = './data/'
        self.s_vocab = s_vocab
        self.t_vocab = t_vocab
        self.args = args
        self.type_ = type_
        assert type_ in ['train', 'test', 'valid', 'binary_train', 'binary_test']
        '''
        self.pkl_path = './data_code_ast/dataset_code_ast.pkl'
        f = open(self.pkl_path, 'rb')
        self.data = pickle.load(f)[:-100]
        '''
        self.data = []
        if type_ == 'train': 
            data_files = os.listdir('./data_training_aug/')
            for iii,data_file in enumerate(data_files):
                #if '300000.pkl' in data_file:
                #    continue
                if '.pkl' in data_file and not '_it' in data_file:
                    print('reading '+data_file)
                    f = open('./data_training_aug/'+data_file, 'rb')
                    self.data.extend(pickle.load(f))
            
            # For overfitting test
            '''
            data_file = 'dataset_vulgen_test2.pkl'
            print('reading '+data_file)
            f = open('./data/'+data_file, 'rb')
            test_objs=pickle.load(f)
            for i in range(100):
                self.data.extend(test_objs)     
            '''
        '''
        if type_ == 'binary_train':
            data_files = os.listdir('./data_pretrain_codet5/')
            for data_file in data_files:
                if '300000.pkl' in data_file:
                    continue
                if '.pkl' in data_file and '_it' in data_file:
                    print('reading '+data_file)
                    f = open('./data_pretrain_codet5/'+data_file, 'rb')
                    self.data.extend(pickle.load(f))
                    #break
        
        if type_ == 'binary_test': 
            data_file = 'dataset_pretraining_it300000.pkl'
            print('reading '+data_file)
            f = open('./data_pretrain_codet5/'+data_file, 'rb')
            self.data.extend(pickle.load(f))
        '''
        if type_ == 'test' or type_ == 'valid':
            '''
            data_file = 'dataset_vulgen_test2.pkl'
            #data_file = 'dataset_vulgen_train0.pkl'     #check overfitting
            print('reading '+data_file)
            f = open('./data/'+data_file, 'rb')
            #f = open('./data_training/'+data_file, 'rb')
            self.data.extend(pickle.load(f))
            '''
            '''
            data_file = 'dataset_pretraining_msp300000.pkl'
            print('reading '+data_file)
            f = open('./data_pretrain_codet5/'+data_file, 'rb')
            self.data.extend(pickle.load(f))
            data_file = 'dataset_pretraining_mip300000.pkl'
            print('reading '+data_file)
            f = open('./data_pretrain_codet5/'+data_file, 'rb')
            self.data.extend(pickle.load(f))
            '''
            '''
            data_files = os.listdir('./data_loc/')
            for data_file in data_files:
                if 'dataset_loc300000' in data_file and '.pkl' in data_file:
                    print('reading '+data_file)
                    f = open('./data_loc/'+data_file, 'rb')
                    self.data.extend(pickle.load(f))
            '''
            data_files = os.listdir('./data/')
            for data_file in data_files:
                if '.pkl' in data_file:
                    print('reading '+data_file)
                    f = open('./data/'+data_file, 'rb')
                    self.data.extend(pickle.load(f))
                #if type_ == 'valid':
                #    break
        self.corpus_line = len(self.data)
        print('dataset size:', self.corpus_line)
        
    def __len__(self):
        return self.corpus_line

    def __getitem__(self, item):
        assert item < self.corpus_line
        data = self.data[item]
        sample = self.process(data)
        return {key: value if torch.is_tensor(value) or isinstance(value, dict) or isinstance(value, str) else torch.tensor(value) for key, value in sample.items()}

    def process(self, data):
        if self.args.pointer:
            assert self.args.uni_vocab, 'separate vocab not support'
            # without AST test
            #e_voc, e_voc_, voc_len = make_extended_vocabulary(data['code_tokens'], self.s_vocab)
            e_voc, e_voc_, voc_len = make_extended_vocabulary(data['input_sequence_tokens'], self.s_vocab)
        else:
            e_voc, e_voc_, voc_len = None, None, None
        f_source, f_target = decoder_process(data['target_tokens'], self.t_vocab, self.args.max_target_len,
                                             e_voc, self.args.pointer)
        row_ = []
        # without AST test
        #content_, content_mask_, named_, content_e = content_process(data['code_tokens'], data['named'], self.s_vocab, self.args.max_code_length, e_voc, self.args.pointer)
        content_, content_mask_, named_, content_e = content_process(data['input_sequence_tokens'], data['named'], self.s_vocab, self.args.max_code_length, e_voc, self.args.pointer)
        
        if 'idx' in data:
            data_dic = {'idx': data['idx'], 'f_source': f_source, 'f_target': f_target, 'content': content_, 'content_mask': content_mask_, 'named': named_, 'row': [], 'abs_subgraph_dict': data['abs_subgraph_dict'], 'subgraph_dict': data['subgraph_dict'], 'whole_graph': data['whole_graph']}
        else:
            data_dic = {'idx': '', 'f_source': f_source, 'f_target': f_target, 'content': content_, 'content_mask': content_mask_, 'named': named_, 'row': [], 'abs_subgraph_dict': data['abs_subgraph_dict'], 'subgraph_dict': data['subgraph_dict'], 'whole_graph': data['whole_graph']}
        
        if self.args.pointer:
            data_dic['e_voc'] = e_voc
            data_dic['e_voc_'] = e_voc_
            data_dic['voc_len'] = voc_len
            data_dic['content_e'] = content_e
        return data_dic

    
def collect_fn(batch):
    data = dict()
    max_content_len, max_target_len = 0, 0
    for sample in batch:
        c_l = torch.count_nonzero(sample['content_mask']).item()
        f_l = torch.count_nonzero(sample['f_source']).item()
        if c_l > max_content_len: max_content_len = c_l
        if f_l > max_target_len: max_target_len = f_l
    data['idx']=[]
    for b in batch:
        data['idx'].append(b['idx'])

    data['f_source'] = torch.stack([b['f_source'] for b in batch], dim=0)[:, :max_target_len]
    data['f_target'] = torch.stack([b['f_target'] for b in batch], dim=0)[:, :max_target_len]
    data['content'] = torch.stack([b['content'] for b in batch], dim=0)[:, :max_content_len]
    data['content_mask'] = torch.stack([b['content_mask'] for b in batch], dim=0)[:, :max_content_len]
    #data['abs_subgraph_batch'] = GGNNBatchGraph()
    #data['abs_subgraph_batch_map'] = [] # n:m means nth token in content is respective to mth subgraph in subgraph bacth 
    data['graph_batch'] = GGNNBatchGraph()
    data['abs_subgraph_batch_dict'] = []
    data['subgraph_batch_dict'] = []
    for b in batch:
        abs_subgraph_map = {}
        whole_graph = b['whole_graph']
        whole_graph['node_features'] = []
        for node_token in whole_graph['node_tokens']:
            whole_graph['node_features'].append(fasttext_model.wv[node_token])
        dgl_graph = DGLGraph()
        features = torch.FloatTensor(whole_graph['node_features'])
        dgl_graph.add_nodes(len(whole_graph['node_features']), data={'features': features})
        for s, _type, t in whole_graph['graph']:
            dgl_graph.add_edge(s, t, data={'etype': torch.LongTensor([_type])})
        data['graph_batch'].add_subgraph(copy.deepcopy(dgl_graph))
        data['abs_subgraph_batch_dict'].append(b['abs_subgraph_dict'])
        data['subgraph_batch_dict'].append(b['subgraph_dict'])
        '''
        for token_id in b['abs_subgraph_dict']:
            subgraph_node_ids = b['abs_subgraph_dict'][token_id]
            node_features = []
            graph = []
            node_map = {}
            for node_id in subgraph_node_ids:
                #node_features.append(fasttext_model.wv[whole_graph['node_tokens'][node_id]])
                node_features.append(whole_graph['node_features'][node_id])
                node_map[node_id]=len(node_features)-1
            for edge in whole_graph['graph']:
                if edge[0] in node_map and edge[2] in node_map:
                    graph.append([node_map[edge[0]], edge[1], node_map[edge[2]]])
            subgraphx = {'node_features': node_features, 'graph': graph}

            subgraph = DGLGraph()
            features = torch.FloatTensor(subgraphx['node_features'])
            subgraph.add_nodes(len(subgraphx['node_features']), data={'features': features})
            for s, _type, t in subgraphx['graph']:
                subgraph.add_edge(s, t, data={'etype': torch.LongTensor([_type])})
            data['abs_subgraph_batch'].add_subgraph(copy.deepcopy(subgraph))
            abs_subgraph_map[token_id] = data['abs_subgraph_batch'].num_of_subgraphs - 1
        data['abs_subgraph_batch_map'].append(abs_subgraph_map)
        '''
    '''
    data['subgraph_batch'] = GGNNBatchGraph()
    data['subgraph_batch_map'] = []
    for b in batch:
        subgraph_map = {}
        whole_graph = b['whole_graph']
        for token_id in b['subgraph_dict']:
            subgraph_node_ids = b['subgraph_dict'][token_id]
            if len(subgraph_node_ids)==0:
                continue
            node_features = []
            graph = []
            node_map = {}
            for node_id in subgraph_node_ids:
                #node_features.append(fasttext_model.wv[whole_graph['node_tokens'][node_id]])
                node_features.append(whole_graph['node_features'][node_id])
                node_map[node_id]=len(node_features)-1
            for edge in whole_graph['graph']:
                if edge[0] in node_map and edge[2] in node_map:
                    graph.append([node_map[edge[0]], edge[1], node_map[edge[2]]])
            subgraphx = {'node_features': node_features, 'graph': graph}
            subgraph = DGLGraph()
            features = torch.FloatTensor(subgraphx['node_features'])
            subgraph.add_nodes(len(subgraphx['node_features']), data={'features': features})
            for s, _type, t in subgraphx['graph']:
                subgraph.add_edge(s, t, data={'etype': torch.LongTensor([_type])})
            
            data['subgraph_batch'].add_subgraph(copy.deepcopy(subgraph))
            subgraph_map[token_id] = data['subgraph_batch'].num_of_subgraphs - 1
        data['subgraph_batch_map'].append(subgraph_map)
    '''
    data['named'] = torch.stack([b['named'] for b in batch], dim=0)[:, :max_content_len]
    data['row'] = torch.Tensor([])
    
    if 'e_voc' in batch[0]:
        data['e_voc'] = [b['e_voc'] for b in batch]
        data['e_voc_'] = [b['e_voc_'] for b in batch]
        max_voc_len = torch.max(torch.stack([b['voc_len'] for b in batch], dim=0)).item()
        data['voc_len'] = torch.tensor(
            [max_voc_len for _ in batch])  # we set e voc len equal for all data in batch, for data parallel
        data['content_e'] = torch.stack([b['content_e'] for b in batch], dim=0)[:, :max_content_len]
    return data
