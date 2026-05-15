from torch import nn
from .embedding import LeftEmbedding, RightEmbedding, GGNNSum
from .encoder import Encoder
import torch
import math
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args, s_vocab, t_vocab):
        super().__init__()
        self.args = args
        self.left_embedding = LeftEmbedding(args, s_vocab)
        self.right_embedding = RightEmbedding(args, t_vocab)
        if args.relation_graph or args.absolute_graph:
            self.graph_embedding = GGNNSum(input_dim=100, output_dim=args.graph_embedding_size, num_steps=6, max_edge_types=2)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=args.hidden, nhead=args.attn_heads, dim_feedforward=args.d_ff_fold * args.hidden, dropout=args.dropout, activation=self.args.activation)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=args.decoder_layers)
        self.encoder = Encoder(args)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.relation_graph = args.relation_graph
        self.absolute_graph = args.absolute_graph
        if self.args.uni_vocab:
            self.right_embedding.embedding.weight = self.left_embedding.embedding.weight
            if args.embedding_size != args.hidden:
                self.right_embedding.in_.weight = self.left_embedding.in_.weight
        if self.args.weight_tying:
            self.right_embedding.out.weight = self.right_embedding.embedding.weight
        if self.args.pointer:
            self.query_linear = nn.Linear(self.args.hidden, self.args.hidden)
            self.sentinel = nn.Parameter(torch.rand(1, self.args.hidden))
            if self.args.pointer_res:
                self.drop = nn.Dropout(p=self.args.dropout)
            if self.args.activation == 'gelu':
                self.activation = torch.nn.GELU()
            elif self.args.activation == 'relu':
                self.activation = torch.nn.ReLU()
            if self.args.pointer_type == 'add':
                self.additive_attention_W = nn.Linear(self.args.hidden * 2, self.args.hidden)
                self.additive_attention_v = nn.Parameter(torch.rand(self.args.hidden))

    def encode(self, data):
        content = data['content']
        content_mask = data['content_mask']
        
        graph_batch = data['graph_batch']
        #abs_subgraph_batch = data['abs_subgraph_batch']
        #subgraph_batch = data['subgraph_batch']
        #abs_subgraph_batch_map = data['abs_subgraph_batch_map']
        #subgraph_batch_map = data['subgraph_batch_map']
        
        named = data['named']
        
        abs_subgraph_batch_dict = data['abs_subgraph_batch_dict']
        subgraph_batch_dict = data['subgraph_batch_dict']
        
        content_ = self.left_embedding(content, named)
        if self.relation_graph:
            batch_node_embeddings = self.graph_embedding(graph_batch, cuda=True, device=content.device.index)
            #graph_embeddings = self.graph_embedding(subgraph_batch, cuda=True, device=content.device.index)
        else:
            #graph_embeddings = None
            batch_node_embeddings = None

        #graph_embeddings_ = torch.zeros(len(content), self.args.max_subgraph_num, self.args.graph_embedding_size).to(graph_embeddings.device)
        #if self.relation_graph:
        graph_embeddings_ = torch.zeros(len(content), self.args.max_subgraph_num, self.args.graph_embedding_size).to(content.device)
        subgraph_batch_map_ = torch.ones(len(content), len(content[0]), len(content[0]), dtype=int).to(content.device)
        subgraph_batch_map_ = subgraph_batch_map_ * self.args.max_subgraph_num
        #else:
        #    graph_embeddings_ = None
        #    subgraph_batch_map_ = None
        #subgraph_batch_map_ = torch.ones(len(content), len(content[0]), len(content[0]), dtype=int).to(graph_embeddings.device)
        if self.relation_graph:      
            for sample_id, subgraph_dict in enumerate(subgraph_batch_dict):
                subgraph_idx = 0
                for token_id in subgraph_dict:
                    subgraph_node_ids = subgraph_dict[token_id]
                    if len(subgraph_node_ids)==0:
                        continue
                    try:
                        idx=torch.tensor(subgraph_node_ids).to(content.device)
                        graph_embeddings_[sample_id][subgraph_idx] = torch.index_select(batch_node_embeddings[sample_id],0,idx).sum(dim=0)
                        subgraph_batch_map_[sample_id][token_id[0]][token_id[1]] = subgraph_idx
                        subgraph_idx += 1
                    except:
                        pass
        '''
        start_point = 0
        for sample_id, subgraph_map in enumerate(subgraph_batch_map):
            for m in subgraph_map:
                try:
                    graph_embeddings_[sample_id][subgraph_map[m]-start_point] = graph_embeddings[subgraph_map[m]]
                    subgraph_batch_map_[sample_id][m[0]][m[1]] = subgraph_map[m]-start_point
                except:
                    pass
            start_point += len(subgraph_map)
        '''

        if self.absolute_graph:
            abs_batch_node_embeddings = self.graph_embedding(graph_batch, cuda=True, device=content.device.index)
            #abs_graph_embeddings = self.graph_embedding(abs_subgraph_batch, cuda=True, device=content.device.index)
        else:
            abs_batch_node_embeddings = None
            #abs_graph_embeddings = None
        #if self.absolute_graph:
        abs_graph_embeddings_ = torch.zeros(len(content), self.args.max_abs_subgraph_num, self.args.graph_embedding_size).to(content.device)
        abs_subgraph_batch_idx_ = torch.ones(len(content), len(content[0]), dtype=int).to(content.device)
        abs_subgraph_batch_idx_ = abs_subgraph_batch_idx_ * self.args.max_abs_subgraph_num
        #else:
        #    abs_graph_embeddings_ = None
        #    abs_subgraph_batch_idx_ = None
        if self.absolute_graph:
            for sample_id, abs_subgraph_dict in enumerate(abs_subgraph_batch_dict):
                abs_subgraph_idx = 0
                for token_id in abs_subgraph_dict:
                    abs_subgraph_node_ids = abs_subgraph_dict[token_id]
                    if len(abs_subgraph_node_ids)==0:
                        continue
                    try:
                        idx=torch.tensor(abs_subgraph_node_ids).to(abs_batch_node_embeddings.device)
                        abs_graph_embeddings_[sample_id][abs_subgraph_idx] = torch.index_select(abs_batch_node_embeddings[sample_id],0,idx).sum(dim=0)
                        abs_subgraph_batch_idx_[sample_id][token_id] = abs_subgraph_idx
                        abs_subgraph_idx += 1
                    except:
                        pass

        '''
        start_point = 0
        for sample_id, abs_subgraph_map in enumerate(abs_subgraph_batch_map):
            for m in abs_subgraph_map:
                try:
                    abs_graph_embeddings_[sample_id][abs_subgraph_map[m]-start_point] = abs_graph_embeddings[abs_subgraph_map[m]]
                    abs_subgraph_batch_idx_[sample_id][m] = abs_subgraph_map[m]-start_point
                except:
                    pass
            start_point += len(abs_subgraph_map)
        '''
        mask_ = (content_mask > 0).unsqueeze(1).repeat(1, content_mask.size(1), 1).unsqueeze(1)
        # bs, 1,max_code_length,max_code_length
        memory = self.encoder(content_, mask_, graph_embeddings_, subgraph_batch_map_, abs_graph_embeddings_, abs_subgraph_batch_idx_)
        # bs, max_code_length, hidden
        return memory, (content_mask == 0)

    def pointer(self, out, feature, memory, memory_key_padding_mask, content_e, voc_len):
        voc_len = torch.max(voc_len).item()
        bs, src_len, tgt_len = memory.shape[0], memory.shape[1], feature.shape[1]
        pointer_key = torch.cat((memory, self.sentinel.unsqueeze(0).expand(bs, -1, -1)), dim=1)  # bs,src_len,hid
        pointer_query = self.activation((self.query_linear(feature)))  # bs,tgt,hid
        if self.args.pointer_res:
            pointer_query = self.drop(pointer_query) + feature
        if self.args.pointer_type == 'mul':
            pointer_atten = torch.einsum('bth,bsh->bts', pointer_query, pointer_key) / math.sqrt(self.args.hidden)
        elif self.args.pointer_type == 'add':
            pointer_query = pointer_query.unsqueeze(2).repeat(1, 1, pointer_key.shape[1], 1)  # bs,tgt,src_len,hid
            pointer_key = pointer_key.unsqueeze(1).repeat(1, tgt_len, 1, 1)  # bs,tgt,src_len,hid
            pointer_atten = self.activation(
                self.additive_attention_W(torch.cat([pointer_query, pointer_key], dim=-1)))  # bs,tgt,src_len,hid
            pointer_atten = torch.einsum('btsh,h->bts', pointer_atten, self.additive_attention_v)  # bs,tgt,src_len
        else:
            pointer_atten = None
        mask = torch.cat((memory_key_padding_mask, torch.ones(bs, 1).to(memory_key_padding_mask.device) == 0),
                         dim=-1).unsqueeze(1)  # bs,1,s
        pointer_atten = pointer_atten.masked_fill(mask, -1e9)
        pointer_atten = F.log_softmax(pointer_atten, dim=-1)
        pointer_gate = pointer_atten[:, :, -1].unsqueeze(-1)  # b,t,1
        pointer_atten = pointer_atten[:, :, :-1]  # b,t,s
        M = torch.zeros((bs, voc_len, src_len))
        # print(content_e.shape)
        # print(bs, voc_len, src_len)
        M[torch.arange(bs).unsqueeze(-1).expand(bs, src_len).reshape(-1),
          content_e.view(-1),
          torch.arange(src_len).repeat(bs)] = 1
        pointer_atten_p = torch.einsum('bts,bvs->btv', pointer_atten.exp(), M.to(pointer_atten.device))
        pointer_atten_log = (pointer_atten_p + torch.finfo(torch.float).eps).log()
        pointer_atten_log = pointer_atten_log - torch.log1p(
            -pointer_gate.exp() + torch.finfo(torch.float).eps)  # norm
        # pointer_atten_log: bs,max_target_len,extend_vocab_size. extend_vocab_size >= vocab_size
        # Avoid having -inf in attention scores as they produce NaNs during backward pass
        pointer_atten_log[pointer_atten_log == float('-inf')] = torch.finfo(torch.float).min
        if torch.isnan(pointer_atten_log).any():
            print("NaN in final pointer attention!", pointer_atten_log)

        out = torch.cat((out, torch.zeros(bs, tgt_len, voc_len - out.shape[-1]).fill_(float('-inf')).to(out.device)),
                        dim=-1)  # not 0 , should -inf

        p = torch.stack(
            [out + pointer_gate, pointer_atten_log + (1 - pointer_gate.exp() + torch.finfo(torch.float).eps).log()],
            dim=-2)  # bs,tgt_len,2,extend_voc
        out = torch.logsumexp(p, dim=-2)
        return out

    def decode(self, memory, f_source, memory_key_padding_mask, content_e=None, voc_len=None, is_binary = False):
        '''
        :param voc_len: a list of voc len
        :param content_e: extended vocab mapped content for pointer
        :param memory_key_padding_mask
        :param memory: # bs, max_code_length, hidden
        :param f_source: # bs,max_target_len
        :return:
        '''
        f_source_ = self.right_embedding(f_source)
        f_len = f_source.shape[-1]
        tgt_mask = (torch.ones(f_len, f_len).tril_() == 0).to(memory.device)
        memory_key_padding_mask = memory_key_padding_mask.to(memory.device)
        tgt_key_padding_mask = (f_source == 0).to(memory.device) 
        if not is_binary:
            feature = self.decoder(f_source_.permute(1, 0, 2), memory.permute(1, 0, 2), tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
        else:
            feature = self.decoder2(f_source_.permute(1, 0, 2), memory.permute(1, 0, 2), tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
        feature = feature.permute(1, 0, 2)

        out = self.softmax(self.right_embedding.prob(feature))
        if self.args.pointer:
            out = self.pointer(out, feature, memory, memory_key_padding_mask, content_e, voc_len)
        return out

    def forward(self, data, is_binary = False):
        f_source = data['f_source']
        memory, memory_key_padding_mask = self.encode(data)
        if self.args.pointer:
            out = self.decode(memory, f_source, memory_key_padding_mask, data['content_e'], data['voc_len'], is_binary = is_binary)
        else:
            out = self.decode(memory, f_source, memory_key_padding_mask, is_binary = is_binary)
        return out


