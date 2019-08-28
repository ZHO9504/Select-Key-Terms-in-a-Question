import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import layers
from utils import vocab, pos_vocab, ner_vocab, rel_vocab
import sys

class MyModel(nn.Module):

    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args
        self.embedding_dim = 300
        self.embedding = nn.Embedding(len(vocab), self.embedding_dim, padding_idx=0)
        self.embedding.weight.data.fill_(0)
        self.embedding.weight.data[:2].normal_(0, 0.1)
        
        self.pos_embedding = nn.Embedding(len(pos_vocab), args.pos_emb_dim, padding_idx=0)
        self.pos_embedding.weight.data.normal_(0, 0.1)
        self.ner_embedding = nn.Embedding(len(ner_vocab), args.ner_emb_dim, padding_idx=0)
        self.ner_embedding.weight.data.normal_(0, 0.1)
        self.rel_embedding = nn.Embedding(len(rel_vocab), args.rel_emb_dim, padding_idx=0)
        self.rel_embedding.weight.data.normal_(0, 0.1)

        #self.emb_match = layers.SeqAttnMatch(self.embedding_dim)
        #self.q_emb_match = layers.SeqAttnMatch(self.embedding_dim)
        #self.c_emb_match = layers.SeqAttnMatch(self.embedding_dim)
        #self.p_q_emb_match = layers.SeqAttnMatch(self.embedding_dim)
        #self.c_q_emb_match = layers.SeqAttnMatch(self.embedding_dim)
        #self.c_p_emb_match = layers.SeqAttnMatch(self.embedding_dim)
        self.RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU}
        print ("###########self.args.matching_order:  %s "%(self.args.matching_order))

        # RNN context encoder
        #rnn_input_size  = self.embedding_dim+ args.pos_emb_dim + args.ner_emb_dim +5+ 2*args.rel_emb_dim
        rnn_input_size  = self.embedding_dim+ args.pos_emb_dim + args.ner_emb_dim +5+ 2*args.rel_emb_dim
        #rnn_input_size  = self.embedding_dim+ 5
        self.context_rnn = layers.StackedBRNN(
            input_size=rnn_input_size,
            #input_size=self.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn_output,  # float
            dropout_output=args.rnn_output_dropout,  #True or False
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        self.hidden_match = layers.SeqDotAttnMatch()
        self.mtinfer= layers.MultiTurnInference(args,self.RNN_TYPES)
        
        #mtinfer output size
        if args.use_multiturn_infer or args.use_bilstm:
            choice_infer_hidden_size = 2 * args.hidden_size
            #choice_infer_hidden_size = 2 * args.hidden_size * len(args.matching_order)
        else:
            #choice_infer_hidden_size = args.hidden_size * len(args.matching_order)
            choice_infer_hidden_size = 2*args.hidden_size 


        #self.c_infer_self_attn = layers.LinearSeqAttn(choice_infer_hidden_size)
        self.q_self_attn = layers.LinearSeqAttn(2*args.hidden_size)

        if args.use_multiturn_infer == True:
            self.c_infer_linear= nn.Linear(4*choice_infer_hidden_size,args.hidden_size)
        #elif args.use_bilstm == True:
        else:
            self.c_infer_linear= nn.Linear(2*2*choice_infer_hidden_size + 2*2*args.hidden_size ,args.hidden_size)

        self.logits_linear= nn.Linear(args.hidden_size,1)



    def forward(self, p, p_pos, p_ner, p_mask, q, q_pos, q_ner, q_mask, c,c_pos,c_ner, c_mask,\
               p_f_tensor,q_f_tensor,c_f_tensor, p_q_relation, p_c_relation,q_p_relation,q_c_relation,c_p_relation,c_q_relation,is_paint=0):

        p_emb, q_emb, c_emb = self.embedding(p), self.embedding(q), self.embedding(c)
        p_pos_emb, q_pos_emb, c_pos_emb = self.pos_embedding(p_pos), self.pos_embedding(q_pos), self.pos_embedding(c_pos)
        p_ner_emb, q_ner_emb, c_ner_emb = self.ner_embedding(p_ner), self.ner_embedding(q_ner), self.ner_embedding(c_ner)
        p_q_rel_emb, p_c_rel_emb = self.rel_embedding(p_q_relation), self.rel_embedding(p_c_relation)
        q_p_rel_emb, q_c_rel_emb = self.rel_embedding(q_p_relation), self.rel_embedding(q_c_relation)
        c_p_rel_emb, c_q_rel_emb = self.rel_embedding(c_p_relation), self.rel_embedding(c_q_relation)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            p_emb = nn.functional.dropout(p_emb, p=self.args.dropout_emb, training=self.training)
            q_emb = nn.functional.dropout(q_emb, p=self.args.dropout_emb, training=self.training)
            c_emb = nn.functional.dropout(c_emb, p=self.args.dropout_emb, training=self.training)
            
            p_pos_emb = nn.functional.dropout(p_pos_emb, p=self.args.dropout_emb, training=self.training)
            q_pos_emb = nn.functional.dropout(q_pos_emb, p=self.args.dropout_emb, training=self.training)
            c_pos_emb = nn.functional.dropout(c_pos_emb, p=self.args.dropout_emb, training=self.training)

            p_ner_emb = nn.functional.dropout(p_ner_emb, p=self.args.dropout_emb, training=self.training)
            q_ner_emb = nn.functional.dropout(q_ner_emb, p=self.args.dropout_emb, training=self.training)
            c_ner_emb = nn.functional.dropout(c_ner_emb, p=self.args.dropout_emb, training=self.training)
        #_,q_weighted_emb = self.q_emb_match(q_emb, q_emb, q_mask)
        #_,q_c_weighted_emb = self.q_emb_match(q_emb, c_emb, c_mask)
        #_,c_p_weighted_emb = self.c_emb_match(c_emb, p_emb, p_mask)
        #_,c_weighted_emb = self.c_emb_match(c_emb, c_emb, c_mask)

        #if self.args.dropout_emb > 0:
        #    q_weighted_emb = nn.functional.dropout(q_weighted_emb, p=self.args.dropout_emb, training=self.training)
            #q_c_weighted_emb = nn.functional.dropout(q_c_weighted_emb, p=self.args.dropout_emb, training=self.training)
            #c_p_weighted_emb = nn.functional.dropout(c_p_weighted_emb, p=self.args.dropout_emb, training=self.training)
        #    c_weighted_emb = nn.functional.dropout(c_weighted_emb, p=self.args.dropout_emb, training=self.training)
        p_rnn_input = torch.cat([p_emb,p_pos_emb,p_ner_emb,p_f_tensor,p_q_rel_emb,p_c_rel_emb],2)
        q_rnn_input = torch.cat([q_emb,q_pos_emb,q_ner_emb,q_f_tensor,q_p_rel_emb,q_c_rel_emb],2)
        c_rnn_input = torch.cat([c_emb,c_pos_emb,c_ner_emb,c_f_tensor,c_p_rel_emb,c_q_rel_emb],2)

        p_hiddens = self.context_rnn(p_rnn_input, p_mask)
        q_hiddens = self.context_rnn(q_rnn_input, q_mask)
        c_hiddens = self.context_rnn(c_rnn_input, c_mask)
        if self.args.dropout_rnn_output > 0:
            p_hiddens = nn.functional.dropout(p_hiddens, p=self.args.dropout_rnn_output, training=self.training)
            q_hiddens = nn.functional.dropout(q_hiddens, p=self.args.dropout_rnn_output, training=self.training)
            c_hiddens = nn.functional.dropout(c_hiddens, p=self.args.dropout_rnn_output, training=self.training)

        ####################################################

        #------p_c_q--------------
        _,p_c_weighted_hiddens = self.hidden_match(p_hiddens,c_hiddens,c_mask)
        _,p_q_weighted_hiddens = self.hidden_match(p_hiddens,q_hiddens,q_mask)

        p_cq_cat = torch.cat([p_hiddens,p_c_weighted_hiddens,p_q_weighted_hiddens],2)
        p_cq_cat_weight,p_cq_cat_weighted_hiddens = self.hidden_match(p_cq_cat,p_cq_cat,p_mask)
        if self.args.dropout_att_score > 0:
            p_cq_cat_weight = nn.functional.dropout(p_cq_cat_weight, p=self.args.dropout_att_score, training= self.training)
        matched_p = p_cq_cat_weight.bmm(p_hiddens)

        #------q_p_c--------------
        _,q_p_weighted_hiddens = self.hidden_match(q_hiddens,p_hiddens,p_mask)
        _,q_c_weighted_hiddens = self.hidden_match(q_hiddens,c_hiddens,c_mask)

        q_p_cat = torch.cat([q_hiddens,q_p_weighted_hiddens,q_c_weighted_hiddens],2)
        q_p_cat_weight,q_p_cat_weighted_hiddens = self.hidden_match(q_p_cat,q_p_cat,q_mask)
        if self.args.dropout_att_score > 0:
            q_p_cat_weight = nn.functional.dropout(q_p_cat_weight, p=self.args.dropout_att_score, training= self.training)
        matched_q = q_p_cat_weight.bmm(q_hiddens)

        q_concat_feature = torch.cat([q_hiddens,q_c_weighted_hiddens,q_p_weighted_hiddens],2)
        q_sub_feature =  (q_hiddens -q_c_weighted_hiddens)*(q_hiddens - q_p_weighted_hiddens)
        q_mul_feature = q_hiddens*q_c_weighted_hiddens*q_p_weighted_hiddens
    
        q_mfeature = {"c":q_concat_feature, "s":q_sub_feature, "m":q_mul_feature }
        dim = q_hiddens.size()
        q_init_mem = torch.zeros(dim[0],dim[1],dim[2]).float().cuda()  #zero mem
        q_infer_emb,self.q_mem_list, self.q_mem_gate_list = self.mtinfer(q_mfeature,q_mask,init_mem=q_init_mem,x_order=self.args.matching_order)

        #------c_p_q--------------
        _,c_p_weighted_hiddens = self.hidden_match(c_hiddens,p_hiddens,p_mask)
        _,c_q_weighted_hiddens = self.hidden_match(c_hiddens,q_hiddens,q_mask)
        concat_feature = torch.cat([c_hiddens,c_q_weighted_hiddens,c_p_weighted_hiddens],2)
        sub_feature =  (c_hiddens -c_q_weighted_hiddens)*(c_hiddens - c_p_weighted_hiddens)
        mul_feature = c_hiddens*c_q_weighted_hiddens*c_p_weighted_hiddens
    
        c_mfeature = {"c":concat_feature, "s":sub_feature, "m":mul_feature }
        dim = c_hiddens.size()
        init_mem = torch.zeros(dim[0],dim[1],dim[2]).float().cuda()  #zero mem
        c_infer_emb,self.mem_list, self.mem_gate_list = self.mtinfer(c_mfeature,c_mask,init_mem=init_mem,x_order=self.args.matching_order)

        #------output-layer--------------
        _,matched_q_self = self.q_self_attn(matched_q,q_mask) 
        _,matched_p_self = self.q_self_attn(matched_p,p_mask) 
        q_infer_hidden_ave = layers.ave_pooling(q_infer_emb,q_mask)
        q_infer_hidden_max = layers.max_pooling(q_infer_emb)
        c_infer_hidden_ave = layers.ave_pooling(c_infer_emb,c_mask)
        c_infer_hidden_max = layers.max_pooling(c_infer_emb)

        #import pdb
        #pdb.set_trace()
        infer_linear = self.c_infer_linear(torch.cat([q_infer_hidden_ave,q_infer_hidden_max,\
                                                      c_infer_hidden_ave,c_infer_hidden_max,\
                                                      matched_q_self,matched_p_self],-1)) 

        logits = self.logits_linear(infer_linear) 
        proba = F.sigmoid(logits.squeeze(1))
      
        return proba
