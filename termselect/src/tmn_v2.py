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

        self.RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU}
        print ("###########self.args.matching_order:  %s "%(self.args.matching_order))

        # RNN context encoder
        #rnn_input_size  = self.embedding_dim+ args.pos_emb_dim + args.ner_emb_dim +5+ 2*args.rel_emb_dim
        rnn_input_size  = self.embedding_dim+ args.pos_emb_dim + args.ner_emb_dim +5+ 2*args.rel_emb_dim
        #rnn_input_size  = self.embedding_dim+ 5
        self.context_rnn = layers.StackedBRNN(
            input_size=rnn_input_size,
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
            self.c_infer_linear= nn.Linear(3*2*choice_infer_hidden_size + 3*2*args.hidden_size ,args.hidden_size)

        self.logits_linear= nn.Linear(args.hidden_size,1)



    def forward(self, p, p_pos, p_ner, p_mask, q, q_pos, q_ner, q_mask, c,c_pos,c_ner, c_mask,\
               p_f_tensor,q_f_tensor,c_f_tensor, p_q_relation, p_c_relation,q_p_relation,q_c_relation,c_p_relation,c_q_relation,is_paint=0):

        p_rnn_input, q_rnn_input, c_rnn_input = self.add_embeddings(p, p_pos, p_ner, q, q_pos, q_ner, c,c_pos,c_ner,p_f_tensor,q_f_tensor,c_f_tensor,\
                       p_q_relation, p_c_relation,q_p_relation,q_c_relation,c_p_relation,c_q_relation)

        p_hiddens = self.context_rnn(p_rnn_input, p_mask)
        q_hiddens = self.context_rnn(q_rnn_input, q_mask)
        c_hiddens = self.context_rnn(c_rnn_input, c_mask)
        if self.args.dropout_rnn_output > 0:
            p_hiddens = nn.functional.dropout(p_hiddens, p=self.args.dropout_rnn_output, training=self.training)
            q_hiddens = nn.functional.dropout(q_hiddens, p=self.args.dropout_rnn_output, training=self.training)
            c_hiddens = nn.functional.dropout(c_hiddens, p=self.args.dropout_rnn_output, training=self.training)

        ####################################################
        #--------------naive attention
        _,p_q_weighted_hiddens = self.hidden_match(p_hiddens,q_hiddens,q_mask)
        _,p_c_weighted_hiddens = self.hidden_match(p_hiddens,c_hiddens,c_mask)

        _,q_p_weighted_hiddens = self.hidden_match(q_hiddens,p_hiddens,p_mask)
        _,q_c_weighted_hiddens = self.hidden_match(q_hiddens,c_hiddens,c_mask)

        _,c_p_weighted_hiddens = self.hidden_match(c_hiddens,p_hiddens,p_mask)
        _,c_q_weighted_hiddens = self.hidden_match(c_hiddens,q_hiddens,q_mask)

        #--------------compound attention
        c_q_p_weighted_hiddens = self.hidden_match(c_hiddens, q_p_weighted_hiddens, q_mask)
        q_c_p_weighted_hiddens = self.hidden_match(q_hiddens, c_p_weighted_hiddens, c_mask)

        p_c_q_weighted_hiddens = self.hidden_match(p_hiddens, c_q_weighted_hiddens, c_mask)
        c_p_q_weighted_hiddens = self.hidden_match(c_hiddens, p_q_weighted_hiddens, p_mask)

        p_q_c_weighted_hiddens = self.hidden_match(p_hiddens, q_c_weighted_hiddens, q_mask)
        q_p_c_weighted_hiddens = self.hidden_match(q_hiddens, p_c_weighted_hiddens, p_mask)

        #------p_c_q--------------
        p_infer_emb,p_mems,p_mem_gates = self.tri_matching(x = p_hiddens,
                                             x_y = p_q_weighted_hiddens,
                                             x_z = p_c_weighted_hiddens,
                                             agg_function=self.mtinfer,
                                             x_mask = p_mask)

        #------q_p_c--------------
        q_infer_emb,q_mems, q_mem_gates = self.tri_matching(x = q_hiddens,
                                              x_y = q_p_weighted_hiddens,
                                              x_z = q_c_weighted_hiddens,
                                              agg_function=self.mtinfer,
                                              x_mask = q_mask)
        
        #------c_p_q--------------
        c_infer_emb,c_mems, c_mem_gates = self.tri_matching(x = c_hiddens,
                                              x_y = c_p_weighted_hiddens,
                                              x_z = c_q_weighted_hiddens,
                                              agg_function=self.mtinfer,
                                              x_mask = c_mask)

        #------matched_self--------------
        matched_p = self.matched_self(x = p_hiddens,
                                      x_y = p_q_weighted_hiddens,
                                      x_z = p_c_weighted_hiddens,
                                      x_mask = p_mask)
        matched_q = self.matched_self(x = q_hiddens,
                                      x_y = q_p_weighted_hiddens,
                                      x_z = q_c_weighted_hiddens,
                                      x_mask = q_mask)
        matched_c = self.matched_self(x = c_hiddens,
                                      x_y = c_p_weighted_hiddens,
                                      x_z = c_q_weighted_hiddens,
                                      x_mask = c_mask)

        #------output-layer--------------
        _,matched_q_self = self.q_self_attn(matched_q,q_mask) 
        _,matched_p_self = self.q_self_attn(matched_p,p_mask) 
        _,matched_c_self = self.q_self_attn(matched_c,c_mask) 

        p_infer_hidden_ave = layers.ave_pooling(p_infer_emb,p_mask)
        p_infer_hidden_max = layers.max_pooling(p_infer_emb)

        q_infer_hidden_ave = layers.ave_pooling(q_infer_emb,q_mask)
        q_infer_hidden_max = layers.max_pooling(q_infer_emb)

        c_infer_hidden_ave = layers.ave_pooling(c_infer_emb,c_mask)
        c_infer_hidden_max = layers.max_pooling(c_infer_emb)

        #import pdb
        #pdb.set_trace()
        infer_linear = self.c_infer_linear(torch.cat([p_infer_hidden_ave,p_infer_hidden_max,\
                                                      q_infer_hidden_ave,q_infer_hidden_max,\
                                                      c_infer_hidden_ave,c_infer_hidden_max,\
                                                      matched_q_self,matched_p_self, matched_c_self],-1)) 

        logits = self.logits_linear(infer_linear) 
        proba = F.sigmoid(logits.squeeze(1))
      
        return proba

    def add_embeddings(self, p, p_pos, p_ner, q, q_pos, q_ner, c,c_pos,c_ner,p_f_tensor,q_f_tensor,c_f_tensor,\
                       p_q_relation, p_c_relation,q_p_relation,q_c_relation,c_p_relation,c_q_relation):

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

        p_rnn_input = torch.cat([p_emb,p_pos_emb,p_ner_emb,p_f_tensor,p_q_rel_emb,p_c_rel_emb],2)
        q_rnn_input = torch.cat([q_emb,q_pos_emb,q_ner_emb,q_f_tensor,q_p_rel_emb,q_c_rel_emb],2)
        c_rnn_input = torch.cat([c_emb,c_pos_emb,c_ner_emb,c_f_tensor,c_p_rel_emb,c_q_rel_emb],2)

        return p_rnn_input, q_rnn_input, c_rnn_input



    def tri_matching(self,x,x_y,x_z, agg_function, x_mask):
        '''
        x,x_y,x_z = l*2d
        agg_function: the object of MultiTurnInference
        
        '''
        concat_feature = torch.cat([x, x_y, x_z],2)
        sub_feature =  (x - x_y)*(x - x_z)
        mul_feature = x*x_y*x_z
    
        x_mfeature = {"c":concat_feature, "s":sub_feature, "m":mul_feature }
        dim = x.size()
        init_mem = torch.zeros(dim[0],dim[1],dim[2]).float().cuda()  #zero mem
        x_infer_emb, x_mem_list, x_mem_gate_list = agg_function(x_mfeature,x_mask,init_mem=init_mem,x_order=self.args.matching_order)
 
        return x_infer_emb, x_mem_list, x_mem_gate_list
          

    def matched_self(self,x,x_y,x_z, x_mask):
    
        x_yz_cat = torch.cat([x, x_y, x_z],2)
        x_yz_weight,x_yz_weighted_hiddens = self.hidden_match(x_yz_cat, x_yz_cat, x_mask)

        if self.args.dropout_att_score > 0:
            x_yz_weight = nn.functional.dropout(x_yz_weight, p=self.args.dropout_att_score, training= self.training)

        matched_x = x_yz_weight.bmm(x)

        return matched_x

