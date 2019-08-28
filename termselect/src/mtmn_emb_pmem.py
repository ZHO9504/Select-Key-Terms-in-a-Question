import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import importlib
import layers
from utils import vocab, pos_vocab, ner_vocab, rel_vocab
import sys
from sns_visual import paint



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
        rnn_input_size  = self.embedding_dim+ args.pos_emb_dim + args.ner_emb_dim +5+ 2*args.rel_emb_dim
        #rnn_input_size  = self.embedding_dim+ args.pos_emb_dim + args.ner_emb_dim +7+ 2*args.rel_emb_dim
        #rnn_input_size  = self.embedding_dim+ 5
        self.context_rnn = layers.StackedBRNN(
            input_size=rnn_input_size,
            #input_size=self.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=0,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding)

        self.hidden_match = layers.SeqDotAttnMatch()
        self.mtinfer= layers.MultiTurnInference(args,self.RNN_TYPES)
        
        #mtinfer output size
        if args.use_multiturn_infer == True:
            choice_infer_hidden_size = 2 * args.hidden_size
            if args.use_bimemory==True: 
                choice_infer_hidden_size = 2*2*args.hidden_size
            #choice_infer_hidden_size = 2 * args.hidden_size * len(args.matching_order)
        else:
            #choice_infer_hidden_size = args.hidden_size * len(args.matching_order)
            choice_infer_hidden_size = 2 * args.hidden_size

        #self.slfp_linear = layers.BilinearVecSeqAttn(x_size = 2*args.hidden_size, y_size = 2*args.hidden_size)
        #self.c_infer_self_attn = layers.LinearSeqAttn(choice_infer_hidden_size)
        self.q_self_attn = layers.LinearSeqAttn(2*args.hidden_size)

        #self.c_infer_linear= nn.Linear(2*choice_infer_hidden_size + 2*2*args.hidden_size,args.hidden_size)
        #self.c_infer_linear= nn.Linear(2*choice_infer_hidden_size + 2*args.hidden_size,args.hidden_size)
        self.c_infer_linear= nn.Linear(2*choice_infer_hidden_size + 2*2*args.hidden_size,args.hidden_size)
        #self.c_infer_linear= nn.Linear(2*choice_infer_hidden_size + 2*args.hidden_size,args.hidden_size)
        #self.c_infer_linear= nn.Linear(3*choice_infer_hidden_size,args.hidden_size)
        self.logits_linear= nn.Linear(args.hidden_size,1)



    def forward(self, p, p_pos, p_ner, p_mask, q, q_pos, q_ner, q_mask, c,c_pos,c_ner, c_mask,\
               p_f_tensor,q_f_tensor,c_f_tensor, p_q_relation, p_c_relation,q_p_relation,q_c_relation,c_p_relation,c_q_relation,is_paint=0):

        self.p = p
        self.q = q
        self.c = c
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
        #q_rnn_input = torch.cat([q_emb,q_f_tensor],2)
        #c_rnn_input = torch.cat([c_emb,c_f_tensor],2)

        p_hiddens = self.context_rnn(p_rnn_input, p_mask)
        q_hiddens = self.context_rnn(q_rnn_input, q_mask)
        c_hiddens = self.context_rnn(c_rnn_input, c_mask)
        # print('p_hiddens', p_hiddens.size())

        ####################################################
        #c_p_weighted_hiddens = self.hidden_match(c_hiddens,p_hiddens,p_mask)
        _,c_q_weighted_hiddens = self.hidden_match(c_hiddens,q_hiddens,q_mask)
        #------q_p--------------
        _,q_p_weighted_hiddens = self.hidden_match(q_hiddens,p_hiddens,p_mask)
        q_p_cat = torch.cat([q_hiddens,q_p_weighted_hiddens],2)
        q_p_cat_weight,q_p_cat_weighted_hiddens = self.hidden_match(q_p_cat,q_p_cat,q_mask)
        matched_q = q_p_cat_weight.bmm(q_hiddens)
        #q_q_cat = torch.cat([q_hiddens, matched_q])

        #------p_c_q--------------
        _,p_c_weighted_hiddens = self.hidden_match(p_hiddens,c_hiddens,c_mask)
        _,p_q_weighted_hiddens = self.hidden_match(p_hiddens,q_hiddens,q_mask)

        #p_cq_cat = torch.cat([p_hiddens,p_c_weighted_hiddens,p_q_weighted_hiddens],2)
        p_cq_cat = torch.cat([(p_hiddens-p_c_weighted_hiddens)*(p_hiddens -p_q_weighted_hiddens)],2)

        p_cq_cat_weight,p_cq_cat_weighted_hiddens = self.hidden_match(p_cq_cat,p_cq_cat,p_mask)

        matched_p = p_cq_cat_weight.bmm(p_hiddens)

        self.matched_p_self_weight,matched_p_self = self.q_self_attn(matched_p,p_mask) 
        #if self.args.dropout_init_mem_emb > 0:
        #    matched_p_self = nn.functional.dropout(matched_p_self, p=self.args.dropout_init_mem_emb, training=self.training)
        self.c_weighted_matched_p_weight,c_weighted_matched_p = self.hidden_match(c_hiddens,matched_p,p_mask)
        #print(self.c_weighted_matched_p_weight)
        #_, q_slfp_weighted_hiddens = self.slfp_linear(x=q_hiddens,y= matched_p_self,x_mask =q_mask)
        #_,c_weighted_q_slfp = self.hidden_match(c_hiddens,q_slfp_weighted_hiddens ,q_mask)
        #print ("q_slfp_weighted_hiddens ",q_slfp_weighted_hiddens.size() )
        #------c_q--------------
        #concat_feature = torch.cat([c_hiddens,c_q_weighted_hiddens],2)
        #sub_feature =  (c_hiddens -c_q_weighted_hiddens)
        #mul_feature = c_hiddens*c_q_weighted_hiddens
        #concat_feature = torch.cat([c_hiddens,c_q_weighted_hiddens],2)
        concat_feature = torch.cat([c_hiddens,c_q_weighted_hiddens],2)
        #concat_feature = c_hiddens+ c_q_weighted_hiddens
        sub_feature =  (c_hiddens -c_q_weighted_hiddens)
        mul_feature = self.args.beta*c_hiddens*c_q_weighted_hiddens
        #mul_feature = c_hiddens*c_q_weighted_hiddens
        #mul_feature = c_hiddens+c_q_weighted_hiddens
       
        c_mfeature = {"c":concat_feature, "s":sub_feature, "m":mul_feature }
        #c_infer_emb = self.mtinfer(c_mfeature,c_mask,x_order=self.args.matching_order,init_mem=c_weighted_matched_p)
        dim = c_hiddens.size()
        #init_mem = torch.zeros(dim[0],dim[1],dim[2]).float().cuda()  #zero mem
        #init_mem = matched_p_self.unsqueeze(1).expand(c_hiddens.size())  #p_self mem
        init_mem = c_weighted_matched_p  #c_weighted_matched_p mem  ,best 
        if self.args.dropout_init_mem_emb > 0:
            init_mem = nn.functional.dropout(init_mem, p=self.args.dropout_init_mem_emb, training=self.training)
        c_infer_emb,self.mem_list, self.mem_gate_list = self.mtinfer(c_mfeature,c_mask,init_mem=init_mem,x_order=self.args.matching_order)
        self.c_infer_emb=c_infer_emb
        #c_infer_emb = self.mtinfer(c_mfeature,c_mask,init_mem=c_weighted_matched_p,x_order=self.args.matching_order)

        #if self.args.dropout_emb > 0:
        #    c_infer_emb = nn.functional.dropout(c_infer_emb, p=self.args.dropout_emb, training=self.training)
        #c_infer_hidden_self = self.c_infer_self_attn(c_infer_emb,c_mask) 
        #c_infer_hidden_self = self.q_self_attn(c_infer_emb,c_mask) 
        self.matched_q_self_weight, matched_q_self = self.q_self_attn(matched_q,q_mask) 
        #matched_q_ave = layers.ave_pooling(matched_q,q_mask) 
        c_infer_hidden_ave = layers.ave_pooling(c_infer_emb,c_mask)
        c_infer_hidden_max = layers.max_pooling(c_infer_emb)

        #c_infer_hidden = self.c_infer_linear(torch.cat([c_infer_hidden_self, c_infer_hidden_ave, c_infer_hidden_max],-1)) 
        #logits = self.logits_linear(c_infer_hidden) 
        #print ("c_infer_hidden_ave",c_infer_hidden_ave.size())
        #print ("c_infer_hidden_max",c_infer_hidden_max.size())
        #print ("matched_p_self",matched_p_self.size())
        #print ("matched_q_self",matched_q_self.size())
        infer_linear = self.c_infer_linear(torch.cat([c_infer_hidden_ave,c_infer_hidden_max,matched_p_self,matched_q_self],-1)) 
        #infer_linear = self.c_infer_linear(torch.cat([c_infer_hidden_ave,c_infer_hidden_max,matched_p_self],-1)) 
        #infer_linear = self.c_infer_linear(torch.cat([c_infer_hidden_ave,c_infer_hidden_self,matched_p_self,matched_q_self],-1)) 
        #infer_linear = self.c_infer_linear(torch.cat([c_infer_hidden_self,matched_p_self,matched_q_self],-1)) 

        logits = self.logits_linear(infer_linear) 
        proba = F.sigmoid(logits.squeeze(1))
     
        if is_paint==1:
            self.paint_data()
         
        return proba

    def paint_data(self):
        
        paint('c_infer_emb',data_id = self.c, data_vec=self.c_infer_emb)
        paint('mem0',data_id = self.c, data_vec=self.mem_list[0])
        paint('mem1',data_id = self.c, data_vec=self.mem_list[1])
        paint('mem2',data_id = self.c, data_vec=self.mem_list[2])

        paint("gate0", data_id=self.c, data_vec=self.mem_gate_list[0]) 
        paint("gate1", data_id=self.c, data_vec=self.mem_gate_list[1]) 
        paint("gate2", data_id=self.c, data_vec=self.mem_gate_list[2]) 
       
        #paint("w_matched_cp", data_id=self.p, data_vec=self.c_weighted_matched_p_weight) 
        #paint("w_matched_p", data_id=self.p, data_vec=self.matched_p_self_weight.unsqueeze(-1)) 
        
        #paint("w_matched_q", data_id=self.q, data_vec=self.matched_q_self_weight.unsqueeze(-1)) 
       
         
