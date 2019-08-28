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
       
        self.Hq_BiLstm = layers.StackedBRNN(
            input_size = rnn_input_size   + args.hidden_size ,
            hidden_size = args.hidden_size,
            num_layers = 1,
            dropout_rate = args.dropout_rnn_output,  # float
            dropout_output = args.rnn_output_dropout,  #True or False
            concat_layers = False,
            rnn_type = self.RNN_TYPES[args.rnn_type],
            padding = args.rnn_padding)
 
        self.hidden_match = layers.SeqDotAttnMatch()
        self.mtinfer= layers.MultiTurnInference(args,self.RNN_TYPES)

        if self.args.tri_input == 'NA':
             self.mfunction= self.NA_TriMatching

        elif self.args.tri_input == 'CA':
            self.mfunction= self.CA_TriMatching
        else:
            self.mfunction=self.NA_CA_TriMatching
        
        #mtinfer output size
        if args.use_multiturn_infer or args.use_bilstm:
            choice_infer_hidden_size = 2 * args.hidden_size
            #choice_infer_hidden_size = 2 * args.hidden_size * len(args.matching_order)
        else:
            #choice_infer_hidden_size = args.hidden_size * len(args.matching_order)
            choice_infer_hidden_size = 2*args.hidden_size 


        #self.c_infer_self_attn = layers.LinearSeqAttn(choice_infer_hidden_size)
        self.q_self_attn = layers.LinearSeqAttn(2*args.hidden_size)
        '''my:'''
        self.linearlayer = nn.Linear(rnn_input_size,args.hidden_size)   ##my
        self.pre_y = nn.Linear(2*args.hidden_size+args.pos_emb_dim + args.ner_emb_dim +5+ 2*args.rel_emb_dim, 1)
        if args.use_multiturn_infer == True:
            #self.c_infer_linear= nn.Linear(4*choice_infer_hidden_size,args.hidden_size)
            self.c_infer_linear= nn.Linear(4*choice_infer_hidden_size + 2*2*args.hidden_size,args.hidden_size)
        #elif args.use_bilstm == True:
        else:
            infer_input_size =2*2*args.hidden_size
            if self.args.p_channel==True:
                infer_input_size +=2*choice_infer_hidden_size

            if self.args.q_channel==True:
                infer_input_size +=2*choice_infer_hidden_size

            if self.args.c_channel==True:
                infer_input_size +=2*choice_infer_hidden_size

            self.c_infer_linear= nn.Linear(infer_input_size ,args.hidden_size)
        #self.logits_linear= nn.Linear(args.hidden_size,1)
        #self.logits_linear= nn.Linear(args.hidden_size,2)



    def forward(self, p, p_pos, p_ner, p_mask, q, q_pos, q_ner, q_mask, c,c_pos,c_ner, c_mask,\
               p_f_tensor,q_f_tensor,c_f_tensor, p_q_relation, p_c_relation,q_p_relation,q_c_relation,c_p_relation,c_q_relation,label,is_paint=0):

        p_rnn_input, q_rnn_input, c_rnn_input, q_features = self.add_embeddings(p, p_pos, p_ner, q, q_pos, q_ner, c,c_pos,c_ner,p_f_tensor,q_f_tensor,c_f_tensor,\
                       p_q_relation, p_c_relation,q_p_relation,q_c_relation,c_p_relation,c_q_relation)
        #print('ccccccccccccccccc_rnn_input',q_rnn_input,c_rnn_input.size())
        #print('c_rnn_input',c_rnn_input)
        #Attention Layer 
        #print(label.size())
        #print(q_rnn_input.size()) #torch.Size([49, 50, 345]) batch*length*hidden_size
        
        ql = self.linearlayer(q_rnn_input) #torch.Size([49, 50, 345])
        cl = self.linearlayer(c_rnn_input) #torch.Size([49, 49, 345])
        #print(ql.weight)
        #ql = F.sigmoid(ql)
       
        '''
        ql = self.context_rnn(q_rnn_input, q_mask)
        cl = self.context_rnn(c_rnn_input, c_mask)
        if self.args.dropout_rnn_output > 0:
            ql = nn.functional.dropout(ql, p=self.args.dropout_rnn_output, training=self.training)
            cl = nn.functional.dropout(cl, p=self.args.dropout_rnn_output, training=self.training)
        '''

        #print('ql',ql,ql.size())
        #print('cl',cl)
        #print('----------forward----------------------')
        #print(ql.size(),cl.size()) #torch.Size([49, 50, 345]) torch.Size([49, 49, 345])
        _,mqc = self.hidden_match(ql, cl, c_mask) #torch.Size([49, 50, 345])
        #print(mqc.size()) #torch.Size([49, 50, 345])
        _,mQC = self.hidden_match(mqc, cl, c_mask)  #is this cl is thw same cl above???
        #print(mqc.size(), mQC.size()) #torch.Size([49, 50, 345])
        #print('mqc',mqc)
        #print('mQC',mQC)
        #Sequence modeling layer Hq = BiLstm(q_rnn_input, mQC) 
        Hq = self.Hq_BiLstm(torch.cat([q_rnn_input,mQC],2),q_mask) #batch*length*hidden_size*2
        #print('Hq',Hq)
        #prediction layer       
        pre = self.pre_y(torch.cat([Hq, q_features],2)) #batch*length*1
        #print('pre',pre)
        #print(Hq.size(),pre.size())
        #print('hhhhhhhhhhhhhhhhhhh')
        proba = F.sigmoid(pre.squeeze(-1)) #batch*length 
        #print(proba.size())
        #print('probq,q',proba,q)
        return  proba

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
            iq_pos_emb = nn.functional.dropout(q_pos_emb, p=self.args.dropout_emb, training=self.training)
            c_pos_emb = nn.functional.dropout(c_pos_emb, p=self.args.dropout_emb, training=self.training)

            p_ner_emb = nn.functional.dropout(p_ner_emb, p=self.args.dropout_emb, training=self.training)
            q_ner_emb = nn.functional.dropout(q_ner_emb, p=self.args.dropout_emb, training=self.training)
            c_ner_emb = nn.functional.dropout(c_ner_emb, p=self.args.dropout_emb, training=self.training)
        q_features = torch.cat([q_pos_emb,q_ner_emb,q_f_tensor,q_p_rel_emb,q_c_rel_emb],2)
        p_rnn_input = torch.cat([p_emb,p_pos_emb,p_ner_emb,p_f_tensor,p_q_rel_emb,p_c_rel_emb],2)
        q_rnn_input = torch.cat([q_emb,q_pos_emb,q_ner_emb,q_f_tensor,q_p_rel_emb,q_c_rel_emb],2)
        c_rnn_input = torch.cat([c_emb,c_pos_emb,c_ner_emb,c_f_tensor,c_p_rel_emb,c_q_rel_emb],2)

        return p_rnn_input, q_rnn_input, c_rnn_input, q_features



    def tri_matching(self,x,x_y, agg_function, x_mask):
        '''
        x,x_y,x_z = l*2d
        agg_function: the object of MultiTurnInference
        
        '''
        concat_feature = torch.cat([x, x_y],2)
        sub_feature =  (x - x_y)
        mul_feature = x*x_y
    
        x_mfeature = {"c":concat_feature, "s":sub_feature, "m":mul_feature }
        dim = x.size()
        init_mem = torch.zeros(dim[0],dim[1],dim[2]).float().cuda()  #zero mem
        x_infer_emb, x_mem_list, x_mem_gate_list = agg_function(x_mfeature,x_mask,init_mem=init_mem,x_order=self.args.matching_order)
 
        return x_infer_emb, x_mem_list, x_mem_gate_list
          

    def matched_self(self,x,x_y, x_mask):
    
        x_yz_cat = torch.cat([x, x_y],2)
        x_yz_weight,x_yz_weighted_hiddens = self.hidden_match(x_yz_cat, x_yz_cat, x_mask)

        if self.args.dropout_att_score > 0:
            x_yz_weight = nn.functional.dropout(x_yz_weight, p=self.args.dropout_att_score, training= self.training)

        matched_x = x_yz_weight.bmm(x)

        return matched_x


    def NA_TriMatching(self,p_hiddens, c_hiddens,p_mask, c_mask):
       
        #--------------naive attention
        _,p_c_hiddens = self.hidden_match(p_hiddens,c_hiddens,c_mask)
        _,c_p_hiddens = self.hidden_match(c_hiddens,p_hiddens,p_mask)

        #------p_c_q--------------
        if self.args.p_channel==True:
            self.p_infer_emb,p_mems,p_mem_gates = self.tri_matching(x = p_hiddens,
                                             x_y = p_c_hiddens,
                                             agg_function=self.mtinfer,
                                             x_mask = p_mask)

        
        #------c_p_q--------------
        if self.args.c_channel==True:
            self.c_infer_emb,c_mems, c_mem_gates = self.tri_matching(x = c_hiddens,
                                              x_y = c_p_hiddens,
                                              agg_function=self.mtinfer,
                                              x_mask = c_mask)

        #------matched_self--------------
        self.matched_p = self.matched_self(x = p_hiddens, x_y = p_c_hiddens, x_mask = p_mask)
        self.matched_c = self.matched_self(x = c_hiddens, x_y = c_p_hiddens, x_mask = c_mask)


        
    def CA_TriMatching(self,p_hiddens, q_hiddens, c_hiddens,p_mask, q_mask, c_mask):

        #--------------naive attention
        _,p_q_hiddens = self.hidden_match(p_hiddens,q_hiddens,q_mask)
        _,p_c_hiddens = self.hidden_match(p_hiddens,c_hiddens,c_mask)

        _,q_p_hiddens = self.hidden_match(q_hiddens,p_hiddens,p_mask)
        _,q_c_hiddens = self.hidden_match(q_hiddens,c_hiddens,c_mask)

        _,c_p_hiddens = self.hidden_match(c_hiddens,p_hiddens,p_mask)
        _,c_q_hiddens = self.hidden_match(c_hiddens,q_hiddens,q_mask)

        #--------------compound attention
        _,p_c_q_hiddens = self.hidden_match(p_hiddens, c_q_hiddens, c_mask)
        _,p_q_c_hiddens = self.hidden_match(p_hiddens, q_c_hiddens, q_mask)

        _,q_p_c_hiddens = self.hidden_match(q_hiddens, p_c_hiddens, p_mask)
        _,q_c_p_hiddens = self.hidden_match(q_hiddens, c_p_hiddens, c_mask)

        _,c_q_p_hiddens = self.hidden_match(c_hiddens, q_p_hiddens, q_mask)
        _,c_p_q_hiddens = self.hidden_match(c_hiddens, p_q_hiddens, p_mask)

        #------p_c_q--------------
        if self.args.p_channel==True:
            self.p_infer_emb,p_mems,p_mem_gates = self.tri_matching(x = p_hiddens,
                                             x_y = p_c_q_hiddens,
                                             x_z = p_q_c_hiddens,
                                             agg_function=self.mtinfer,
                                             x_mask = p_mask)

        #------q_p_c--------------
        if self.args.q_channel==True:
            self.q_infer_emb,q_mems, q_mem_gates = self.tri_matching(x = q_hiddens,
                                              x_y = q_p_c_hiddens,
                                              x_z = q_c_p_hiddens,
                                              agg_function=self.mtinfer,
                                              x_mask = q_mask)
        
        #------c_p_q--------------
        if self.args.c_channel==True:
            self.c_infer_emb,c_mems, c_mem_gates = self.tri_matching(x = c_hiddens,
                                              x_y = c_p_q_hiddens,
                                              x_z = c_q_p_hiddens,
                                              agg_function=self.mtinfer,
                                              x_mask = c_mask)

        self.matched_p = self.matched_self(x = p_hiddens, x_y = p_c_q_hiddens, x_z = p_q_c_hiddens, x_mask = p_mask)
        self.matched_q = self.matched_self(x = q_hiddens, x_y = q_p_c_hiddens, x_z = q_c_p_hiddens, x_mask = q_mask)
        self.matched_c = self.matched_self(x = c_hiddens, x_y = c_q_p_hiddens, x_z = c_p_q_hiddens, x_mask = c_mask)

    def NA_CA_TriMatching(self, p_hiddens, q_hiddens, c_hiddens,p_mask, q_mask, c_mask):

        #--------------naive attention
        _,p_q_hiddens = self.hidden_match(p_hiddens,q_hiddens,q_mask)
        _,p_c_hiddens = self.hidden_match(p_hiddens,c_hiddens,c_mask)

        _,q_p_hiddens = self.hidden_match(q_hiddens,p_hiddens,p_mask)
        _,q_c_hiddens = self.hidden_match(q_hiddens,c_hiddens,c_mask)

        _,c_p_hiddens = self.hidden_match(c_hiddens,p_hiddens,p_mask)
        _,c_q_hiddens = self.hidden_match(c_hiddens,q_hiddens,q_mask)

        #--------------compound attention
        _,p_c_q_hiddens = self.hidden_match(p_hiddens, c_q_hiddens, c_mask)
        _,p_q_c_hiddens = self.hidden_match(p_hiddens, q_c_hiddens, q_mask)

        _,q_p_c_hiddens = self.hidden_match(q_hiddens, p_c_hiddens, p_mask)
        _,q_c_p_hiddens = self.hidden_match(q_hiddens, c_p_hiddens, c_mask)

        _,c_q_p_hiddens = self.hidden_match(c_hiddens, q_p_hiddens, q_mask)
        _,c_p_q_hiddens = self.hidden_match(c_hiddens, p_q_hiddens, p_mask)

        if self.args.tri_input == 'NA_CA' :   
            #------p_c_q--------------
            if self.args.p_channel==True:
                self.p_infer_emb,p_mems,p_mem_gates = self.tri_matching(x = p_hiddens,
                                                 x_y = p_q_hiddens,
                                                 x_z = p_q_c_hiddens,
                                                 agg_function=self.mtinfer,
                                                 x_mask = p_mask)

            #------q_p_c--------------
            if self.args.q_channel==True:
                self.q_infer_emb,q_mems, q_mem_gates = self.tri_matching(x = q_hiddens,
                                                  x_y = q_c_hiddens,
                                                  x_z = q_c_p_hiddens,
                                                  agg_function=self.mtinfer,
                                                  x_mask = q_mask)
            
            #------c_p_q--------------
            if self.args.c_channel==True:
                self.c_infer_emb,c_mems, c_mem_gates = self.tri_matching(x = c_hiddens,
                                                  x_y = c_q_hiddens,
                                                  x_z = c_q_p_hiddens,
                                                  agg_function=self.mtinfer,
                                                  x_mask = c_mask)

            self.matched_p = self.matched_self(x = p_hiddens, x_y = p_q_hiddens, x_z = p_q_c_hiddens, x_mask = p_mask)
            self.matched_q = self.matched_self(x = q_hiddens, x_y = q_c_hiddens, x_z = q_c_p_hiddens, x_mask = q_mask)
            self.matched_c = self.matched_self(x = c_hiddens, x_y = c_q_hiddens, x_z = c_q_p_hiddens, x_mask = c_mask)

        elif self.args.tri_input == 'CA_NA' :   
            #------p_c_q--------------
            if self.args.p_channel==True:
                self.p_infer_emb,p_mems,p_mem_gates = self.tri_matching(x = p_hiddens,
                                                 x_y = p_q_c_hiddens,
                                                 x_z = p_q_hiddens,
                                                 agg_function=self.mtinfer,
                                                 x_mask = p_mask)

            #------q_p_c--------------
            if self.args.q_channel==True:
                self.q_infer_emb,q_mems, q_mem_gates = self.tri_matching(x = q_hiddens,
                                                  x_y = q_p_c_hiddens,
                                                  x_z = q_p_hiddens,
                                                  agg_function=self.mtinfer,
                                                  x_mask = q_mask)
            
            #------c_p_q--------------
            if self.args.c_channel==True:
                self.c_infer_emb,c_mems, c_mem_gates = self.tri_matching(x = c_hiddens,
                                                  x_y = c_p_q_hiddens,
                                                  x_z = c_p_hiddens,
                                                  agg_function=self.mtinfer,
                                                  x_mask = c_mask)

            self.matched_p = self.matched_self(x = p_hiddens, x_y = p_q_c_hiddens, x_z = p_q_hiddens, x_mask = p_mask)
            self.matched_q = self.matched_self(x = q_hiddens, x_y = q_p_c_hiddens, x_z = q_p_hiddens, x_mask = q_mask)
            self.matched_c = self.matched_self(x = c_hiddens, x_y = c_q_p_hiddens, x_z = c_p_hiddens, x_mask = c_mask)
