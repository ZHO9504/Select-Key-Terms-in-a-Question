import torch
import numpy as np
from config import args
from utils import vocab, pos_vocab, ner_vocab, rel_vocab
from torch.autograd import Variable
class Example:

    def __init__(self, input_dict):
        self.id = input_dict['id']
        self.passage = input_dict['p_words']
        self.question = input_dict['q_words']
        self.choice = input_dict['c_words']
        self.d_pos = input_dict['p_pos']
        self.q_pos = input_dict['q_pos']
        self.c_pos = input_dict['c_pos']
        self.d_ner = input_dict['p_ner']
        self.q_ner = input_dict['q_ner']
        self.c_ner = input_dict['c_ner']
        self.p_q_rel = input_dict['p_q_relation']
        self.p_c_rel = input_dict['p_c_relation']
        self.q_p_rel = input_dict['q_p_relation']
        self.q_c_rel = input_dict['q_c_relation']
        self.c_p_rel = input_dict['c_p_relation']
        self.c_q_rel = input_dict['c_q_relation']
        assert len(self.q_pos) == len(self.question.split()), (self.q_pos, self.question)
        #assert len(self.d_pos) == len(self.passage.split())
        self.p_features = np.stack([input_dict['p_in_q'], input_dict['p_in_c'], \
                                    input_dict['p_lemma_in_q'], input_dict['p_lemma_in_c'], \
                                    input_dict['p_tf']], 1)

        self.q_features = np.stack([input_dict['q_in_p'], input_dict['q_in_c'], \
                                    input_dict['q_lemma_in_p'], input_dict['q_lemma_in_c'], \
                                    input_dict['q_tf']], 1)

        self.c_features = np.stack([input_dict['c_in_p'], input_dict['c_in_q'], \
                                    input_dict['c_lemma_in_p'], input_dict['c_lemma_in_q'], \
                                    input_dict['c_tf']], 1)
        self.label = input_dict['label']
        
        self.d_tensor = torch.LongTensor([vocab[w] for w in self.passage.split()])
        self.q_tensor = torch.LongTensor([vocab[w] for w in self.question.split()])
        self.c_tensor = torch.LongTensor([vocab[w] for w in self.choice.split()])
        self.d_pos_tensor = torch.LongTensor([pos_vocab[w] for w in self.d_pos])
        self.q_pos_tensor = torch.LongTensor([pos_vocab[w] for w in self.q_pos])
        self.c_pos_tensor = torch.LongTensor([pos_vocab[w] for w in self.c_pos])
        self.d_ner_tensor = torch.LongTensor([ner_vocab[w] for w in self.d_ner])
        self.q_ner_tensor = torch.LongTensor([ner_vocab[w] for w in self.q_ner])
        self.c_ner_tensor = torch.LongTensor([ner_vocab[w] for w in self.c_ner])

        self.p_features = torch.from_numpy(self.p_features).type(torch.FloatTensor)
        self.q_features = torch.from_numpy(self.q_features).type(torch.FloatTensor)
        self.c_features = torch.from_numpy(self.c_features).type(torch.FloatTensor)

        self.p_q_relation = torch.LongTensor([rel_vocab[r] for r in input_dict['p_q_relation']])
        self.p_c_relation = torch.LongTensor([rel_vocab[r] for r in input_dict['p_c_relation']])
        self.q_p_relation = torch.LongTensor([rel_vocab[r] for r in input_dict['q_p_relation']])
        self.q_c_relation = torch.LongTensor([rel_vocab[r] for r in input_dict['q_c_relation']])
        self.c_p_relation = torch.LongTensor([rel_vocab[r] for r in input_dict['c_p_relation']])
        self.c_q_relation = torch.LongTensor([rel_vocab[r] for r in input_dict['c_q_relation']])

    def __str__(self):
        return 'Passage: %s\n Question: %s\n Answer: %s, Label: %d' % (self.passage, self.question, self.choice, self.label)
'''
def _to_indices_and_mask(batch_tensor, need_mask=True,mx_len = None):
    if mx_len ==None: 
        mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(0)
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(1)
    for i, t in enumerate(batch_tensor):
        if len(t)>mx_len:
            true_len=mx_len
        else:
            true_len=len(t)
        indices[i, :true_len].copy_(t[:true_len])
    
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(0)
    
    if need_mask:
        #print ("indices",indices)
        #print ("mask",mask)
        #print ("#"*30)
        return indices, mask
    else:
        return indices
'''
def _to_indices_and_mask(batch_tensor, need_mask=True,mx_len=None):
    if mx_len ==None: 
        mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(0)
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(1)
  
        position =np.array([np.arange(1, mx_len+1) for x in batch_tensor])
        position_tensor = Variable(torch.LongTensor(position))

    for i, t in enumerate(batch_tensor):
        if len(t)>mx_len:
            true_len=mx_len
        else:
            true_len=len(t)
        indices[i, :true_len].copy_(t[:true_len])
        if need_mask:
            #print ("position_tensor",position_tensor)
            mask[i, :true_len].fill_(0)
            position_tensor[i,true_len:].fill_(0)
    if need_mask:
        return indices, mask,position_tensor
    else:
        return indices
'''
def _to_feature_tensor(features):
    mx_len = max([f.size(0) for f in features])
    #print ("mx_len",mx_len)
    batch_size = len(features)
    f_dim = features[0].size(1)
    f_tensor = torch.FloatTensor(batch_size, mx_len, f_dim).fill_(0)
    for i, f in enumerate(features):
        f_tensor[i, :len(f), :].copy_(f)
    return f_tensor
'''
def _to_feature_tensor(features,mx_len=None):
    if mx_len ==None:
        mx_len = max([f.size(0) for f in features])
    #print ("mx_len",mx_len)
    batch_size = len(features)
    f_dim = features[0].size(1)
    f_tensor = torch.FloatTensor(batch_size, mx_len, f_dim).fill_(0)
    for i, f in enumerate(features):
        if len(f)>mx_len:
            true_len=mx_len
        else:
            true_len=len(f)
        f_tensor[i, :true_len, :].copy_(f[:true_len])
    return f_tensor

def batchify(batch_data):
    passage_mx_len = args.max_length
    #makelen = max([t.size(0) for t in batch_tensor])
    p, p_mask,p_position = _to_indices_and_mask([ex.d_tensor for ex in batch_data],mx_len = passage_mx_len)
    p_pos = _to_indices_and_mask([ex.d_pos_tensor for ex in batch_data], need_mask=False,mx_len = passage_mx_len)
    p_ner = _to_indices_and_mask([ex.d_ner_tensor for ex in batch_data], need_mask=False,mx_len = passage_mx_len)
    p_q_relation = _to_indices_and_mask([ex.p_q_relation for ex in batch_data], need_mask=False,mx_len = passage_mx_len)
    p_c_relation = _to_indices_and_mask([ex.p_c_relation for ex in batch_data], need_mask=False,mx_len = passage_mx_len)

    q, q_mask,q_position = _to_indices_and_mask([ex.q_tensor for ex in batch_data])
    q_pos = _to_indices_and_mask([ex.q_pos_tensor for ex in batch_data], need_mask=False)
    q_ner = _to_indices_and_mask([ex.q_ner_tensor for ex in batch_data], need_mask=False)
    q_p_relation = _to_indices_and_mask([ex.q_p_relation for ex in batch_data], need_mask=False)
    q_c_relation = _to_indices_and_mask([ex.q_c_relation for ex in batch_data], need_mask=False)
    #llabel = _to_indices_and_mask([ex.label for ex in batch_data])
    choices = [ex.choice.split() for ex in batch_data]
    c, c_mask,c_position = _to_indices_and_mask([ex.c_tensor for ex in batch_data])
    c_pos = _to_indices_and_mask([ex.c_pos_tensor for ex in batch_data], need_mask=False)
    c_ner = _to_indices_and_mask([ex.c_ner_tensor for ex in batch_data], need_mask=False)
    c_p_relation = _to_indices_and_mask([ex.c_p_relation for ex in batch_data], need_mask=False)
    c_q_relation = _to_indices_and_mask([ex.c_q_relation for ex in batch_data], need_mask=False)

    p_f_tensor = _to_feature_tensor([ex.p_features for ex in batch_data],mx_len = passage_mx_len)
    q_f_tensor = _to_feature_tensor([ex.q_features for ex in batch_data])
    c_f_tensor = _to_feature_tensor([ex.c_features for ex in batch_data])
    #for ex in batch_data:
    #   if ex.label != '1' and ex.label != '0' and ex.label != '2': 
    #       print(ex.label)
    #       ex.label = '0'
    #y = torch.FloatTensor([float(ex.label) for ex in batch_data])
    
    yy = []
    for ex in batch_data:
         #print(ex.label)
        makelen = max([ex.q_tensor.size(0) for ex in batch_data])
        yyy = [0]*makelen
        n = 0
        #print(ex.q_tensor)
        #print(ex.label)
        #print('2222222222')
        #print(len(ex.label.split()))
        for l in ex.label.split():
            yyy[n] = float(l)
            n = n + 1
        #print(yyy)
        yy.append(yyy)
    y1 = torch.FloatTensor([[float(l) for l in yyy] for yyy in yy])
    #y2 = torch.FloatTensor(yy)
    #print('djfhkswhgfwww--------------------------------------------------------')
    
    length = torch.IntTensor([int(len(ex.q_tensor))  for ex in batch_data])
    #print(length)
    #y3 = torch.FloatTensor([float(l) for l in ex ]) for ex in yyy
    #print(y1.size(),q.size(),q_mask.size())
    #import re
    #y = torch.FloatTensor(yy)
    #y = torch.FloatTensor([[float(l) for l in ex.label.split()] for ex in batch_data])
    return p, p_pos, p_ner, p_mask, q, q_pos,q_ner, q_mask, c, c_pos, c_ner,c_mask, p_f_tensor, q_f_tensor, c_f_tensor,p_q_relation, p_c_relation,q_p_relation, q_c_relation,c_p_relation, c_q_relation, length, y1
