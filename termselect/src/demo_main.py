import os
import sys
import time
import torch
import random
import numpy as np
import math
import json
from datetime import datetime

from utils import load_data, build_vocab
from config import args
from demo_model import Model
#import logger

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

def split_data(inp_list, k):
    '''
    inp_list: a list
    k:k-held out validation 
    m: the m-th part for validation
    '''
    #n = int(math.ceil(len(inp_list) / float(k)))
    n = int(math.floor(len(inp_list) / float(k)))
    chunked= [inp_list[i:i + n] for i in range(0, len(inp_list), n)]

    return chunked 


def run(args,model,test_data,datapath):
    
    start_time = time.time()
    #np.random.shuffle(train_data)
    model.predict(test_data, demo=True, outputpath = datapath)
    #test_acc,test_prelabel,test_problist,test_truelabel,test_precision,test_recall,test_f1_score = model.evaluate(test_data, debug=True)



if __name__ == '__main__':

    build_vocab()
    '''
    datapath1 = '/home/hd_1T/haiou/ARC/arc-solvers/data/ARC-V1-Feb2018/ARC-Challenge/init/ARC-Challenge-Dev.jsonl-processed.jsonl'
    test_data1 = load_data(datapath1)

    datapath2 = '/home/hd_1T/haiou/ARC/arc-solvers/data/ARC-V1-Feb2018/ARC-Challenge/init/ARC-Challenge-Train.jsonl-processed.jsonl'
    test_data2 = load_data(datapath2) 

    datapath3 = '/home/hd_1T/haiou/ARC/arc-solvers/data/ARC-V1-Feb2018/ARC-Challenge/init/ARC-Challenge-Test.jsonl-processed.jsonl'
    test_data3 = load_data(datapath3)   
    '''
    datapath1 = '/home/hd_1T/haiou/ARC/arc-solvers/data/ARC-V1-Feb2018/ARC-Easy/init/ARC-Easy-Dev.jsonl-processed.jsonl'
    test_data1 = load_data(datapath1)

    datapath2 = '/home/hd_1T/haiou/ARC/arc-solvers/data/ARC-V1-Feb2018/ARC-Easy/init/ARC-Easy-Train.jsonl-processed.jsonl'
    test_data2 = load_data(datapath2) 

    datapath3 = '/home/hd_1T/haiou/ARC/arc-solvers/data/ARC-V1-Feb2018/ARC-Easy/init/ARC-Easy-Test.jsonl-processed.jsonl'
    test_data3 = load_data(datapath3)
 
    model = Model(args)
    print ("model",model)
    run(args,model, test_data1, datapath1 + '-termselect')
    run(args,model, test_data2, datapath2 + '-termselect')
    run(args,model, test_data3, datapath3 + '-termselect')

