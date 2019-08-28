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
from model import Model
#import logger

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

'''
def split_data(inp_list, k, m):
    #inp_list: a list
    #k:k-held out validation 
    #m: the m-th part for validation
    #n = int(math.ceil(len(inp_list) / float(k)))
    n = int(math.floor(len(inp_list) / float(k)))
    chunked= [inp_list[i:i + n] for i in range(0, len(inp_list), n)]

    dev = chunked[m] 
   
    print ("###len(dev)",len(dev))
    train_list =  chunked[0:m] + chunked[m+1:] 
    train = []
    for ele in train_list:
        train.extend(ele)
    #for x in train:
    #    print ( x.id)
    return train,dev 
'''
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


def run_epoch(args,model,train_data,dev_data,test_data,checkpoint_path):
    
    best_dev_acc = 0.0
    best_test_acc = 0.0
    best_dev_epoch = 0
    dev_acc=0.0
    for i in range(args.epoch):
        print('Epoch %d...' % i)
        #if i < 0:
        #    dev_acc,dev_prelabel,dev_problist,dev_truelabel = model.evaluate(dev_data)
        #    print('Dev accuracy: %f' % dev_acc)
        start_time = time.time()
        np.random.shuffle(train_data)
        cur_train_data = train_data

        model.train(cur_train_data)
        train_acc = model.evaluate(train_data[:2000], debug=False, eval_train=True)
        print('Train accuracy: %f' % train_acc)
        dev_acc,dev_prelabel,dev_problist,dev_truelabel = model.evaluate(dev_data, debug=True)
        print('Dev accuracy: %f' % dev_acc)
        test_acc,test_prelabel,test_problist,test_truelabel = model.evaluate(test_data, debug=True)
        print('Test accuracy: %f' % test_acc)

        if dev_acc > best_dev_acc :
            best_dev_acc = dev_acc
            best_dev_epoch = i
            best_test_acc = test_acc
            os.system('mv ./data/output.log ./data/best-dev'+args.train_name+'.log')
            writer = open('./checkpoint/dev_dir/dev_prelabel_'+args.train_name, 'w', encoding='utf-8')
            for p in dev_prelabel:
                writer.write('%d\n' % p)
            writer.close() 
            writer = open('./checkpoint/dev_dir/dev_problist_'+args.train_name, 'w', encoding='utf-8')
            for p in dev_problist:
                writer.write('%f\n' % p)
            writer.close()        
            writer2 = open('./checkpoint/dev_dir/dev_truelabel_'+args.train_name, 'w', encoding='utf-8')
            for p in dev_truelabel:
                writer2.write('%d\n' % p)
            writer2.close()
            model.save(checkpoint_path)
        elif args.test_mode:
            model.save(checkpoint_path)
        print('Epoch %d use %d seconds.' % (i, time.time() - start_time))

    print ("\n","-"*40)
    best_info = 'Epoch: %d best_dev_acc: %4f  best_test_acc: %4f' % (best_dev_epoch, best_dev_acc,best_test_acc)
    print (best_info)
    print ("-"*40)

    return best_info,best_dev_acc, best_test_acc


if __name__ == '__main__':

    build_vocab()
    if args.debug: 
        train_data = load_data('./data/tqa_v1_val.json-processed_multi.json')
    else:
       train_data = load_data('./data/tqa_v1_train.json-processed_multi.json')
    #train_data= train_data[:2000]
    dev_data = load_data('./data/tqa_v1_val.json-processed_multi.json')
    test_data = load_data('./data/tqa_v2_test.json-processed_multi.json')

    #build_vocab(train_data+dev_data)
    # total 3742
    os.makedirs('./checkpoint', exist_ok=True)
    os.makedirs('./checkpoint/test_dir', exist_ok=True)
    os.makedirs('./checkpoint/dev_dir', exist_ok=True)
    #sys.exit()
    #all_best_info={}
    model = Model(args)
    print ("model",model)
    checkpoint_path = './checkpoint/%s-%s.mdl' % (args.model_name,args.checkpoint_name)
    print('Trained model will be saved to %s' % checkpoint_path)

    b_info,b_dev_acc, b_test_acc = run_epoch(args,model,train_data,dev_data,test_data,checkpoint_path)
    
    #all_best_info[j]= b_info

    #print ("-"*40)
    #print ("all_best_info",json.dumps(all_best_info,indent=4))
    #print ("final#%d ave_dev_acc:%4f ave_test_acc:%4f"%(j,sum_dev_acc/(j+1),sum_test_acc/(j+1)))
    
