import logging
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from sklearn import metrics
from utils import vocab
from doc import batchify
import importlib
#from trian import TriAN

logger = logging.getLogger()

class Model:

    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        #self.batch_size = 1
        self.finetune_topk = args.finetune_topk
        self.lr = args.lr
        self.use_cuda = (args.use_cuda == True) and torch.cuda.is_available()
        print('Use cuda:', self.use_cuda)
        if self.use_cuda:
            torch.cuda.set_device(int(args.gpu))
        
        module = importlib.import_module(".".join([args.model_name]))
        MyModel = getattr(module, 'MyModel')
        self.network = MyModel(args)
        self.init_optimizer()
        if args.pretrained:
            print('Load pretrained model from %s...' % args.pretrained)
            self.load(args.pretrained)
        else:
            self.load_embeddings(vocab.tokens(), args.embedding_file)
        self.network.register_buffer('fixed_embedding', self.network.embedding.weight.data[self.finetune_topk:].clone())
        if self.use_cuda:
            self.network.cuda()
        print(self.network)
        self._report_num_trainable_parameters()

    def _report_num_trainable_parameters(self):
        num_parameters = 0
        for p in self.network.parameters():
            if p.requires_grad:
                sz = list(p.size())
                if sz[0] == len(vocab):
                    sz[0] = self.finetune_topk
                num_parameters += np.prod(sz)
        print('Number of parameters: ', num_parameters)

    def train(self, train_data):
        self.network.train()
        self.updates = 0
        n = 0
        iter_cnt, num_iter = 0, (len(train_data) + self.batch_size - 1) // self.batch_size
        for batch_input in self._iter_data(train_data):
            #print('-------------n',n)
            n =n + 1
            feed_input = [x for x in batch_input[:-1]]
            y = batch_input[-1]
            lengths = batch_input[-2]
            #print(y)
            feed_input.append(0)
            #ql,cl,mqc,mQC = self.network(*feed_input)
            #print(ql.size(),cl.size(),mqc.size(),mQC.size())
            pred_proba = self.network(*feed_input)
            #print('--------------------------------------------')
            #print(pred_proba.size())
            #print(y.size())
            #print ("y",y)
            #print ("pred_proba",pred_proba)
            #print(pred_proba.size(),lengths)
            #pre = []
            #gold = []
            #for i in range(len(lengths)):
            #    for j in range(lengths[i]):
            #        pre.append(float(pred_proba[i][j]))
            #        gold.append(float(y[i][j]))
            #print(pre,gold)
            #loss = torch.nn.BCELoss(pred_proba, y)
            loss = F.binary_cross_entropy(pred_proba, y)
            #loss = F.binary_cross_entropy(torch.FloatTensor(pre), torch.FloatTensor(gold))
            self.optimizer.zero_grad()
            loss.backward()
            '''
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(pred_proba, y.type(torch.LongTensor).cuda())
            self.optimizer.zero_grad()
            loss.backward()
            '''
            torch.nn.utils.clip_grad_norm(self.network.parameters(), self.args.grad_clipping)

            # Update parameters
            self.optimizer.step()
            self.network.embedding.weight.data[self.finetune_topk:] = self.network.fixed_embedding
            self.updates += 1
            iter_cnt += 1

            if self.updates % 200 == 0:
                print('Iter: %d/%d, Loss: %f' % (iter_cnt, num_iter, loss.item()))
        self.scheduler.step()
        print('LR:', self.scheduler.get_lr()[0])

    def evaluate(self, dev_data, debug=False, eval_train=False,is_paint=False):
        if len(dev_data) == 0:
            return -1.0
        self.network.eval()
        correct, total, prediction, gold, problist= 0, 0, [], [], []
        dev_data = sorted(dev_data, key=lambda ex: ex.id)
        g=0
        y,lengths = [],[]
        for batch_input in self._iter_data(dev_data):
            feed_input = [x for x in batch_input[:-1]]
            feed_input.append(1)
            g+=1
            if(g<=4 and is_paint):
                self.network(*feed_input)
                print("come in painting")
            feed_input[-1]=0
            y = batch_input[-1].data.cpu().numpy()
            #y += list(batch_input[-1].data.cpu().numpy())
            lengths += list(batch_input[-2].data.cpu().numpy())
            #print(y,lengths)
            pred_proba = self.network(*feed_input)
            pred_proba = pred_proba.data.cpu()
            prediction += list(pred_proba)
            #print(pred_proba,prediction)
            gold +=[ [int(label) for label in labels ] for labels in y]
            assert(len(prediction) == len(gold))
        pppp = prediction
        
        for i in range(len(lengths)):
            for j in range(lengths[i]):
                pppp[i][j] = 1 if prediction[i][j] > 0.5 else 0

        sums = 0
        num = 0
        truth=[]
        virt = []
        correct = 0
        print(len(lengths))
        for i in range(len(lengths)):
            num = num + lengths[i]
            #print(lengths[i])
            #print(gold[i])
            #print(pppp[i])
            for j in range(lengths[i]):
                prediction[i][j] = 1 if prediction[i][j] > 0.5 else 0
                #sums = sums + (1 if gold[i][j] == prediction[i][j] else 0)
                truth.append(gold[i][j])
                virt.append(int(prediction[i][j].numpy()))
                if gold[i][j] == int(prediction[i][j].numpy()):
                    correct+=1
        #print('truth',truth)
        #print('virt',virt)
        #print(truth)
        print(num, correct,correct/num)
        ss = 0
        for i in range(len(lengths)):
            s = 1
            for j in range(lengths[i]):
                prediction[i][j] = 1 if prediction[i][j] > 0.5 else 0
                if not gold[i][j] == int(prediction[i][j].numpy()):
                    s = 0
            if s == 1:
                 ss +=1
        print(ss/len(lengths))
        if eval_train:
            #prediction = [1 if p > 0.5 else 0 for p in prediction]
            #acc = sum([1 if y1 == y2 else 0 for y1, y2 in zip(prediction, gold)]) / len(gold)
            #print('gold',gold)
            #print('prediction',prediction)
            return metrics.accuracy_score(truth, virt), metrics.precision_score(truth, virt), metrics.recall_score(truth, virt), metrics.f1_score(truth, virt)
        
        if debug:
            writer = open('./data/output.log', 'w', encoding='utf-8')
        #prediction = [1 if p > 0.5 else 0 for p in prediction]
        for i, ex in enumerate(dev_data):
            if debug:
                writer.write('Passage: %s\n' % dev_data[i - 1].passage)
                writer.write('Question: %s\n' % dev_data[i - 1].question)
                #writer.write('*' if prediction[i] == gold[i] else ' ')
                writer.write(str(prediction[i]))
                writer.write(str(gold[i]))
                writer.write('\n')
            #print(prediction[i], gold[i])
            #if prediction[i] == gold[i]:
                #correct += 1
                #print(prediction[i], gold[i])
            #total += 1

        #acc = 1.0 * correct / total
        if debug:
            writer.write('Accuracy: %f\n' %  metrics.accuracy_score(truth, virt))
            writer.close()
        return metrics.accuracy_score(truth, virt), prediction, problist, gold, metrics.precision_score(truth, virt), metrics.recall_score(truth, virt), metrics.f1_score(truth, virt)

    def evaluate1(self, dev_data, debug=False, eval_train=False,is_paint=False):
        if len(dev_data) == 0:
            return -1.0
        self.network.eval()
        correct, total, prediction, gold, problist= 0, 0, [], [], []
        dev_data = sorted(dev_data, key=lambda ex: ex.id)
        g=0
        for batch_input in self._iter_data(dev_data):
            feed_input = [x for x in batch_input[:-1]]
            feed_input.append(1)
            g+=1
            if(g<=4 and is_paint):
                self.network(*feed_input)
                print("come in painting")
            feed_input[-1]=0
            y = batch_input[-1].data.cpu().numpy()
            pred_proba = self.network(*feed_input)
            pred_proba = pred_proba.data.cpu()
            prediction += list(pred_proba)
            gold += [int(label) for label in y]
            assert(len(prediction) == len(gold))
        problist = prediction 
        if eval_train:
            prediction = [1 if p > 0.5 else 0 for p in prediction]
            acc = sum([1 if y1 == y2 else 0 for y1, y2 in zip(prediction, gold)]) / len(gold)
            return acc

        cur_pred, cur_gold, cur_choices = [], [], []
        if debug:
            writer = open('./data/output.log', 'w', encoding='utf-8')
        for i, ex in enumerate(dev_data):
            if i + 1 == len(dev_data):
                cur_pred.append(prediction[i])
                cur_gold.append(gold[i])
                cur_choices.append(ex.choice)
            if (i > 0 and ex.id[:-1] != dev_data[i - 1].id[:-1]) or (i + 1 == len(dev_data)):
                py, gy = np.argmax(cur_pred), np.argmax(cur_gold)
                if debug:
                    writer.write('Passage: %s\n' % dev_data[i - 1].passage)
                    writer.write('Question: %s\n' % dev_data[i - 1].question)
                    for idx, choice in enumerate(cur_choices):
                        writer.write('*' if idx == gy else ' ')
                        writer.write('%s  %f\n' % (choice, cur_pred[idx]))
                    writer.write('\n')
                if py == gy:
                    correct += 1
                total += 1
                cur_pred, cur_gold, cur_choices = [], [], []
            cur_pred.append(prediction[i])
            cur_gold.append(gold[i])
            cur_choices.append(ex.choice)

        acc = 1.0 * correct / total
        if debug:
            writer.write('Accuracy: %f\n' % acc)
            writer.close()
        return acc, prediction, problist, gold

    def predict(self, test_data):
        # DO NOT SHUFFLE test_data
        self.network.eval()
        prediction = []
        for batch_input in self._iter_data(test_data):
            feed_input = [x for x in batch_input[:-1]]
     
            pred_proba = self.network(*feed_input)
            pred_proba = pred_proba.data.cpu()
            prediction += list(pred_proba)
        return prediction

    def _iter_data(self, data):
        num_iter = (len(data) + self.batch_size - 1) // self.batch_size
        for i in range(num_iter):
            start_idx = i * self.batch_size
            batch_data = data[start_idx:(start_idx + self.batch_size)]
            batch_input = batchify(batch_data)
            #print ("batch_input",batch_input)
            # Transfer to GPU
            if self.use_cuda:
                batch_input = [Variable(x.cuda(async=True)) for x in batch_input]
            else:
                batch_input = [Variable(x) for x in batch_input]
            yield batch_input

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in vocab}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = vocab.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[vocab[w]].copy_(vec)
                    else:
                        logging.warning('WARN: Duplicate embedding found for %s' % w)
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[vocab[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[vocab[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def init_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.lr,
                                       momentum=0.4,
                                       weight_decay=0)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                        lr=self.lr,
                                        #weight_decay=0)
                                        weight_decay=self.args.l2_strength)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)
        #self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 12, 16,18,25], gamma=0.5) #0817

        #self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 12, 16], gamma=0.5) #0816
        #self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[6, 12, 20,25], gamma=0.5) #
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 15,20], gamma=0.5) #0815
        #self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[9, 13], gamma=0.4)

    def save(self, ckt_path):
        state_dict = copy.copy(self.network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {'state_dict': state_dict}
        torch.save(params, ckt_path)

    def load(self, ckt_path):
        logger.info('Loading model %s' % ckt_path)
        saved_params = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        state_dict = saved_params['state_dict']
        return self.network.load_state_dict(state_dict)

    def cuda(self):
        self.use_cuda = True
        self.network.cuda()