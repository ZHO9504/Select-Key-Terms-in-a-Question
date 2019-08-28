"""Definitions of model layers/NN modules"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sns_visual import paint
class StackedBRNN(nn.Module):
    """Stacked Bi-directional RNNs.

    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
        """Encode either padded or non-padded sequences.

        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.

        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        if x_mask.data.sum() == 0:
            # No padding necessary.
            output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            # Pad if we care or if its during eval.
            output = self._forward_padded(x, x_mask)
        else:
            # We don't care.
            output = self._forward_unpadded(x, x_mask)

        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise, encoding that handles
        padding.
        """
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.

    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size()) #flatten x 
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)))
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        return alpha,matched_seq

class SeqDotAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    score = XY

    """

    def __init__(self):
        super(SeqDotAttnMatch, self).__init__()
        #if not identity:
        #    self.linear = nn.Linear(input_size, input_size)
        #else:
        #    self.linear = None

    def forward(self, x, y, y_mask,is_paint=0):

        # Compute scores
        scores = x.bmm(y.transpose(2, 1))

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)))
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))
        #if is_paint==1:
        #    paint('dot-alpha',alpha)
        # Take weighted average
        matched_seq = alpha.bmm(y)
        return alpha, matched_seq

class BilinearVecSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:

    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearVecSeqAttn, self).__init__()
        self.normalize = normalize

        # If identity is true, we just use a dot product without transformation.
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.normalize:
            alpha = F.softmax(xWy)
        else:
            alpha = xWy.exp()
        alpha = alpha.unsqueeze(2) 
        #print ("alpha",alpha.size())
        #print ("x",x.size())
        weighted_x = alpha*x
        #weighted_x  = alpha.bmm(x)
        #print ("weighted_x",weighted_x.size())
        return alpha, weighted_x


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    * o_i * x
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask,is_paint=0):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        #if is_paint==1:
        #    paint("self-alpha",alpha)
        weighted_x  = weighted_avg(x, alpha)
        return alpha,weighted_x


class LayerNorm(torch.nn.Module):
    # pylint: disable=line-too-long
    """
    An implementation of `Layer Normalization
    <https://www.semanticscholar.org/paper/Layer-Normalization-Ba-Kiros/97fb4e3d45bb098e27e0071448b6152217bd35a5>`_ .

    Layer Normalization stabilises the training of deep neural networks by
    normalising the outputs of neurons from a particular layer. It computes:

    output = (gamma * (tensor - mean) / (std + eps)) + beta

    Parameters
    ----------
    dimension : ``int``, required.
        The dimension of the layer output to normalize.
    eps : ``float``, optional, (default = 1e-6)
        An epsilon to prevent dividing by zero in the case
        the layer has zero variance.

    Returns
    -------
    The normalized layer output.
    """
    def __init__(self,
                 dimension: int,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(dimension))
        self.beta = torch.nn.Parameter(torch.zeros(dimension))
        self.eps = eps

    def forward(self, tensor: torch.Tensor):  # pylint: disable=arguments-differ
        mean = tensor.mean(-1, keepdim=True)
        std = tensor.std(-1, unbiased=False, keepdim=True)
        return self.gamma * (tensor - mean) / (std + self.eps) + self.beta




class MultiTurnInference(nn.Module):
    #Given sequences X and summary_Y, match X and summary_Y and infer by BiLSTM.


    def __init__(self, args,RNN_TYPES):
        super(MultiTurnInference, self).__init__()

        self.args = args
        #self.cat_linear= nn.Linear(3*2*args.hidden_size,args.hidden_size)
        if "c" in args.matching_order:
            #self.cat_linear= nn.Linear(args.cat_num*2*args.hidden_size,args.hidden_size)
            self.cat_linear= nn.Linear(4*args.hidden_size,args.hidden_size)
        if "s" in args.matching_order:
            self.sub_linear= nn.Linear(2*args.hidden_size,args.hidden_size)
        if "m" in args.matching_order:
            self.mul_linear= nn.Linear(2*args.hidden_size,args.hidden_size)

        self.order_buffer={'c':None,'s':None,'m':None}

        #print ("(2+1)*args.hidden_size",(2+1)*args.hidden_size)
        #print ("self.inp_linear.weight",self.inp_linear.weight)
        if self.args.use_multiturn_infer == True:
            self.inp_linear= nn.Linear((2+1)*args.hidden_size,args.hidden_size) #2:mem 1:inp
            self.out_linear= nn.Linear((2+2)*args.hidden_size, 2*args.hidden_size) #2:mem 2:out

            self.inference_rnn = StackedBRNN(
                input_size=args.hidden_size,
                hidden_size=args.hidden_size,
                num_layers=args.infer_layers,
                dropout_rate=args.dropout_rnn_output,  # float
                dropout_output=args.rnn_output_dropout,  #True or False
                concat_layers=False,
                rnn_type=RNN_TYPES[args.rnn_type],
                padding=args.rnn_padding)
        else:
            if  self.args.use_bilstm==True:
                if self.args.use_conv==True:
                    self.conv =nn.Conv2d(in_channels=len(args.matching_order), 
                                         out_channels=1, kernel_size=(1,1), 
                                         stride=1, padding=0, dilation=1, groups=1, bias=True)
                    self.inference_rnn = StackedBRNN(
                        input_size=args.hidden_size,
                        hidden_size=args.hidden_size,
                        num_layers=args.infer_layers,
                        dropout_rate=args.dropout_rnn_output,  # float
                        dropout_output=args.rnn_output_dropout,  #True or False
                        concat_layers=False,
                        rnn_type=RNN_TYPES[args.rnn_type],
                        padding=args.rnn_padding)
                else:
                    self.inference_rnn = StackedBRNN(
                        input_size=len(args.matching_order)*args.hidden_size,
                        hidden_size=args.hidden_size,
                        num_layers=args.infer_layers,
                        dropout_rate=args.dropout_rnn_output,  # float
                        dropout_output=args.rnn_output_dropout,  #True or False
                        concat_layers=False,
                        rnn_type=RNN_TYPES[args.rnn_type],
                        padding=args.rnn_padding)
            else:
                self.inp_self_attn_linear= nn.Linear(len(self.args.matching_order)*args.hidden_size, 2*args.hidden_size) #2:mem 1:inp
                #self.inp_self_attn_linear= nn.Linear((len(self.args.matching_order)*2)*args.hidden_size, 2*args.hidden_size) #2:mem 1:inp
    def forward(self, x,x_mask,init_mem, x_order="csm",is_paint=0):

        x_linear = self._input_linear(x,x_order,is_paint=is_paint)
        mem_list = []
        mem_gate_list = []

        if self.args.use_multiturn_infer == True:
            infer_emb,mem_list, mem_gate_list = self._multi_turn_inference(inp=x_linear,inp_mask =x_mask,init_mem=init_mem,is_paint=is_paint)
           
        else:
            infer_emb ,mem_list, mem_gate_list= self._single_turn_inference(inp=x_linear,inp_mask =x_mask,init_mem=init_mem,is_paint=is_paint)
        return infer_emb,mem_list, mem_gate_list

    def _input_linear(self, x,x_order,is_paint=0):
        '''
        convert the element in x to fixed dimension 
        x: a list of 2d tensor 
        '''
        if 'c' in self.args.matching_order:
            cat =F.relu(self.cat_linear(x['c']))
            self.order_buffer['c']=cat
        if 's' in self.args.matching_order:
            sub =F.relu(self.sub_linear(x['s'])) 
            self.order_buffer['s']=sub
        if 'm' in self.args.matching_order:
            mul =F.relu(self.mul_linear(x['m'])) 
            self.order_buffer['m']=mul

        if is_paint==1:
            paint('Union',cat)
            paint('Difference',sub)
            paint('Similar',mul)
     

        self.matching_order = {
        "csm":  [self.order_buffer['c'],self.order_buffer['s'],self.order_buffer['m']],
        "cms":  [self.order_buffer['c'],self.order_buffer['m'],self.order_buffer['s']],
        "smc":  [self.order_buffer['s'],self.order_buffer['m'],self.order_buffer['c']], 
        "scm":  [self.order_buffer['s'],self.order_buffer['c'],self.order_buffer['m']],
        "mcs":  [self.order_buffer['m'],self.order_buffer['c'],self.order_buffer['s']],
        "msc":  [self.order_buffer['m'],self.order_buffer['s'],self.order_buffer['c']],

        "cs": [self.order_buffer['c'],self.order_buffer['s']],
        "cm": [self.order_buffer['c'],self.order_buffer['m']],
        "sc": [self.order_buffer['s'],self.order_buffer['c']],
        "sm": [self.order_buffer['s'],self.order_buffer['m']],
        "ms": [self.order_buffer['m'],self.order_buffer['s']],  
        "mc": [self.order_buffer['m'],self.order_buffer['c']],
        "c":  [self.order_buffer['c']],
        "s":  [self.order_buffer['s']],
        "m":  [self.order_buffer['m']],
        } 
        #out =  self.matching_order.get(x_order)
        out =  self.matching_order[x_order]

        #return [cat,sub,mul]
        # map the suquence by dict:matching order
        return out
    def _single_turn_inference(self,inp,inp_mask,init_mem,is_paint=0):

        inp_cat = torch.cat(inp,-1)

        if self.args.use_bilstm==True:
            if self.args.use_conv==True:
                rnn_input=self.conv(torch.stack(inp,1)).squeeze(1) #b,l,d
                rnn_output = self.inference_rnn(rnn_input,inp_mask) #2*hiden_size
            else:
                rnn_output = self.inference_rnn(inp_cat,inp_mask) #2*hiden_size
            #output_gate  = F.sigmoid( self.out_linear( torch.cat([rnn_output,init_mem],-1))) 
            #output = output_gate*rnn_output + (1-output_gate)*init_mem
            return rnn_output,None,None
        else: 
            inp_linear = F.relu(self.inp_self_attn_linear(inp_cat))
            return inp_linear,None, None

    def _multi_turn_inference(self,inp,inp_mask,init_mem,is_paint=0):
        '''
        multi_turn inferring the x
        '''
        #print ("###########init_mem",init_mem)
        #print ("###########type(init_mem)",type(init_mem))
        '''
        if init_mem !=None: 
            prev_mem = init_mem
        else:
            dim = inp[0].size()
            prev_mem = torch.zeros(dim[0],dim[1],2*dim[2]).float().cuda()
        '''
        prev_mem =init_mem
        mem_list = []
        mem_gate_list = []
        for ele in inp: 
            ele_linear = self.inp_linear(torch.cat([ele,prev_mem],-1))
            rnn_output = self.inference_rnn(ele_linear,inp_mask) #2*hiden_size

            output_gate  = F.sigmoid( self.out_linear( torch.cat([rnn_output,prev_mem],-1))) 
            output = output_gate*rnn_output + (1-output_gate)*prev_mem
            prev_mem = output
            mem_list.append(prev_mem)
            mem_gate_list.append(output_gate)
        #outputs = torch.cat(output_list,-1)
        outputs = prev_mem
        return outputs, mem_list, mem_gate_list


# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------


def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        x_mask: batch * len (1 for padding, 0 for true)
    Output:
        x_avg: batch * hdim
    """
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha


def weighted_avg(x, weights):
    """Return a weighted average of x (a sequence of vectors).

    Args:
        x: batch * len * hdim
        weights: batch * len, sum(dim = 1) = 1
    Output:
        x_avg: batch * hdim
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)


def max_pooling(inp):
    return torch.max(inp,1)[0]  #[max_value,max_indices]

def ave_pooling(inp, inp_mask):
    inp_len = inp_mask.size()[-1] - torch.sum(inp_mask,-1)
    inp_len = inp_len.float()

    inp_sum = torch.sum(inp, 1)  #(b,d_len.2*h) ->(b,2*h)

    inp_ave = torch.div(inp_sum, inp_len.unsqueeze(-1) ) #div true length

    return inp_ave


