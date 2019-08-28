import os
import argparse
import logging

logger = logging.getLogger(__name__)

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

parser = argparse.ArgumentParser()
parser.register('type', 'bool', str2bool)
parser.add_argument("model_name", type=str, help="Give model name.")
parser.add_argument("--train_name", type=str, default='model_00', help="Give train name.")
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--epoch', type=int, default=30, help='Number of epoches to run')
parser.add_argument('--optimizer', type=str, default='adamax', help='optimizer, adamax or sgd')
parser.add_argument('--use_cuda', type='bool', default=True, help='use cuda or not')
parser.add_argument('--grad_clipping', type=float, default=10.0, help='maximum L2 norm for gradient clipping')
parser.add_argument('--l2_strength', type=float, default=3e-8, help='learning rate')
parser.add_argument('--lr', type=float, default=6e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--embedding_file', type=str, default='/home/hd_1T/haiou/test0730_race_update/data/glove.840B.300d.txt', help='embedding file')
parser.add_argument('--hidden_size', type=int, default=96, help='default size for RNN layer')
parser.add_argument('--infer_layers', type=int, default=1, help='number of RNN layers for inference')
parser.add_argument('--doc_layers', type=int, default=1, help='number of RNN layers for doc encoding')
parser.add_argument('--rnn_type', type=str, default='lstm', help='RNN type, lstm or gru')
parser.add_argument('--dropout_rnn_output', type=float, default=0.20, help='dropout for RNN output')
parser.add_argument('--dropout_att_score', type=float, default=0.30, help='dropout for attention score')
parser.add_argument('--dropout_residual', type=float, default=0.10, help='dropout for the residual connection')
parser.add_argument('--dropout_emb', type=float, default=0.4, help='dropout rate for embeddings')
parser.add_argument('--dropout_init_mem_emb', type=float, default=0.5, help='dropout rate for initial memory embeddings')
parser.add_argument('--pretrained', type=str, default='', help='pretrained model path')
parser.add_argument('--rnn_padding', type='bool', default=True, help='Use padding or not')
parser.add_argument('--rnn_output_dropout', type='bool', default=True, help='Dropout rnn_output or not')
parser.add_argument('--finetune_topk', type=int, default=10, help='Finetune topk embeddings during training')
parser.add_argument('--pos_emb_dim', type=int, default=12, help='Embedding dimension for part-of-speech')
parser.add_argument('--ner_emb_dim', type=int, default=8, help='Embedding dimension for named entities')
parser.add_argument('--rel_emb_dim', type=int, default=10, help='Embedding dimension for ConceptNet relations')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--test_mode', type='bool', default=False, help='In test mode, validation data will be used for training')


parser.add_argument('--matching_order', type=str, default='csm', help='the matching order of matching sequence')
parser.add_argument('--cat_num', type=int, default=2, help='the number of concatenation in matching')
parser.add_argument('--use_multiturn_infer', type='bool', default=True, help='use multi_turn inference or not')
parser.add_argument('--use_bilstm', type='bool', default=True, help='use bilstm in single turn or not')
parser.add_argument('--use_bimemory', type='bool', default=False, help='use both forward abd backward multi_turn inference or not')
parser.add_argument('--c_rnn_input_type', type=str, default='pqc', help='the type of c_rnn_input, one of c/pc/qc/pqc')
parser.add_argument('--k_held_out', type=int, default=10, help='cross validation')
parser.add_argument('--checkpoint_name', type=str, default='best', help='the type of c_rnn_input, one of c/pc/qc/pqc')
parser.add_argument('--matched_p', type=str, default='pcq', help='the type of matched_q, one of p/pc/pq/pqc')
parser.add_argument('--beta', type=float, default=10.0, help='coefficient for mul feature')

parser.add_argument('--use_conv', type='bool', default=False, help='use the conv to process the csm or not,for no multi-turn')

parser.add_argument('--tri_input', type=str, default='NA', help='the type of tri-matching input, one of NA/CA/NA_CA/CA_NA')
parser.add_argument('--p_channel', type='bool', default=False, help='use the p channel fro tri-matching')
parser.add_argument('--q_channel', type='bool', default=False, help='use the q channel fro tri-matching')
parser.add_argument('--c_channel', type='bool', default=True, help='use the c channel fro tri-matching')
parser.add_argument('--max_length', type=int, default=50, help='max length of passage')
parser.add_argument('--debug', type='bool', default=False, help='use the dev as traing to debug or not')
args = parser.parse_args()

print(args)

if args.pretrained:
    assert all(os.path.exists(p) for p in args.pretrained.split(',')), 'Checkpoint %s does not exist.' % args.pretrained
