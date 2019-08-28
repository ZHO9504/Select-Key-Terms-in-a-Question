import os
import csv
import sys
import spacy
import copy
import json
import math
import wikiwords
import time

from collections import Counter

class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """
        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups

class SpacyTokenizer():

    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').
        """
        model = kwargs.get('model', 'en')
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        nlp_kwargs = {'parser': False}
        if not {'lemma', 'pos', 'ner'} & self.annotators:
            nlp_kwargs['tagger'] = False
        if not {'ner'} & self.annotators:
            nlp_kwargs['entity'] = False
        self.nlp = spacy.load(model, **nlp_kwargs)

    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ').replace('\t', ' ').replace('/', ' / ').strip()
        # remove consecutive spaces
        if clean_text.find('  ') >= 0:
            clean_text = ' '.join(clean_text.split())
        tokens = self.nlp.tokenizer(clean_text)
        if {'lemma', 'pos', 'ner'} & self.annotators:
            self.nlp.tagger(tokens)
        if {'ner'} & self.annotators:
            self.nlp.entity(tokens)

        data = []
        for i in range(len(tokens)):
            # Get whitespace
            start_ws = tokens[i].idx
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1].idx
            else:
                end_ws = tokens[i].idx + len(tokens[i].text)

            data.append((
                tokens[i].text,
                text[start_ws: end_ws],
                (tokens[i].idx, tokens[i].idx + len(tokens[i].text)),
                tokens[i].tag_,
                tokens[i].lemma_,
                tokens[i].ent_type_,
            ))

        # Set special option for non-entity tag: '' vs 'O' in spaCy
        return Tokens(data, self.annotators, opts={'non_ent': ''})


TOK = None

def init_tokenizer():
    global TOK
    TOK = SpacyTokenizer(annotators={'pos', 'lemma', 'ner'})


digits2w = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three',
            '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}
def replace_digits(words):
    global digits2w
    return [digits2w[w] if w in digits2w else w for w in words]

def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)
    output = {
        'words': replace_digits(tokens.words()),
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'lemma': tokens.lemmas(),
        'ner': tokens.entities(),
    }
    return output


def preprocess_dataset_test(path, is_test_set=False):
    writer = open(path.replace('.csv', '') + '-processed2.json', 'w', encoding='utf-8')
    dataset = []
    ex_cnt = 0
    for line_count, sp in enumerate(csv.reader(open(path))):
        if line_count == 0:
            continue
        # InputStoryid    InputSentence1  InputSentence2  InputSentence3
        # InputSentence4  RandomFifthSentenceQuiz1
        # RandomFifthSentenceQuiz2        AnswerRightEnding
        
        sp = [s.strip() for s in sp]

        id, plots,climax,ending1,ending2,label = path + '_' + sp[0], sp[1:4], sp[4], sp[5], sp[6], sp[7]
        plot = ""  
        for p in plots:
            plot += " " +p
        d_dict = tokenize(plot)
        q_dict = tokenize(climax)
        c2_dict = tokenize(ending2)
        c1_dict = tokenize(ending1)
        if label == '1':
            example = get_example(id+'|||1', d_dict, q_dict, c1_dict, '1')
            example.update(compute_features(d_dict, q_dict, c1_dict))
            writer.write(json.dumps(example))
            writer.write('\n')
            example = get_example(id+'|||2', d_dict, q_dict, c2_dict, '0')
            example.update(compute_features(d_dict, q_dict, c2_dict))
            writer.write(json.dumps(example))
            writer.write('\n')
            ex_cnt += 1
        if label == '2':
            example = get_example(id+'|||1', d_dict, q_dict, c1_dict, '0')
            example.update(compute_features(d_dict, q_dict, c1_dict))
            writer.write(json.dumps(example))
            writer.write('\n')
            example = get_example(id+'|||2', d_dict, q_dict, c2_dict, '1')
            example.update(compute_features(d_dict, q_dict, c2_dict))
            writer.write(json.dumps(example))
            writer.write('\n')
            ex_cnt += 1

    print('Found %d examples in %s...' % (ex_cnt, path))
    writer.close()

from utils import is_stopword, is_punc
def compute_features(p_dict, q_dict, c_dict):
    # p_in_q, p_in_c, lemma_p_in_q, lemma_p_in_c, tf


    p_words_set = set([w.lower() for w in p_dict['words']])
    q_words_set = set([w.lower() for w in q_dict['words']])
    c_words_set = set([w.lower() for w in c_dict['words']])

    p_in_q = [int(w.lower() in q_words_set and not is_stopword(w) and not is_punc(w)) for w in p_dict['words']]
    p_in_c = [int(w.lower() in c_words_set and not is_stopword(w) and not is_punc(w)) for w in p_dict['words']]

    q_in_p = [int(w.lower() in p_words_set and not is_stopword(w) and not is_punc(w)) for w in q_dict['words']]
    q_in_c = [int(w.lower() in c_words_set and not is_stopword(w) and not is_punc(w)) for w in q_dict['words']]

    c_in_p = [int(w.lower() in p_words_set and not is_stopword(w) and not is_punc(w)) for w in c_dict['words']]
    c_in_q = [int(w.lower() in q_words_set and not is_stopword(w) and not is_punc(w)) for w in c_dict['words']]


    p_words_set = set([w.lower() for w in p_dict['lemma']])
    q_words_set = set([w.lower() for w in q_dict['lemma']])
    c_words_set = set([w.lower() for w in c_dict['lemma']])
    p_lemma_in_q = [int(w.lower() in q_words_set and not is_stopword(w) and not is_punc(w)) for w in p_dict['lemma']]
    p_lemma_in_c = [int(w.lower() in c_words_set and not is_stopword(w) and not is_punc(w)) for w in p_dict['lemma']]
 
    q_lemma_in_p = [int(w.lower() in p_words_set and not is_stopword(w) and not is_punc(w)) for w in q_dict['lemma']]
    q_lemma_in_c = [int(w.lower() in c_words_set and not is_stopword(w) and not is_punc(w)) for w in q_dict['lemma']]

    c_lemma_in_p = [int(w.lower() in p_words_set and not is_stopword(w) and not is_punc(w)) for w in c_dict['lemma']]
    c_lemma_in_q = [int(w.lower() in q_words_set and not is_stopword(w) and not is_punc(w)) for w in c_dict['lemma']]

    p_tf = [0.1 * math.log(wikiwords.N * wikiwords.freq(w.lower()) + 10) for w in p_dict['words']]
    p_tf = [float('%.2f' % v) for v in p_tf]
    q_tf = [0.1 * math.log(wikiwords.N * wikiwords.freq(w.lower()) + 10) for w in q_dict['words']]
    q_tf = [float('%.2f' % v) for v in q_tf]
    c_tf = [0.1 * math.log(wikiwords.N * wikiwords.freq(w.lower()) + 10) for w in c_dict['words']]
    c_tf = [float('%.2f' % v) for v in c_tf]
    d_words = Counter(filter(lambda w: not is_stopword(w) and not is_punc(w), p_dict['words']))
    
    from conceptnet import concept_net
    p_q_relation = concept_net.p_q_relation(p_dict['words'], q_dict['words'])
    p_c_relation = concept_net.p_q_relation(p_dict['words'], c_dict['words'])

    q_p_relation = concept_net.p_q_relation(q_dict['words'], p_dict['words'])
    q_c_relation = concept_net.p_q_relation(q_dict['words'], c_dict['words'])

    c_p_relation = concept_net.p_q_relation(c_dict['words'], p_dict['words'])
    c_q_relation = concept_net.p_q_relation(c_dict['words'], q_dict['words'])
 

    assert len(p_tf) == len(p_q_relation) and len(p_tf) == len(p_c_relation)
    assert len(q_tf) == len(q_p_relation) and len(q_tf) == len(q_c_relation)
    assert len(c_tf) == len(c_p_relation) and len(c_tf) == len(c_q_relation)


    return {
        'p_in_q': p_in_q,
        'p_in_c': p_in_c,
        'p_lemma_in_q': p_lemma_in_q,
        'p_lemma_in_c': p_lemma_in_c,
        'p_tf': p_tf,
        'p_q_relation': p_q_relation,
        'p_c_relation': p_c_relation,

        'q_in_p': q_in_p,
        'q_in_c': q_in_c,
        'q_lemma_in_p': q_lemma_in_p,
        'q_lemma_in_c': q_lemma_in_c,
        'q_tf': q_tf,
        'q_p_relation': q_p_relation,
        'q_c_relation': q_c_relation,

        'c_in_p': c_in_p,
        'c_in_q': c_in_q,
        'c_lemma_in_p': c_lemma_in_p,
        'c_lemma_in_q': c_lemma_in_q,
        'c_tf': c_tf,

        'c_p_relation': c_p_relation,
        'c_q_relation': c_q_relation,

    }


def get_example(id, plot_dict, climax_dict, ending1_dict,label = '1' ):
    return {
            'id': id,
            'p_words': ' '.join(plot_dict['words']),
            'p_pos': plot_dict['pos'],
            'p_ner': plot_dict['ner'],
            'q_words': ' '.join(climax_dict['words']),
            'q_pos': climax_dict['pos'],
            'q_ner': climax_dict['ner'],
            'c_words': ' '.join(ending1_dict['words']),
            'c_pos': ending1_dict['pos'],
            'c_ner': ending1_dict['ner'],
            'label': label
        }


def preprocess_dataset_train(path, is_test_set=False):
    writer = open(path.replace('.csv', '') + '-processed.json', 'w', encoding='utf-8')
    dataset = []
    ex_cnt = 0
    for line_count, sp in enumerate(csv.reader(open(path))):
        if line_count == 0:
            continue
        # InputStoryid    InputSentence1  InputSentence2  InputSentence3
        # InputSentence4  RandomFifthSentenceQuiz1
        # RandomFifthSentenceQuiz2        AnswerRightEnding
        
        sp = [s.strip() for s in sp]

     
        id, plots,climax,ending1 = path + '_' + sp[0], sp[1:4], sp[4], sp[5]
        plot = ""  
        for p in plots:
            plot += " " +p
        d_dict = tokenize(plot)
        q_dict = tokenize(climax)
        c1_dict = tokenize(ending1)
		
        example = get_example(id, d_dict, q_dict, c1_dict)
        example.update(compute_features(d_dict, q_dict, c1_dict))
        writer.write(json.dumps(example))
        writer.write('\n')
        ex_cnt += 1
    print('Found %d examples in %s...' % (ex_cnt, path))
    writer.close()

def _get_race_obj(d):
    for root_d, _, files in os.walk(d):
        for f in files:
            if f.endswith('txt'):
                obj = json.load(open(root_d + '/' + f, 'r', encoding='utf-8'))
                yield obj

def preprocess_dataset_race(path, is_test_set=False):
    start_time = time.time()
    writer = open(path  + '-processed.json', 'w', encoding='utf-8')
    ex_cnt = 0
    for line in open(path, 'r', encoding='utf-8'):
            obj = json.loads(line)
            if not obj['questions']:
                continue
            d_dict = tokenize(obj['article'])
            d_id = obj['id']
            qs = [q for q in obj['questions']]
            opss = [ops for ops in obj['options']]
            anss = [ans for ans in obj['answers']]
            length = len(qs)
            d = 0
            for i in range(length):
                q_dict = tokenize(qs[i])
                n = 0
                if anss[i] == 'A':
                    d = 1
                if anss[i] == 'B':
                    d = 2
                if anss[i] == 'C':
                    d = 3
                if anss[i] == 'D':
                    d = 4
                #print(d)
                for j in range(4):
                    #print(anss[i])
                    if j + 1 == d:
                        label = "1"
                    else: label = "0"
                    #print(label)
                    c_dict = tokenize(opss[i][j])
                    example = get_example(d_id + '_' + str(i)+'_'+ str(j), d_dict, q_dict, c_dict,label)
                    example.update(compute_features(d_dict, q_dict, c_dict))
                    writer.write(json.dumps(example))
                    writer.write('\n')
                    ex_cnt += 1

    print('Found %d examples in %s...   use %d seconds' % (ex_cnt, path, time.time()-start_time))
    writer.close()




def preprocess_race_dataset(d):
    import utils
    utils.build_vocab()
    def is_passage_ok(words):
        return len(words) >= 50 and len(words) <= 500 and sum([int(w in utils.vocab) for w in words]) >= 0.85 * len(words)
    def is_question_ok(words):
        return True
    def is_option_ok(words):
        s = ' '.join(words).lower()
        return s != 'all of the above' and s != 'none of the above'
    writer = open('./data/race-processed.json', 'w', encoding='utf-8')
    ex_cnt = 0
    for obj in _get_race_obj(d):
        d_dict = tokenize(obj['article'].replace('\n', ' ').replace('--', ' '))
        if not is_passage_ok(d_dict['words']):
            continue
        d_id = obj['id']
        assert len(obj['options']) == len(obj['answers']) and len(obj['answers']) == len(obj['questions'])
        q_cnt = 0
        for q, ans, choices in zip(obj['questions'], obj['answers'], obj['options']):
            q_id = str(q_cnt)
            q_cnt += 1
            ans = ord(ans) - ord('A')
            assert 0 <= ans < len(choices)
            q_dict = tokenize(q.replace('_', ' _ '))
            if not is_question_ok(q_dict['words']):
                continue
            for c_id, choice in enumerate(choices):
                c_dict = tokenize(choice)
                if not is_option_ok(c_dict['words']):
                    continue
                label = int(c_id == ans)
                c_id = str(c_id)
                example = get_example(d_id, q_id, c_id, d_dict, q_dict, c_dict, label)
                example.update(compute_features(d_dict, q_dict, c_dict))
                writer.write(json.dumps(example))
                writer.write('\n')
                ex_cnt += 1
    print('Found %d examples in %s...' % (ex_cnt, d))
    writer.close()

def preprocess_conceptnet(path):
    import utils
    utils.build_vocab()
    writer = open('concept.filter', 'w', encoding='utf-8')
    def _get_lan_and_w(arg):
        arg = arg.strip('/').split('/')
        return arg[1], arg[2]
    for line in open(path, 'r', encoding='utf-8'):
        fs = line.split()
        relation, arg1, arg2 = fs[1].split('/')[-1], fs[2], fs[3]
        lan1, w1 = _get_lan_and_w(arg1)
        if lan1 != 'en' or not all(w in utils.vocab for w in w1.split('_')):
            continue
        lan2, w2 = _get_lan_and_w(arg2)
        if lan2 != 'en' or not all(w in utils.vocab for w in w2.split('_')):
            continue
        obj = json.loads(fs[-1])
        if obj['weight'] < 1.0:
            continue
        writer.write('%s %s %s\n' % (relation, w1, w2))
    writer.close()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'conceptnet':
        preprocess_conceptnet('conceptnet-assertions-5.5.5.csv')
        exit(0)
    init_tokenizer()
    #preprocess_dataset_race('./data/dev_middle')
    #preprocess_dataset_race('./data/test_middle')
    #preprocess_dataset_race('./data/train_middle')

    import utils
    train_data = utils.load_data('./data/train_middle-processed.json')
    dev_data = utils.load_data('./data/dev_middle-processed.json')
    test_data =utils.load_data('./data/test_middle-processed.json')
    utils.build_vocab(train_data+dev_data)
    #utils.build_vocab(test_data)

