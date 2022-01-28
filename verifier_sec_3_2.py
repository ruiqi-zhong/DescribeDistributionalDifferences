import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSequenceClassification, T5Config
import re
import numpy as np
import tqdm
import random

# Using the UnifiedQA (T5 tokenizer)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_count = torch.cuda.device_count()
tok = AutoTokenizer.from_pretrained('allenai/unifiedqa-t5-11b')
BSIZE = 32


# resize the samples in the UnifiedQA context
def resize(sent_A, sent_B, max_length):
    combined_cap = max_length - 30
    toks_A = tok(sent_A)['input_ids']
    toks_B = tok(sent_B)['input_ids']
    
    toks_A_new, toks_B_new = [], []
    total_token_count = 0
    for i in range(max(len(toks_A), len(toks_B)) - 1):
        if total_token_count >= combined_cap:
            break
        if i < len(toks_A) - 1:
            toks_A_new.append(toks_A[i])
            total_token_count += 1
        
        if total_token_count >= combined_cap:
            break
        if i < len(toks_B) - 1:
            toks_B_new.append(toks_B[i])
            total_token_count += 1
    new_A, new_B = tok.decode(toks_A_new), tok.decode(toks_B_new)
    return new_A, new_B


def normalize(t):
    return re.sub("'(.+)'", r'\1', t.lower())


def qc2input(d):
    return normalize(d['q'] + '\\n' + d['c'])


class T5ZeroShotClfQA(torch.nn.Module):

    def __init__(self, qa_model_name, max_seq_length = 128, half_precision=True):
        super(T5ZeroShotClfQA, self).__init__()
        if 'scratch' not in qa_model_name:
            self.model = T5ForConditionalGeneration.from_pretrained(qa_model_name)#.to(device)
        else:
            self.model = T5ForConditionalGeneration(T5Config.from_pretrained(qa_model_name.replace('scratch', 't5'))) 
        if half_precision:
            print('Using half precision')
            self.half_precision = half_precision
            self.model = self.model.half()
        if device == 'cuda':
            self.model.to(device)
        self.vocab = tok.get_vocab()
        self.yes_id, self.no_id = self.vocab['▁yes'], self.vocab['▁no']
        self.max_seq_length = max_seq_length
        self.lsm = torch.nn.LogSoftmax(dim=-1)

    def create_batch(self, q_dicts):
        input_strings = [qc2input(d) for d in q_dicts]
        input_strings = [normalize(i) for i in input_strings]
        input_dict = tok(input_strings, padding=True, return_tensors="pt",
                                    truncation=True, max_length=self.max_seq_length).to(device)
        return input_dict

    def forward(self, input_dict):
        starts = torch.tensor([[self.model.config.decoder_start_token_id]] * len(input_dict['input_ids'])).to(device)
        output = self.model(**input_dict, decoder_input_ids=starts)
        logits = self.lsm(output.logits[:, 0, [self.no_id, self.yes_id]])
        return logits

    def get_logits_from_input_dict_(self, input_strings):
        input_dict = self.create_batch(input_strings)
        return self.forward(input_dict)

    def get_logits_from_input_dict(self, q_dicts, bsize=32, progress_bar=True):
        self.model.eval()
        result_logits = []
        iter_count = (len(q_dicts) - 1) // bsize + 1
        ranger = range(iter_count) if not progress_bar else tqdm.trange(iter_count)
        for i in ranger:
            l = self.get_logits_from_input_dict_(q_dicts[i*bsize:(i+1) * bsize]).detach().cpu().numpy().tolist()
            result_logits.extend(l)
        return np.array(result_logits)        


# h is a natural language hypothesis
# pos, neg are list of samples
# num_examples are the number of positive-negative pairs to evaluate on
# m is the same class as T5ZeroShotClfQA (see below)
def compute_classification_accuracy(s, pos, neg, num_examples, m, max_length=256):
    q = 'Is it true that compared to sentence B, sentence A ' + s + '?'
    
    pairs = []
    for i in range(num_examples):
        sent_A = random.choice(pos)
        sent_B = random.choice(neg)
        pairs.append((sent_A, sent_B))

    qc_dicts = []
    for sent_A, sent_B in pairs:
        sent_A, sent_B = resize(sent_A, sent_B, max_length)
        c = 'sentence A: ' + sent_A + '\n\nsentence B: ' + sent_B
        
        ############### Uncomment to see what the prompt looks like ############### 
#         print('q', q)
#         print('c', c)
#         exit(0)

        qc_dicts.append({'q': q, 'c': c})
    positive_logits = m.get_logits_from_input_dict(qc_dicts, bsize=BSIZE)
    
    # V(s, x_{1}, x_{0})
    pos_score = np.mean((np.e ** positive_logits[:,1]) > 0.5)

    qc_dicts = []

    for sent_A, sent_B in pairs:
        sent_A, sent_B = resize(sent_A, sent_B, max_length)
        c = 'sentence A: ' + sent_B + '\n\nsentence B: ' + sent_A
        qc_dicts.append({'q': q, 'c': c})
    reverse_logits = m.get_logits_from_input_dict(qc_dicts, bsize=BSIZE)
    
    # V(s, x_{0}, x_{1})
    reverse_score = np.mean((np.e ** reverse_logits[:,1]) > 0.5)
    
    return {
        'classification_accuracy': ((pos_score - reverse_score) + 1) / 2,
        'dicts': pairs,
        'logits': {
            'positive_logits': positive_logits,
            'reverse_logits': reverse_logits
        }
    }

# natural language hypothesis
s = 'is a positive review'

# D_0
neg = [
    'I hate this film.',
    'Total waste of time.',
    'Not recommended'
]

# D_1
pos = [
    'I like this film!!',
    'The best movie I have seen.',
    'The director did a good job attracting the audience attention.'
]

# number of sample pairs to estimate the classification accuracy
num_samples = 16

# initialize the model
# this is a proof-of-concept code. model size this small does not work
# even for simple examples like this you might need an 11B parameter model to work well
model_size = 'small'
m = T5ZeroShotClfQA('allenai/unifiedqa-t5-%s' % model_size)
# you can also replace it with our fine-tuned verifier
# "ruiqi-zhong/verifier11b"

# calculate the classification accuracy of the hypothesis, approximated by the model m, 
# using num_samples pairs of positive-negative samples
result = compute_classification_accuracy(s, pos, neg, num_samples, m, max_length=256)
print(result['classification_accuracy'])


