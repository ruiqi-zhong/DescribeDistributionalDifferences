import sys
sys.path.append('./')
from models.engine import Engine
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import random
from gadgets.util import parallelize_across_device


class T5Proposer:

    def __init__(self, model_size=None, model_path=None):
        try:
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        except Exception as e:
            print(e)
            self.tokenizer = T5Tokenizer.from_pretrained("../workflow/mount/models/t5tok/")
        self.tokenizer.model_max_length = 1024
        if model_path is None:
            try:
                self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-%s" % model_size)
            except Exception as e:
                print(e)
                self.model = T5ForConditionalGeneration.from_pretrained("../workflow/mount/models/flan-t5-xxl")
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        parallelize_across_device(self.model)
        self.model_tokenizer = (self.model, self.tokenizer)
        self.engine = Engine(self.model_tokenizer)
    
    def propose(self, input_dicts, bsize=8):
        return self.engine.propose_hypotheses(input_dicts, bsize=bsize)


if __name__ == '__main__':
    test_complete_data = json.load(open('models/old_data/test_complete_data.json', 'r'))
    input_dicts = []
    for d in test_complete_data:
        input_dict = {
            'pos_sents': random.sample(d['pos'], 4),
            'neg_sents': random.sample(d['neg'], 4)
        }
        input_dicts.append(input_dict)
    proposer = T5Proposer('xxl')
    for result in proposer.propose(input_dicts):
        print(result)






