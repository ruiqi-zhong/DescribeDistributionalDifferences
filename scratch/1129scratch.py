import pickle as pkl
import sys
sys.path.append('./')
import random
from models.preprocess import construct_blocks, tok_subspan
from transformers import AutoTokenizer
import openai
import os
from gadgets.util import gpt3wrapper
import tqdm

openai.api_key = os.environ['openai_key']

benchmark = pkl.load(open('data/benchmark_1128.pkl', 'rb'))
tok = AutoTokenizer.from_pretrained('gpt2-medium')
SINGLE_SAMPLE_MAX_LENGTH = 256
proposer_template1129 = open('models/templates/1129proposer_template_w_context.txt').read()
MAX_PROMPT_LENGTH = 2048
query_count = 0

logs = []
for _ in range(2):
    print('iteration ', _)
    for pair_id, pair in tqdm.tqdm(enumerate(benchmark)):
        dataset_description = pair['dataset_description']
        generation = pair['generation']
        positive_description = pair['pos_desc']
        negative_description = pair['neg_desc']
        user = pair['user']
        K = 100

        pos_samples, neg_samples = pair['pos_samples'], pair['neg_samples']
        if len(pos_samples) > K:
            pos_samples = random.sample(pos_samples, K)
        if len(neg_samples) > K:
            neg_samples = random.sample(neg_samples, K)

        A_sentences = [tok_subspan(x, SINGLE_SAMPLE_MAX_LENGTH, tok) for x in pos_samples]
        B_sentences = [tok_subspan(x, SINGLE_SAMPLE_MAX_LENGTH, tok) for x in neg_samples]

        for target in pair['target']:
            num_incontext_samples = 10
            prompt = None
            while num_incontext_samples > 1:
                sent_subset = construct_blocks(A_sentences, B_sentences, num_incontext_samples=num_incontext_samples)
                
                A_block, B_block = sent_subset['A_block'], sent_subset['B_block']

                arg_dict = {
                    'dataset_description': dataset_description,
                    'generation': generation,
                    'positive_description': positive_description,
                    'negative_description': negative_description,
                    'user': user,
                    'target': target,
                    'A_block': A_block,
                    'B_block': B_block
                }
                prompt = proposer_template1129.format(**arg_dict)

                prompt_length = len(tok.encode(prompt))
                if prompt_length < MAX_PROMPT_LENGTH:
                    break
                else:
                    num_incontext_samples -= 1
            
            query_args = {
                'engine': 'text-davinci-003',
                'prompt': prompt,
                'temperature': 0.7,
                'max_tokens': 512,
                'top_p': 1,
                'n': 1
            }
            query_count += 1

            result = gpt3wrapper(
                **query_args
            )


            save_result = {
                'pair_id': pair_id,
                'result': result,
                'sent_subset': sent_subset,
                'query_args': query_args
            }
            logs.append(save_result)
            pkl.dump(logs, open('scratch/1129query_openai.pkl', 'wb'))

print(query_count)
        





