import torch
import openai
import os
import time
from transformers import GPT2Tokenizer
from tqdm import tqdm
import random
import json
import nltk

tok = GPT2Tokenizer.from_pretrained('gpt2')

openai.api_key = os.environ['openai_key']

query_logs_dir = 'querylogs/'

def parallelize_across_device(model):
    num_heads = len(model.encoder.block)
    num_device = torch.cuda.device_count()
    other_device_alloc = num_heads // num_device + 1
    first_device = num_heads - (num_device - 1) * other_device_alloc
    device_map = {}
    cur = 0
    end = max(cur + first_device, 1)
    device_map[0] = list(range(cur, end))
    cur = end
    for i in range(1, num_device):
        end = min(cur + other_device_alloc, num_heads)
        device_map[i] = list(range(cur, end))
        cur += other_device_alloc
    print('device_map', device_map)
    model.parallelize(device_map)


def gpt3wrapper(max_repeat=20, tag="no-tag", **arguments):
    i = 0
    while i < max_repeat:
        try:
            start_time = time.time()
            response = openai.Completion.create(**arguments)
            end_time = time.time()
            print('completed one query in', end_time - start_time)
            with open(query_logs_dir + tag + '.json', 'a') as f:
                f.write(json.dumps(response))
            return response
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print(e)
            print('now sleeping')
            time.sleep(30)
            i += 1
    return None

def get_hyps_from_LM_output(output_str):
    return [h.replace('"', '').strip() for h in output_str.split('\n\n')[0].split('\n-')]

batch_conversion_prompt = open('models/templates/convert_cmp_hyp_batch.txt').read()
def convert_cmp_hs_one_pass(hyps):
    prompt = str(batch_conversion_prompt)
    for i, h in enumerate(hyps):
        prompt += f'Property {i + 1}: {h}\n'
    prompt += '\n'
    prompt += 'Output 1:'
    response = gpt3wrapper(prompt=prompt, max_tokens=2048, temperature=0.0, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=['\n\n'], engine='text-davinci-003')
    if response is None:
        return hyps
    raw_text = response['choices'][0]['text'].strip()
    raw_new_hyps = raw_text.split('\n')
    if len(raw_new_hyps) != len(hyps):
        print(raw_text)
        print('Expected {} hyps, got {}'.format(len(hyps), len(raw_new_hyps)))
    new_hyps, success = [], []
    for i in range(len(hyps)):
        if i == 0:
            new_hyps.append(raw_new_hyps[0])
            success.append(True)
        else:
            if i >= len(raw_new_hyps):
                new_hyps.append(hyps[i])
                success.append(False)
                continue
            h = raw_new_hyps[i]
            prefix = f'Output {i + 1}:'
            if not h.startswith(prefix):
                new_hyps.append(hyps[i])
                success.append(False)
            else:
                new_hyps.append(h[len(prefix):].strip())
                success.append(True)
    assert len(new_hyps) == len(hyps)
    return new_hyps, success


def convert_cmp_hs_(hyps):
    new_hyps = []

    count = 0
    success = []
    query_hyps = []
    for h in tqdm(hyps):
        if h[-1] == '.':
            h = h[:-1]
        word_count = len(tok.encode(h))
        if count + word_count > 256:
            print('querying ...')
            new_hyps_, success_ = convert_cmp_hs_one_pass(query_hyps)
            new_hyps.extend(new_hyps_)
            success.extend(success_)
            query_hyps = [h]
            count = word_count
        else:
            count += word_count
            query_hyps.append(h)
    if len(query_hyps) > 0:
        print('querying ...')
        new_hyps_, success_ = convert_cmp_hs_one_pass(query_hyps)
        new_hyps.extend(new_hyps_)
        success.extend(success_)
    assert len(new_hyps) == len(hyps)
    return new_hyps, success

def sus(h):
    return any(w in h.lower() for w in ['more', 'less', 'group', 'smaller', 'larger'])


def convert_cmp_hs(hyps):
    sus_counts = [sum(sus(h) for h in hyps)]
    success = [True] * len(hyps)
    new_hyps = list(hyps)

    for _ in range(5):
        reprocess_idxes = []
        for i in range(len(hyps)):
            if sus(new_hyps[i]):
                reprocess_idxes.append(i)
        random.shuffle(reprocess_idxes)
        hyps_to_reprocess = [hyps[i] for i in reprocess_idxes]
        new_hyps_to_reprocess, success_to_reprocess = convert_cmp_hs_(hyps_to_reprocess)
        for i, idx in enumerate(reprocess_idxes):
            new_hyps[idx] = new_hyps_to_reprocess[i]
            success[idx] = success_to_reprocess[i]
        sus_counts.append(sum(sus(h) for h in new_hyps))
        # if sus_counts[-1] == sus_counts[-2]:
        #     break
    print('sus count', '->'.join(str(c) for c in sus_counts))
    
    return new_hyps, [sus(h) for h in new_hyps], success


def classify_cmp(i):
    my_text = nltk.word_tokenize(i)
    pos_tags = nltk.pos_tag(my_text)
    all_tags = {t[1] for t in pos_tags}
    return any(tag in ('JJR', 'RBR') for tag in all_tags)

classify_cmp_template = open('models/templates/classify_comparison.txt').read()
def classify_cmp_(i):
    prompt = classify_cmp_template.format(input=i)
    response = gpt3wrapper(prompt=prompt, max_tokens=5, temperature=0.0, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, engine='text-davinci-002', tag='classify_cmp')
    if response is None:
        return False
    raw_text = response['choices'][0]['text'].strip()
    return 'yes' in raw_text.lower()

rm_cmp_prompt = open('models/templates/1223rm_cmp_prompt.txt').read()
def convert_cmp_to_ind(s):
    for _ in range(3):
        if not classify_cmp(s):
            break
        prompt = rm_cmp_prompt.format(input=s)
        response = gpt3wrapper(prompt=prompt, max_tokens=2048, temperature=0.0, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=['\n\n'], engine='text-davinci-002', tag='convert_cmp_to_ind')
        if response is None:
            return s
        s = response['choices'][0]['text'].strip()
    if classify_cmp(s) or 'group a' in s.lower() or 'group b' in s.lower():
        return None
    return s