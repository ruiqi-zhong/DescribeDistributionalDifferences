from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
import random
from tqdm import trange
from torch.optim import AdamW
from transformers.optimization import Adafactor, AdafactorSchedule
import json
import numpy as np
from collections import OrderedDict
import pickle as pkl
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device %s' % device)


def get_loss(model_tokenizer, ds, max_source_length=512, max_target_length=512):
    model, tokenizer = model_tokenizer

    if 'ConditionalGeneration' not in str(type(model)):
        assert tokenizer.padding_side == 'left'
        all_seqs = [d['prompt'] + d['completion'] for d in ds]

        encoding = tokenizer(
            all_seqs,
            padding="longest",
            max_length=max_source_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        lens = torch.tensor([len(tokenizer(d['prompt'])['input_ids']) for d in ds]).to(device)
        target_lens = torch.tensor([len(tokenizer(d['completion'])['input_ids']) for d in ds]).to(device)
        max_len = input_ids.shape[1]
        threshold = max_len - target_lens
        mask = (torch.arange(max_len).to(device).expand(len(lens), max_len) < threshold.unsqueeze(1)) * 1
        labels = input_ids * (1 - mask) + mask * (-100)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

        # debugging code to check whether masking is correct

        # print(attention_mask)     
        # print(labels)
        # print(input_ids)
        # print(tokenizer(ds[0]['prompt']))
        # print(tokenizer(ds[1]['prompt']))
        # print(tokenizer(ds[0]['completion']))
        # print(tokenizer(ds[1]['completion']))
        # exit(0)

        return loss

    else:
        input_sequences, output_sequences = [[d[k] for d in ds] for k in ['prompt', 'completion']]
        encoding = tokenizer(
            input_sequences,
            padding="longest",
            max_length=max_source_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        target_encoding = tokenizer(
            output_sequences, padding="longest", max_length=max_target_length, truncation=True
        )
        labels = target_encoding.input_ids
        labels = torch.tensor(labels).to(device)
        labels[labels == tokenizer.pad_token_id] = -100
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        return loss


num_heads_device_count_maps = {
    (24, 2): {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        1: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    },
    (32, 2): {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        1: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    }
}


def train(model_tokenizer, train_dicts, pred_prompts=None, save_name='debug', 
          save_every=2000, num_updates=10000, bsize=16, accum=4, max_source_length=512, 
          max_target_length=512, temperature=.8, n_samples=8, save_score_tok_idx=None, 
          eval_initial=False, save_model=True, stop_strs=None, map_to_device=True, optimizer_scheduler=None):
    save_path = 'models/' + save_name

    model, tokenizer = model_tokenizer
    if map_to_device:
        if torch.cuda.device_count() != 1:
            device_map = num_heads_device_count_maps[(model.config.num_heads, torch.cuda.device_count())]
            model.parallelize(device_map)
        elif torch.cuda.device_count() == 1:
            model = model.to(device)
    random.shuffle(train_dicts)

    is_t5family = 't5' in str(type(model)).lower()
    if optimizer_scheduler is None:
        if is_t5family:
            optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
            scheduler = AdafactorSchedule(optimizer)
        else:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for name, p in model.named_parameters() if not any(nd in name for nd in no_decay)],
                'weight_decay': 0.01},
                {'params': [p for name, p in model.named_parameters() if any(nd in name for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)
            scheduler = get_linear_schedule_with_warmup(optimizer, 3000, num_updates)
    else:
        optimizer, scheduler = optimizer_scheduler

    pred_dict = {}
    pbar = trange(num_updates * accum)
    udpate_count = 0

    for i in pbar:
        ds = [train_dicts[j % len(train_dicts)] for j in range(i * bsize, (i + 1) * bsize)]
        loss = get_loss(model_tokenizer, ds, max_source_length=max_source_length, max_target_length=max_target_length)
        loss.backward()
        pbar.set_description('loss %f' % loss.item())

        if (i + 1) % accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            udpate_count += 1
        
        if i % accum == 0 and udpate_count % save_every == 0 and (udpate_count != 0 or eval_initial):
            if udpate_count != 0 and save_model:
                model.save_pretrained(save_path + 'step%d.ckpt' % udpate_count)
            if pred_prompts is not None:
                with torch.no_grad():
                    model.eval()
                    result = sample_batched(model_tokenizer, pred_prompts, max_source_length=max_source_length, max_target_length=max_target_length, log_path='preds/' + save_name + '_decode_log.jsonl', temperature=temperature, n=n_samples, save_score_tok_idx=save_score_tok_idx, bsize=bsize * 2, stop_strs=stop_strs, verbose=True)
                    pred_dict[udpate_count] = result
                pkl.dump(result, open('preds/' + save_name + 'step%d_pred.pkl' % udpate_count, 'wb'))
            model.train()
    return pred_dict


def truncate_string(s, stop_strs):
    for stop_str in stop_strs:
        if stop_str in s:
            s = s[:s.index(stop_str)]
    return s

def remove_prefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix):]
    return s


def sample_batched(
    model_tokenizer, prompts, temperature=0.8, n=8, bsize=8, 
    max_source_length=512, max_target_length=512, 
    log_path='tmp.json', save_score_tok_idx=None, verbose=False, stop_strs=None):
    model, tokenizer = model_tokenizer
    prompts_inflated = []
    
    if stop_strs is None:
        stop_strs = []
    for prompt in prompts:
        prompts_inflated.extend([prompt] * n)
    all_completions, all_first_scores = [], []

    if save_score_tok_idx is None:
        save_score_tok_idx = []

    with torch.no_grad():
        model.eval()
        num_batches = (len(prompts_inflated) - 1) // bsize + 1
        if verbose:
            pbar = trange(num_batches)
            pbar.set_description('inference')
        else:
            pbar = range(num_batches)
        with open(log_path, 'w') as out_file:
            if verbose:
                print('writing to path %s' % log_path)
            for batch_idx in pbar:
                input_prompts = prompts_inflated[batch_idx * bsize: (batch_idx + 1) * bsize]
                inputs = tokenizer(input_prompts, 
                                   return_tensors="pt",
                                   padding="longest",
                                   max_length=max_source_length,
                                   truncation=True,
                                  ).to(device)
                generation_result = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_target_length,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                decoded_strs = tokenizer.batch_decode(
                    generation_result.sequences,
                    skip_special_tokens=True,
                    return_dict_in_generate=True,
                    clean_up_tokenization_spaces=False
                )

                for s in decoded_strs:
                    out_file.write(json.dumps(s) + '\n')
                out_file.flush()
                all_completions.extend(decoded_strs)
                all_first_scores.extend(generation_result.scores[0].detach().cpu().numpy().tolist())
    return_dict = OrderedDict()
    for i, prompt in enumerate(prompts):
        return_dict[prompt] = [
            {'text': 
            truncate_string(remove_prefix(all_completions[idx].replace(tokenizer.pad_token, ''), prompt), stop_strs), 'scores': [all_first_scores[idx][j] for j in save_score_tok_idx]} 
            for idx in range(i * n, (i + 1) * n)
        ]
    return return_dict


if __name__ == '__main__':

    # ds = [{'prompt': ' '.join(str(i)), 'completion': ' '.join(str(i + 1))} for i in range(100000)]
    ds = [{'prompt': 'a -> a;\n1 0 1 -> 1 0 1;\n' + ' '.join(str(i)) + ' -> ', 'completion': ' '.join(str(i)) + ';'} for i in range(100000)]
    print(ds[0])
    # test_case_name = 'codegen'
    test_case_name = 'incoder'
    if test_case_name == 't5':
        model_name = 't5-small'
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    if test_case_name == 'codegen':
        model_name = 'Salesforce/codegen-350M-mono'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        # if tokenizer.pad_token is None:
        #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #     model.resize_token_embeddings(len(tokenizer))
        model = AutoModelForCausalLM.from_pretrained(model_name)
    if test_case_name == 'incoder':
        model_name = 'facebook/incoder-1B'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = '<pad>'
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model_tokenizer = (model, tokenizer)

    pred_dict = train(model_tokenizer, ds, ['1 9 1', '1 9 2'], save_every=1000, stop_strs=[';'], num_updates=1000, max_target_length=20)
    print(sample_batched(model_tokenizer, ['1 9 1', '1 9 2'], n=1, temperature=0.01, stop_strs=[';']))
    
    print(pred_dict)


