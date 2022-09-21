from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle as pkl
from ft_infer import train
import time


if __name__ == '__main__':
    model_name = 'unifiedqa-t5-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = '<pad>'
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    task_name = 'filter'
    data_f_name = 'filter_finetune.pkl'

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model_tokenizer = (model, tokenizer)

    train_prompt_completion, test_prompt_completion = pkl.load(open(data_f_name, 'rb'))
    test_prompts = [d['prompt'] for d in test_prompt_completion]

    save_name = str(time.time()) + task_name

