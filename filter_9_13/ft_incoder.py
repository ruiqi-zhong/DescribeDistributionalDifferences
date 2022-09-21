from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle as pkl
from ft_infer import train
import time
from sklearn.model_selection import train_test_split

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

    prompt_completion = pkl.load(open(data_f_name, 'rb'))
    
    train_prompt_completion, test_prompt_completion = train_test_split(prompt_completion, test_size=.2, random_state=0)
    test_prompts = [d['prompt'] for d in test_prompt_completion]

    save_name = str(time.time()) + task_name

    train(
        model_tokenizer, train_prompt_completion, test_prompts, 
        save_name=save_name, max_source_length=2048, max_target_length=200, 
        bsize=4, accum=8, num_updates=2002, save_every=500, eval_initial=False, 
        n_samples=1, temperature=0.01, save_model=True, stop_strs=[';']
    )