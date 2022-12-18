import sys
sys.path.append('./')

from gadgets.util import gpt3wrapper

short_ind_prompt_template = open('models/templates/gpt3_ind_verifier_short.txt').read()
def ind_verify_cheapest(hypothesis, text):
    prompt = short_ind_prompt_template.format(hypothesis=hypothesis, text=text)
    engine_name = 'text-davinci-002'
    query_args = {
        'engine': engine_name,
        'prompt': prompt,
        'temperature': 0.,
        'max_tokens': 3,
        'top_p': 1,
        'n': 1
    }
    result = gpt3wrapper(tag='ind_verifier', **query_args)
    text = result['choices'][0]['text']
    if 'yes' in text.lower():
        return True
    return False

short_cmp_prompt_template = open('models/templates/gpt3_cmp_verifier_short.txt').read()
def cmp_verify_cheapest(hypothesis, text_A, text_B):
    engine_name = 'text-davinci-002'
    prompt = short_cmp_prompt_template.format(hypothesis=hypothesis, text_A=text_A, text_B=text_B)
    query_args = {
        'engine': engine_name,
        'prompt': prompt,
        'temperature': 0.,
        'max_tokens': 3,
        'top_p': 1,
        'n': 1
    }
    result1 = gpt3wrapper(tag='cmp_verifier', **query_args)['choices'][0]['text'].lower()
    if 'unsure' in result1:
        return 'unsure'
    
    ans1 = 'A' if 'a' in result1 else 'B'

    prompt = short_cmp_prompt_template.format(hypothesis=hypothesis, text_A=text_B, text_B=text_A)
    query_args = {
        'engine': engine_name,
        'prompt': prompt,
        'temperature': 0.,
        'max_tokens': 3,
        'top_p': 1,
        'n': 1
    }
    result2 = gpt3wrapper(tag='cmp_verifier', **query_args)['choices'][0]['text'].lower()
    if 'unsure' in result2:
        return 'unsure'
    ans2 = 'A' if 'a' in result2 else 'B'

    if ans1 == ans2:
        return 'unsure'
    
    return ans1


if __name__ == '__main__':
    # print(ind_verify_cheapest(hypothesis= 'describes an animal.', text= 'this cat is black'))
    # print(ind_verify_cheapest(hypothesis= 'describes an animal.', text= 'this car is black'))

    print(cmp_verify_cheapest(hypothesis= 'describes an animal.', text_A= 'this cat is black', text_B= 'this car is black'))
    print(cmp_verify_cheapest(hypothesis= 'is longer', text_A= 'this cat is black and I really like it', text_B= 'sure'))
    print(cmp_verify_cheapest(hypothesis= 'is longer', text_A= 'sure', text_B= 'this cat is black and I really like it'))
    print(cmp_verify_cheapest(hypothesis= 'is longer', text_A= 'sure', text_B= 'sure'))