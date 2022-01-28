import openai
import random

# This is the implementation of our proposer as described in Section 3.1 .

# Enter you GPT-3 key here
openai.api_key = ''

# a list of forbidden output tokens, such as "while", "group", etc.
discouraged_toks = [4514, 8094, 33, 40798, 392, 273, 14, 11, 981, 4514, 8094, 1448, 33, 347, 1884, 40798, 290, 392, 273, 393, 14, 1220, 837, 11]

# the bias added to the decoding tokens; effectively forbiding all discouraged_toks
logit_bias = {i: -100 for i in discouraged_toks}

# number of completions you want
n = 1 

# the temperature for decoding
temperature = 0.7

# the GPT-3 inference engine
# davinci-instruct-beta will give you the best description if you do not have fine-tuning access
engine = 'davinci-instruct-beta'

# the stop tokens
stop_tokens = ["\n", '.']

# the maximum number of decoding token
max_tokens = 50

# only consider tokens that are included in the top_p probability
top_p = 1

# the maximum number of characters for each sample
max_single_length = 50

# number of sample in prompt
K = 5


# trim the length of each sample by character length;
# although a better way to do this is to use the GPT-2 tokenizer
def normalize(x):
    x = x.strip()
    if len(x) > max_single_length:
        x = x[:max_single_length] + ' ...'
    return x
        

def sample_sentences(xs, k, group_id):
    random.shuffle(xs)
    return '\n'.join(['Group %s: %s' % (group_id, normalize(x)) for x in xs[:k]])


# This function maps from a list of positive examples (D_1) and a list of negative examples (D_0) to a prompt
# we will query GPT-3
def create_prompt_from_pos_neg_samples(positive_samples, negative_samples, k=K):
    group_A_text = sample_sentences(positive_samples, k=k, group_id='A')
    group_B_text = sample_sentences(negative_samples, k=k, group_id='B')

    prompt = group_A_text + '\n\n' + group_B_text + '\n\n'
    prompt += 'Compared to sentences from Group B, each sentence from Group A'
    return prompt


positive_samples = [
    "How much in miles is a ten K run?",
    "When is the Jimmy Buffett concert coming to the E center in Camden NJ?", 
    "What chapter of Gone with the Wind has Rhett Butler leaving Scarlett O 'Hara?",
    "What is the latitude and longitude of El Paso, Texas?", 
    "How old was Elvis Presley when he died?"
]

negative_samples = [
    "What is the daily requirement of folic acid for an expectant mother?", 
    "What type of bridge is the Golden Gate Bridge?", 
    "Where do the Blackhawks maintain their operations?", 
    "What attorneys work for The Center for the Defense of Free Enterprise?", 
    "What college football team did Knute Rockne build into a power?"
]

prompt = create_prompt_from_pos_neg_samples(positive_samples, negative_samples)
print('======== the proposer prompt is ======== ')
print(prompt)

# a dictionary of hyperparameters
hyperparameters = {
    'max_tokens': max_tokens,
    'top_p': top_p,
    'temperature': temperature,
    'engine': engine
}

# query GPT-3 to propose hypotheses
response = openai.Completion.create(
    prompt=prompt,
    stop=stop_tokens,
    logit_bias=logit_bias,
    **hyperparameters
)

# for the above example, our fine-tuned GPT-3 Davinci (175B) model will sometimes respond 
# "is a question requiring a numerical answer", which describes the difference
# however, more often it generates plausible but incorrect ones, e.g., is more difficult to understand
# so we need to use a verifier to filter out incorrect ones
print(response)

