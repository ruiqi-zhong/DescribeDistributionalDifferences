This repo contains the code and data for our paper: 

**Describing Differences between Text Distributions with Natural Language**

Ruiqi Zhong, Charlie Snell, Dan Klein, Jacob Steinhardt

```proposer_sec_3_1.py``` contains an example of how to run the proposer described in Section 3.1

```verifier_sec_3_2.py``` contains an example of how to run the verifier described in Section 3.2

```finetune_sec_3_3``` contains the fine-tuning data for both the proposer and the verifier, and more detailed information about the configuration of fine-tuning. Our fine-tuned verifier can be downloaded from the Huggingface Model Repo "ruiqi-zhong/verifier11b"

```benchmark_sec_4``` contains the 54 binary classification tasks we used to benchmark our system and our manual evaluation of the descriptions by the system. See load_benchmark.py and table_1.py for more detail.

If you have any questions, feel free to email me at ruiqi-zhong@berkeley.edu!
