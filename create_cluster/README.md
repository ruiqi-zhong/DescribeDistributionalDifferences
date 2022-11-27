
# Embed

```embed.py``` embed a list of strings into a embeddings
Example command:
```
python3 embed.py --model_name roberta-base --subpart_name wikitext-103-raw-v1
```

- ```model_name``` is the name of the pre-trained model. Currently it supports the BERT, RoBERTa, and T5 family.
- ```subpart_name``` is the name of the dataset you are embedding. If you want to add more datasets, create a new "entry" at the line of ```elif subpart_name ==``` and replace data with a list of strings you want to embed

after you run this command it will automatically create a folder in the ```embeddings/``` directory, and save the corresponding text samples as ```.json``` and the embeddings as the ```.npy``` file. 
I wrote partial results to disk just in case the process breaks, and in the worst case we still have something to use. In many cases I suspect that 100K datapoints suffice. 

After running the command above, you will see the following file structure
```
embeddings/
├── model_name=roberta-base_subpart=wikitext-103-raw-v1
│   ├── 100032.json
│   ├── 100032.npy
│   ├── 10016.json
│   ├── 10016.npy
│   ├── 102016.json
│   ├── 102016.npy
```

# Cluster

After dumping the embedding in the above directory, you can run the command

```python3 create_cluster_from_embeddings.py --embed_dir embeddings/model_name\=roberta-base_subpart\=wikitext-103-raw-v1/ --subset_size 3000```

- ```embed_dir``` is the folder where you dumped the embedding just now
- ```subset_size``` means the number of datapoints you want to use to learn the cluster. Larger subset size probably gives better quality data but it might take really a long time.

There are two other optional arguments
- ```cluster_method```, which defaults to kmeans. You can also use gmm (Gaussian mixture), but it's extremeley slow
- ```k```, the number of clusters to produce. default to 128

After you run the above command, it will produce the following file structure

clusters/
├── model_name=roberta-base_subpart=wikitext-103-raw-v1_k=128_method=kmeans_subset=3000
│   ├── all_texts.json
│   ├── center_pairwise_distance.npy
│   ├── centers.npy
│   ├── cluster_idxes.npy
│   └── data_center_distance.npy
└── placeholder.txt

Say we have n text samples and k clusters
- ```all_texts.json``` is a list of strings, length n
- ```data_center_distance.npy``` is the distance between each datapoint to each cluster center. Shape n X k.
- ```center_pairwise_distance.npy``` pairwise distance between centers. Shape k X k.
- ```cluster_idxes.npy``` an integer index of which cluster each datapoint belongs to. Shape (k, )