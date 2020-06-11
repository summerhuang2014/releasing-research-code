## Code for paper "An information-theoretic framework for learning models of instance-independent label noise".

### 1. Gather LID sequences for the given dataset
#### 1.1 The given dataset
Some noisy dataset labels with symmetric noise transition matrices are provided in the directory "./given_datasets/intact/". Its title indicates its underlying clean dataset (CIFAR-10 or MNIST), the amount of noise in it and the random seed used to generate this noisy dataset.
An example: `"./given_datasets/intact/cifar-10_train_labels_seed_42_add_20.0.npy"` is an intact CIFAR-10 dataset with 20% symmetric noise rate (generated with random seed 42).

#### 1.2 Gather LID sequences

An example: 
`CUDA_VISIBLE_DEVICES=0 python main.py -l ./given_datasets/intact/mnist_train_labels_seed_42_add_20.0.npy -d mnist -b 128`

`-l`: location of the given noisy dataset
`-d`: dataset in ['mnist', 'cifar-10']
`-b`: training batch size number. 

It should be noted that to estimate the noise transition matrix of a given dataset, we used minimum 10 random seeds with a fixed set of noise vectors (the fixed uniform alpha-set), which requires sufficient GPU computational power. The bottom part of the code `main.py` can be modified to reduce the number of random seeds used for a smaller trial.

#### Expected result
LID sequences would be generated under the directory "./lid". 
Note that different devices could generate different LID sequences even with the same seed. However, on the same device, with the same random seed and running with complete number of epochs, the LID sequences can be replicated with our code. It is recommended to gather the LID sequences on the same device.


### 2. train PU models

#### 2.1 consolidate LID sequences
`python consolidate_LID.py`
This consolidates the LID sequences of the 10 seeds with the alpha-set into the directory "./LID_init/" as csv files. Modify the dataset accordingly.

#### 2.2 initial training of the PU models
`python initialization_train_PU.py`
This trains the PU models on the LID sequences for each triple (from the 10 random seeds with the alpha-set, if not modified). Modify the dataset accordingly. A csv file will be generated: `10_seeds_init.csv`.

#### 2.3 reconsolidate the LID sequences for the triples with refined recall
Select those triples with recall above 0.9 by `10_seeds_init.csv`. According to the number of non-zero votes before noise rate 86%, further refine the range of recalls, then select the triples with the refined recalls (detailed in the supplementary material).
Reconsolidate the LID sequences according to the refined recall. The correspondence between recall and the highest noise rate in the alpha-set is listed below. For example, if recall is 1, the highest alpha noise rate to use is 86.4%. If 0.90, 88.6%. 
Modify the recall in the code `reconsolidate_LID.py` accordingly then `python reconsolidate_LID.py` to reconsolidate the LID sequences.

| recall         | the highest alpha noise rate to use |
|       1        |                86.4%                |
| -------------- |-------------------------------------|
|       0.98     |                88.0%                |
| -------------- |-------------------------------------|
|       0.96     |                88.2%                |
| -------------- |-------------------------------------|
|       0.94     |                88.4%                |
| -------------- |-------------------------------------|
|       0.92     |                88.5%                |
| -------------- |-------------------------------------|
|       0.90     |                88.6%                |

#### 2.4 fine-tune the refined triples
Modify the dataset, triples with refined recalls, their repective recall in `retrain_val_PU.py`. Then `python retrain_val_PU.py`, which produces `10_seeds_retrain.csv`.

### 3. solve system of linear equations

Use the template `template.xlsx` to estimate the matrix (it contains the estimation when use MPEIA as priors for estimation. The underlying dataset is CIFAR-10, its noise transition matrix is pairwise matrix with 80% noise rate). 
Modify the prior in the `prior` tab. 
Check `10_seeds_retrain.csv`, copy and paste the retrained triples to the `10_seeds_retrain` tab. 
Find those triples can be used to estimate the noise transition matrix, detailed in Section E.2.2 last 2 paragraphs in the supplementary material. Make individual estimate tabs.
Average the estimations by recall.
Average the estimations from each recall's estimate.
Compute KL loss against the ground-truth matrix.

### Requirements:
pytorch, numpy, scipy, sklearn
