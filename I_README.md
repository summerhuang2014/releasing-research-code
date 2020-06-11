## Code for paper "An information-theoretic framework for learning models of instance-independent label noise".

### 1. Gather LID sequences for the given dataset
#### 1.1 The given dataset
Some noisy dataset labels with symmetric noise transition matrices are provided in the directory "./given_datasets/intact/". Its title indicates its underlying clean dataset (CIFAR-10 or MNIST), the amount of noise in it and the random seed used to generate this noisy dataset.
An example: `"./given_datasets/intact/cifar-10_train_labels_seed_42_add_20.0.npy"` is an intact CIFAR-10 dataset with 20% symmetric noise rate (generated with random seed 42).

#### 1.2 Gather LID sequences

An example: 
`CUDA_VISIBLE_DEVICES=0 python main.py -l ./given_datasets/intact/cifar-10_train_labels_seed_42_add_20.0.npy -d cifar-10 -b 128`

`-l`: location of the given noisy dataset
`-d`: dataset in ['mnist', 'cifar-10']
`-b`: training batch size number. 

It should be noted that to estimate the noise transition matrix of a given dataset, we used minimum 10 random seeds with a fixed set of noise vectors (the fixed uniform alpha-set), which requires sufficient GPU computational power. The bottom part of the code `main.py` can be modified to reduce the number of random seeds used for a smaller trial.

#### Expected result
LID sequences would be generated under the directory "./lid". 
Note that different devices could generate different LID sequences even with the same seed. However, on the same device, with the same random seed and running with complete number of epochs, the LID sequences can be replicated with our code. It is recommended to gather the LID sequences on the same device.


### Requirements:
pytorch, numpy, scipy
