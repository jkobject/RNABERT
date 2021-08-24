# RNABERT
This repo contains the code for our paper "Informative RNA-base embedding for functional RNA clustering and structural alignment". Please contact me at akiyama@dna.bio.keio.ac.jp for any question. Please cite this paper if you use our code or system output.

In this package, we provides resources including: source codes of the RNABERT model, pre-trained weights, prediction module.

## 1. Environment setup

Our code is written with python Python 3.6.5. Our code requires PyTorch version >= 1.4.0, biopython version >=1.76, and C++17 compatible compiler. Please follow the instructions here: https://github.com/pytorch/pytorch#installation.
Also, please make sure you have at least one NVIDIA GPU. 

#### 1.1 Install the package and other requirements

(Required)

```
git clone https://github.com//RNABERT
cd RNABERT
python setup.py install
```


## 2. Pre-train (Skip this section if you only want to make predictions)

#### 2.1 Data processing

Pre-train consists of two tasks, SFP and SAL.The SFP task trains with multiple files for each RNA family. SAL tasks use family-specific multiple alignments for training. If you want to train with your own data, see the template data at /sample/mlm/ for MLM task and /sample/sal/ for SAL task. RNABERT requires that RNA sequences be represented in fasta format. All nucleotides are represented by A, U (T), G, C. 


#### 2.2 Model Training

The MLM task specifies the percentage of nucleotides to be masked "--maskrate" and the number of mask patterns "--mag". Adjust the batch size according to the memory size of your GPU.
```
export TRAIN_FILE=sample/mlm/sample.fa
export PRE_WEIGHT= #optional
export OUTPUT_WEIGHT=/path/to/output/weight

python MLM_SFP.py 
    --pretraining ${PRE_WEIGHT} \
    --outputweight ${OUTPUT_WEIGHT} \
    --data_mlm ${TRAIN_FILE} \
    --epoch 10 \
    --batch 40 \
    --mag 3 \
    --maskrate 0.2 \
```
The SAL task takes multiple alignments per family as input, and "--mag" can be used to specify how many pairwise alignments should be generated for a single sequence.
```
export TRAIN_FILE=sample/sal/sample.afa.txt
export PRE_WEIGHT= #optional
export OUTPUT_WEIGHT=/path/to/output/weight

python MLM_SFP.py 
    --pretraining ${PRE_WEIGHT} \
    --outputweight ${OUTPUT_WEIGHT} \
    --data_mul ${TRAIN_FILE} \
    --epoch 10 \
    --batch 40 \
    --mag 5 \
```



#### 2.3 Download pre-trained DNABERT

[RNABERT](https://drive.google.com/file/d/1FqE_c0X6OA75AzYI8ChpB7WH8Oq6TRJS/view?usp=sharing)

Download the pre-trained model in to a directory. 
This model has been created using a partial Rfam dataset. Trained model using the full Rfam seed alignment dataset will be available soon.


## 3. Prediction

After the model is fine-tuned, we can get predictions by running

```
export PRED_FILE=sample/sal/sample.fa
export PRE_WEIGHT=/path/to/pretrained/weight

python MLM_SFP.py 
    --pretraining ${PRE_WEIGHT} \
    --data_alignment ${PRED_FILE} \
    --batch 40 \
```