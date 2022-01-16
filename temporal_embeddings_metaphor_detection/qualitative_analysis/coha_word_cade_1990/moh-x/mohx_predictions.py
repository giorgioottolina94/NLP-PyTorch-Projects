from util import get_num_lines, get_pos2idx_idx2pos, index_sequence, get_vocab, embed_indexed_sequence, \
    get_word2idx_idx2word, get_embedding_matrix, get_embedding_matrix2, write_predictions, get_performance_VUAverb_val
from util import TextDatasetWithGloveElmoSuffix as TextDataset
from util import evaluate
from util import save_checkpoint
from model_rnn_hg import RNNSequenceModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os.path

import pandas as pd
import csv
import h5py
import pickle
import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Path to saved model weights(as hdf5)
#resume_weights = "/model/checkpoint.pth.tar"

# Modified based on Gao Ge https://github.com/gao-g/metaphor-in-context

print("PyTorch version:")
print(torch.__version__)
print("GPU Detected:")
print(torch.cuda.is_available())
using_GPU = True

"""
1. Data pre-processing
"""

'''
1.3 MOH-X
get raw dataset as a list:
  Each element is a triple:
    a sentence: string
    a index: int: idx of the focus verb
    a label: int 1 or 0
'''

df = pd.read_csv('../../../data/MOH-X/MOH-X_formatted_svo_cleaned.csv')
nouns = df["arg1"].iloc[0:640]
verbs = df["verb"].iloc[0:640]
frasi = df["sentence"].iloc[0:640]



raw_mohx = []

with open('../../../data/MOH-X/MOH-X_formatted_svo_cleaned.csv') as f:
    # arg1  	arg2	verb	sentence	verb_idx	label
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        sentence = line[3]
        label_seq = [0] * len(sentence.split())
        pos_seq = [0] * len(label_seq)
        verb_idx = int(line[4])
        verb_label = int(line[5])
        label_seq[verb_idx] = verb_label
        pos_seq[verb_idx] = 1   # idx2pos = {0: 'words that are not focus verbs', 1: 'focus verb'}
        raw_mohx.append([sentence.strip(), label_seq, pos_seq])
        



print('MOH-X dataset division: ', len(raw_mohx))

#random.seed(0)
#np.random.seed(0)
#torch.manual_seed(0)
#torch.cuda.manual_seed(0)
#torch.backends.cudnn.deterministic = True
#random.shuffle(raw_mohx)
"""
2. Data preparation
"""
'''
2. 1
get vocabulary and glove embeddings in raw dataset 
'''
# vocab is a set of words
vocab = get_vocab(raw_mohx)
# two dictionaries. <PAD>: 0, <UNK>: 1
word2idx, idx2word = get_word2idx_idx2word(vocab)
# glove_embeddings a nn.Embeddings
#glove_embeddings = get_embedding_matrix2(word2idx, idx2word, normalization=False)
hist_embeddings = get_embedding_matrix(word2idx, idx2word, normalization=False)
# elmo_embeddings
# set elmos_mohx=None to exclude elmo vectors. Also need to change the embedding_dim in later model initialization
elmos_mohx = h5py.File('../../../elmo/MOH-X_cleaned.hdf5', 'r')

bert_mohx = None

suffix_embeddings = None
#suffix_embeddings = nn.Embedding(15, 50)

'''
2. 2
embed the datasets
'''

# second argument is the post sequence, which we don't need
embedded_mohx = [[embed_indexed_sequence(example[0], example[2], word2idx,
                                      hist_embeddings, elmos_mohx, bert_mohx, suffix_embeddings),
                       example[2], example[1]]
                      for example in raw_mohx]

#100 times 10-fold cross validation
#for valid in range(100):

'''
2. 3 10-fold cross validation
'''
# separate the embedded_sentences and labels into 2 list, in order to pass into the TextDataset as argument
sentences = [example[0] for example in embedded_mohx]
poss = [example[1] for example in embedded_mohx]
labels = [example[2] for example in embedded_mohx]
# ten_folds is a list of 10 tuples, each tuple is (list_of_embedded_sentences, list_of_corresponding_labels)
ten_folds = []
fold_size = int(647 / 10)
for i in range(10):
    ten_folds.append((sentences[i * fold_size:(i + 1) * fold_size],
                      poss[i * fold_size:(i + 1) * fold_size],
                      labels[i * fold_size:(i + 1) * fold_size]))

idx2pos = {0: 'words that are not focus verbs', 1: 'focus verb'}



# dataloader

predictions = []
real_values = []

for i in range(10):
    '''
    2. 3
    set up Dataloader for batching
    '''
    training_sentences = []
    training_labels = []
    training_poss = []
    for j in range(10):
        if j != i:
            training_sentences.extend(ten_folds[j][0])
            training_poss.extend(ten_folds[j][1])
            training_labels.extend(ten_folds[j][2])
    training_dataset_mohx = TextDataset(training_sentences, training_poss, training_labels)
    val_dataset_mohx = TextDataset(ten_folds[i][0], ten_folds[i][1], ten_folds[i][2])


    model = torch.load('mohx_model_{}.pth'.format(i))
    

    batch_size = 2

    # Set up a DataLoader for the training, validation, and test dataset
    train_dataloader_mohx = DataLoader(dataset=training_dataset_mohx, batch_size=batch_size, shuffle=True,
                                        collate_fn=TextDataset.collate_fn)
    val_dataloader_mohx = DataLoader(dataset=val_dataset_mohx, batch_size=100, shuffle=False,
                                        collate_fn=TextDataset.collate_fn)

    #sentence_texts = []
    #predictions = []
    #real_values = []                                    

    for (eval_pos_seqs, eval_text, eval_lengths, eval_labels) in val_dataloader_mohx:
        
            
            eval_text = Variable(eval_text)
            eval_lengths = Variable(eval_lengths)
            eval_labels = Variable(eval_labels)

            if using_GPU:
                
                eval_text = eval_text.cuda()
                eval_lengths = eval_lengths.cuda()
                eval_labels = eval_labels.cuda()

            #sentence_texts = []
            #predictions = []
            #real_values = []


            # predicted shape: (batch_size, seq_len, 2)
            predicted = model(eval_text, eval_lengths)
            # Calculate loss for this test batch. This is averaged, so multiply
            # by the number of examples in batch to get a total.
            #total_eval_loss += criterion(predicted.view(-1, 2), eval_labels.view(-1))
            # get 0 or 1 predictions
            # predicted_labels: (batch_size, seq_len)
            _, predicted_labels = torch.max(predicted.data, 2)

            #eval_text = eval_text.detach().cpu().numpy()
            predicted_labels = predicted_labels.detach().cpu().numpy()
            eval_labels = eval_labels.detach().cpu().numpy()
            #eval_pos_seqs = eval_pos_seqs.numpy()

            #sentence_texts.extend(eval_text)
            predictions.extend(predicted_labels)
            real_values.extend(eval_labels)
            
            #print(len(predictions))
            #print(len(real_values))
            

my_submission_preds = pd.DataFrame()
my_submission_preds["nouns"] = nouns
my_submission_preds["verbs"] = verbs
my_submission_preds['sentences'] = frasi
my_submission_preds['preds'] = predictions
my_submission_preds['true'] = real_values
my_submission_preds.to_csv("mohx_preds.csv", index=False)

#my_submission_orig = pd.DataFrame()
#my_submission_orig['sentence'] = frasi
#my_submission_orig['true'] = real_values
#my_submission_orig.to_csv("mohx_orig.csv", index=False)
