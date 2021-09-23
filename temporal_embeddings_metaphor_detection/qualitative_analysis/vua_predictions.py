from util import get_num_lines, get_pos2idx_idx2pos, index_sequence, get_vocab, embed_indexed_sequence, \
    get_word2idx_idx2word, get_embedding_matrix, get_embedding_matrix2, write_predictions, get_performance_VUAverb_val
from util import TextDatasetWithGloveElmoSuffix as TextDataset
from util import evaluate
from model_rnn_hg import RNNSequenceModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import pandas as pd
import ast
import csv
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import random

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
1.2 TroFi
get raw dataset as a list:
  Each element is a triple:
    a sentence: string
    a index: int: idx of the focus verb
    a label: int 1 or 0
'''

df = pd.read_csv('../data/VUAsequence/VUA_seq_formatted_test.csv', encoding='latin-1')
pos = df["pos_seq"].iloc[0:2690]
frasi = df["sentence"].iloc[0:2690]
genre = df["genre"].iloc[0:2690]


raw_trofi = []

pos_set = set()
raw_trofi = []
with open('../data/VUAsequence/VUA_seq_formatted_test.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        sentence = line[2]
        pos_seq = ast.literal_eval(line[4])
        label_seq = ast.literal_eval(line[3])
        assert (len(pos_seq) == len(label_seq))
        assert (len(line[2].split()) == len(pos_seq))
        raw_trofi.append([sentence.strip(), label_seq, pos_seq])
        pos_set.update(pos_seq)


print('TroFi dataset division: ', len(raw_trofi))

"""
2. Data preparation
"""
'''
2. 1
get vocabulary and glove embeddings in raw dataset 
'''
# vocab is a set of words
vocab = get_vocab(raw_trofi)
# two dictionaries. <PAD>: 0, <UNK>: 1
word2idx, idx2word = get_word2idx_idx2word(vocab)
# glove_embeddings a nn.Embeddings
glove_embeddings = get_embedding_matrix(word2idx, idx2word, normalization=False)
# elmo_embeddings
# set elmos_trofi=None to exclude elmo vectors. Also need to change the embedding_dim in later model initialization
elmos_trofi = h5py.File('../elmo/VUA_test.hdf5', 'r')

bert_trofi = None

suffix_embeddings = None
#suffix_embeddings = nn.Embedding(15, 50)

'''
2. 2
embed the datasets
'''

#random.seed(0)
#random.shuffle(raw_trofi)
#np.random.seed(0)
#torch.manual_seed(0)
#torch.cuda.manual_seed(0)
#torch.backends.cudnn.deterministic = True

embedded_trofi = [[embed_indexed_sequence(example[0], example[2], word2idx,
                                      glove_embeddings, elmos_trofi, bert_trofi, suffix_embeddings),
                       example[2], example[1]]
                      for example in raw_trofi]

'''
2. 3 10-fold cross validation
'''
# separate the embedded_sentences and labels into 2 list, in order to pass into the TextDataset as argument
sentences = [example[0] for example in embedded_trofi]
poss = [example[1] for example in embedded_trofi]
labels = [example[2] for example in embedded_trofi]
# ten_folds is a list of 10 tuples, each tuple is (list_of_embedded_sentences, list_of_corresponding_labels)
ten_folds = []
fold_size = int(2695 / 10)
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
    training_dataset_trofi = TextDataset(training_sentences, training_poss, training_labels)
    val_dataset_trofi = TextDataset(ten_folds[i][0], ten_folds[i][1], ten_folds[i][2])


    model = torch.load('vua_model_1549.pth')
    

    # Data-related hyperparameters
    batch_size = 2
    # Set up a DataLoader for the training, validation, and test dataset
    train_dataloader_trofi = DataLoader(dataset=training_dataset_trofi, batch_size=batch_size, shuffle=True,
                                        collate_fn=TextDataset.collate_fn)
    val_dataloader_trofi = DataLoader(dataset=val_dataset_trofi, batch_size=batch_size, shuffle=False,
                                      collate_fn=TextDataset.collate_fn)

    
    #sentence_texts = []
    #predictions = []
    #real_values = []  


    for (eval_pos_seqs, eval_text, eval_lengths, eval_labels) in val_dataloader_trofi:
        
            eval_text = Variable(eval_text)
            eval_lengths = Variable(eval_lengths)
            eval_labels = Variable(eval_labels)

            if using_GPU:
                eval_text = eval_text.cuda()
                eval_lengths = eval_lengths.cuda()
                eval_labels = eval_labels.cuda()

            # predicted shape: (batch_size, seq_len, 2)
            predicted = model(eval_text, eval_lengths)
            # Calculate loss for this test batch. This is averaged, so multiply
            # by the number of examples in batch to get a total.
            #total_eval_loss += criterion(predicted.view(-1, 2), eval_labels.view(-1))
            # get 0 or 1 predictions
            # predicted_labels: (batch_size, seq_len)
            _, predicted_labels = torch.max(predicted.data, 2)

            eval_text = eval_text.detach().cpu().numpy()
            predicted_labels = predicted_labels.detach().cpu().numpy()
            eval_labels = eval_labels.detach().cpu().numpy()

            #sentence_texts.extend(eval_text)
            predictions.extend(predicted_labels)
            real_values.extend(eval_labels)

my_submission_preds = pd.DataFrame()
my_submission_preds['sentence'] = frasi
my_submission_preds['preds'] = predictions
my_submission_preds['true'] = real_values
my_submission_preds['pos_seq'] = pos
my_submission_preds["genre"] = genre
my_submission_preds.to_csv("vua_preds.csv", index=False)

#my_submission_orig = pd.DataFrame()
#my_submission_orig['sentence'] = frasi
#my_submission_orig['true'] = real_values
#my_submission_orig.to_csv("trofi_orig.csv", index=False)