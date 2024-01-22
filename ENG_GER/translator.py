import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
import random
import os

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import pickle

BATCH_SIZE = 64
EMBEDDING_DIM = 50
UNITS = 50
NUM_EPOCHS = 20
max_lines = 180000

def preprocess(sent, exclude, sp_tokens=False):
    sent = sent.lower()
    sent = re.sub("'", '', sent)
    sent = ''.join(ch for ch in sent if ch not in exclude)
    sent = sent.strip()
    sent = re.sub(" +", " ", sent)
    if sp_tokens:
        sent = '<start> ' + sent + ' <end>'
    
    return sent

# padding
def pad_sequences(x, max_len):
    padded = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_len, padding='post', truncating='post')
    return padded


def detokenize(token,tokenizer):
    # token = token.numpy()
    res = ""
    for i in token:
        if i!=0:
            if(tokenizer.index_word[i]=="<end>"):
                break
            if(tokenizer.index_word[i]=="<start>"):
                continue
            res += tokenizer.index_word[i] + " "
    return res


def predict_seq2seq(encoder, decoder, src_tokens, tar_tokenizer, num_steps):
    enc_X = tf.expand_dims(src_tokens, axis=0)
    mask = tf.expand_dims(enc_X != 0, 1)

    enc_outputs, enc_state = encoder(enc_X, training=False)
    dec_state = enc_state
    dec_X = tf.expand_dims(tf.constant([tar_tokenizer.word_index['<start>']]), axis=0)
    output_seq = []
    attention_weights = []
    for _ in range(num_steps):
        Y, dec_state, att_wgts = decoder(
            dec_X, enc_outputs, dec_state, mask,training=False)
        dec_X = tf.argmax(Y, axis=2)
        pred = tf.squeeze(dec_X, axis=0)
        if pred[0].numpy() == tar_tokenizer.word_index['<end>']:
            break
        output_seq.append(pred[0].numpy())
        attention_weights.append(tf.squeeze(att_wgts, 0))
    attention_weights = tf.squeeze(tf.stack(attention_weights, axis=0), 1)
    return detokenize(output_seq, tar_tokenizer), attention_weights


def translate(lines,inp_tokenizer,tar_tokenizer,encoder,decoder,max_len=10):
    
    exclude = set(string.punctuation)
    prep_lines = []

    for i in lines:
        prep_lines.append([preprocess(i, exclude, sp_tokens=False)])
    
    input_tensors = []

    for i in prep_lines:
        inp = i[0].split(' ')
        tensor = []
        for j in inp:
            # check for oov
            if j not in inp_tokenizer.word_index.keys():
                tensor.append(inp_tokenizer.word_index['<unk>'])
            else:
                tensor.append(inp_tokenizer.word_index[j])
        input_tensors.append(tensor)
            
    for input_tensor in input_tensors:
        temp_max_len = max(max_len,len(input_tensor))
        for i in range(temp_max_len-len(input_tensor)):
            input_tensor.append(0)

    for input_tensor in input_tensors:
        input_tensor = tf.convert_to_tensor(input_tensor)

    translations = []

    for input_tensor in input_tensors:
        translation, _ = predict_seq2seq(encoder, decoder, input_tensor, tar_tokenizer,10)
        translations.append(translation)
        
    return translations

