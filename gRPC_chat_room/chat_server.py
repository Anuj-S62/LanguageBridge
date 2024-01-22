import sys
sys.path.append('../')

import tensorflow as tf
import string
import pickle
import numpy as np
import os
import re

import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

from ENG_GER.translator import translate
from LanguageClassifier.languageClassifier import classifier


class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, embed_dim, units, vocab_size):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embed_dim, mask_zero=True)
        self.rnn = tf.keras.layers.LSTM(
            units, return_sequences=True, return_state=True)
    
    def call(self, x):
        # x => (batch_size, max_len)
        x = self.embedding(x) # => (batch_size, s, embed_dim)
        enc_outputs = self.rnn(x)
        return enc_outputs[0], enc_outputs[1:]

class AdditiveAttention(tf.keras.layers.Layer):

    def __init__(self, units):
        super().__init__()
        self.W_q = tf.keras.layers.Dense(units, use_bias=False)
        self.W_k = tf.keras.layers.Dense(units, use_bias=False)
        self.W_v = tf.keras.layers.Dense(1, use_bias=False)
    

    def call(self, query, key, value, mask=None):
        query, key = self.W_q(query), self.W_k(key)
        # query => (batch_size, t, units)
        # key => (batch_size, s, units)

        score = self.W_v(
            tf.math.tanh(
                tf.expand_dims(query, 2) + tf.expand_dims(key, 1)
            )
        )
        score = tf.squeeze(score, -1)
        # score => (batch_size, t, s)
        
        if mask is not None:
            score = tf.where(mask, score, -1e6)
        
        attention_weights = tf.nn.softmax(score, axis=-1)
        # attention_weights => (batch_size, t, s)

        context = tf.matmul(attention_weights, value)
        # context => (batch_size, t, units)

        return context, attention_weights

class Decoder(tf.keras.layers.Layer):

    def __init__(self, embed_dim, units, vocab_size):
        super().__init__()

        # Embedding layer to convert tokens to vectors
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embed_dim, mask_zero=True)
        
        # RNN layer
        self.rnn = tf.keras.layers.LSTM(
            units, return_sequences=True, return_state=True)
        
        # Attention layer
        self.attention = AdditiveAttention(units)

        # Final layer to output logits, we can use 
        # argmax to know which output token is predicted.
        self.fc = tf.keras.layers.Dense(vocab_size)
    

    def call(self, x, enc_outputs, state, mask=None):
        x = self.embedding(x)
        # x => (batch_size, t, embed_dim)

        dec_outputs = self.rnn(x, initial_state=state)
        output = dec_outputs[0]
        state = dec_outputs[1:]
        # output   => (batch_size, t, units) 
        # state[i] => (batch_size, s, units)

        context_vector, attention_weights = self.attention(
            query=output,
            key=enc_outputs,
            value=enc_outputs,
            mask=mask
        )
        # context_vector => (batch_size, t, units)
        # attention_weights => (batch_size, t, s)

        context_rnn_output = tf.concat(
            [context_vector, output], axis=-1)
        # context_rnn_output => (batch_size, t, 2*units)

        pred = self.fc(context_rnn_output)
        # pred => (batch_size, t, vocab_size)
        
        return pred, state, attention_weights
    
def preprocessing(text):
    text = text.lower()
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

clf_model = pickle.load(open("../LanguageClassifier/classifier_model", "rb"))

encoder_loaded_ger = pickle.load(open('../ENG_GER/encoder_attention_v1.pickle', 'rb'))
decoder_loaded_ger = pickle.load(open('../ENG_GER/decoder_attention_v1.pickle', 'rb'))

X_tokenizer_loaded_ger = pickle.load(open('../ENG_GER/X_tokenizer.pickle', 'rb'))
Y_tokenizer_loaded_ger = pickle.load(open('../ENG_GER/Y_tokenizer.pickle', 'rb'))

encoder_loaded_en = pickle.load(open('../GER_ENG/encoder_attention_v1.pickle', 'rb'))
decoder_loaded_en = pickle.load(open('../GER_ENG/decoder_attention_v1.pickle', 'rb'))

X_tokenizer_loaded_en = pickle.load(open('../GER_ENG/X_tokenizer.pickle', 'rb'))
Y_tokenizer_loaded_en = pickle.load(open('../GER_ENG/Y_tokenizer.pickle', 'rb'))

# print(translate(["i am very smart"],X_tokenizer_loaded_ger,Y_tokenizer_loaded_ger,encoder=encoder_loaded_ger,decoder=decoder_loaded_ger))

nlp = spacy.load("en_core_web_sm")


def translator(text):
    # chech if text is english
    res = ""
    if classifier(text,clf_model) == "English":
        arr = translate([text],X_tokenizer_loaded_ger,Y_tokenizer_loaded_ger,encoder=encoder_loaded_ger,decoder=decoder_loaded_ger)
        res = "Original: " + text + "\n" + "Translated: " + arr[0] + "\n"

    # check if text is german
    else:
        arr = translate([text],X_tokenizer_loaded_en,Y_tokenizer_loaded_en,encoder=encoder_loaded_en,decoder=decoder_loaded_en)
        res = "Original: " + text + "\n" + "Translated: " + arr[0] + "\n"

    return res

print(translator("i am very smart"))

from concurrent import futures

import grpc
import time

import chat_pb2 as chat
import chat_pb2_grpc as rpc



class ChatServer(rpc.ChatServerServicer):  # inheriting here from the protobuf rpc file which is generated

    def __init__(self):
        # List with all the chat history
        self.chats = []

    # The stream which will be used to send new messages to clients
    def ChatStream(self, request_iterator, context):
        """
        This is a response-stream type call. This means the server can keep sending messages
        Every client opens this connection and waits for server to send new messages

        :param request_iterator:
        :param context:
        :return:
        """
        lastindex = 0
        # For every client a infinite loop starts (in gRPC's own managed thread)
        while True:
            # Check if there are any new messages
            while len(self.chats) > lastindex:
                n = self.chats[lastindex]
                lastindex += 1
                yield n

    def SendNote(self, request: chat.Note, context):
        """
        This method is called when a clients sends a Note to the server.

        :param request:
        :param context:
        :return:
        """
        # this is only for the server console
        print("[{}] {}".format(request.name, request.message))
        s = request.message
        s = translator(s)
        request.message = s
        # Add it to the chat history
        self.chats.append(request)
        return chat.Empty()  # something needs to be returned required by protobuf language, we just return empty msg


if __name__ == '__main__':
    port = 11912  # a random port for the server to run on
    # the workers is like the amount of threads that can be opened at the same time, when there are 10 clients connected
    # then no more clients able to connect to the server.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))  # create a gRPC server
    rpc.add_ChatServerServicer_to_server(ChatServer(), server)  # register the server to gRPC
    # gRPC basically manages all the threading and server responding logic, which is perfect!
    print('Starting server. Listening...')
    server.add_insecure_port('[::]:' + str(port))
    server.start()
    # Server starts in background (in another thread) so keep waiting
    # if we don't wait here the main thread will end, which will end all the child threads, and thus the threads
    # from the server won't continue to work and stop the server
    while True:
        time.sleep(64 * 64 * 100)