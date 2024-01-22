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

def get_data(path):
    data = pd.read_csv(path)
    return data

# data  = get_data("language.csv")
labels = {0: 'English', 1: 'Malayalam', 2: 'Hindi', 3: 'Tamil', 4: 'Portugeese', 5: 'French', 6: 'Dutch', 7: 'Spanish', 8: 'Greek', 9: 'Russian', 10: 'Danish', 11: 'Italian', 12: 'Turkish', 13: 'Sweedish', 14: 'Arabic', 15: 'German', 16: 'Kannada'}


# for i in range(len(data)):
#     if(data["label"][i] not in labels):
#         labels[data["label"][i]] = data["Language"][i]

# print(labels)
# load model
nlp = spacy.load("en_core_web_sm")

# preprocessing
def preprocessing(text):
    text = text.lower()
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens


def classifier(text,model):
    text = preprocessing(text)
    text = " ".join(text)
    return labels[model.predict([text])[0]]

# print(classifier("ich bin sher klug",model))
