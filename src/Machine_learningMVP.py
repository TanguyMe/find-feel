# Import Pandas
import pandas as pd
import streamlit as st



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from wordcloud import STOPWORDS
from reg_lin import get_metrix_v2
import numpy as np
from sklearn.linear_model import LogisticRegression


from reg_lin import get_metrix
from sklearn.linear_model import LogisticRegression ,SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from tensorflow.keras import layers, preprocessing
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import load_model


# Import data
df_dataK = pd.read_csv(r'../data/d03_cleaned_data/CleanKaggleBin.csv')
df_dataK.head()

from nltk.stem import WordNetLemmatizer
def lemmatizer(text):
    """Apply Wordnet lemmatizer to text (go to root word)"""
    wnl = WordNetLemmatizer()
    text = [wnl.lemmatize(word) for word in text.split()]
    return " ".join(text)
df_dataK['text'] = df_dataK['text'].apply(lemmatizer)

# suppression des caratères spéciaux, formatage des contractions...
from nltk.corpus import stopwords
from emoji import demojize
import re

def clean_str(texts):
    from nltk.corpus import stopwords
    # Lowercasing
    texts = texts.str.lower()

    # Remove special chars
    texts = texts.str.replace(r"(http|@)\S+", "")
    texts = texts.apply(demojize)
    texts = texts.str.replace(r"::", ": :")
    texts = texts.str.replace(r"’", "'")
    texts = texts.str.replace(r"[^a-z\':_]", " ")

    # Remove repetitions
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    texts = texts.str.replace(pattern, r"\1")

    # Transform short negation form
    texts = texts.str.replace(r"(can't|cannot)", 'can not')
    texts = texts.str.replace(r"(ain't|wasn't|weren't)", 'be not')
    texts = texts.str.replace(r"(don't|didn't|didnt)", 'do not')
    texts = texts.str.replace(r"(haven't|hasn't)", 'have not')
    texts = texts.str.replace(r"(won't)", 'will not')
    texts = texts.str.replace(r"(im)", ' i am')
    texts = texts.str.replace(r"(ive)", ' i have')
    texts = texts.str.replace(r"(n't)", ' not')

    # Remove stop words
    stopwords = stopwords.words('english')
    stopwords.remove('not')
    stopwords.remove('nor')
    stopwords.remove('no')
    texts = texts.apply(lambda x: ' '.join([word for word in x.split() if (word not in stopwords and len(word) > 1 )]))
    return texts


df_display = pd.DataFrame()
df_display['brutText'] = df_dataK['text']
df_dataK['text'] = clean_str(df_dataK['text'])
df_display['cleanedText'] = df_dataK['text']
df_dataK_ready = df_dataK

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer


def model_used(df, model):
    """Given a model choice, return the model and the computed matrix"""
    if model == 'Tfidf':
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0)
        tfidf_matrix = tf.fit_transform(df['text'])
        return tf, tfidf_matrix
    elif model == 'CountVectorizer':
        cv = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0)
        matrix = cv.fit_transform(df['text'])
        return cv, matrix
    elif model == 'BERT':
#         bert = SentenceTransformer('distiluse-base-multilingual-cased-v1') # Multilingue
#         bert = SentenceTransformer('average_word_embeddings_glove.6B.300d') # + rapide
        bert = SentenceTransformer('paraphrase-MiniLM-L6-v2') # Meilleur score en théorie, à vérifier sur nos données
        matrix = bert.encode(df['text'].astype('str'), show_progress_bar=True)
        return bert, matrix


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# # Set X and y
# enc, X = model_used(df_dataK,'BERT')
#
# Labelenc = LabelEncoder()
# df_dataK['emotion'] = Labelenc.fit_transform(df_dataK['emotion'])
# y = df_dataK['emotion']

from reg_lin import get_metrix, get_metrix_v2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from tensorflow.keras import layers, preprocessing
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score




@st.cache(allow_output_mutation=True)
def model_results(stop_words=STOPWORDS):

    # Import data
    df_dataK = pd.read_csv(r'../data/d03_cleaned_data/CleanKaggleBin.csv')
    df_dataK.head()

    STOPWORDS.update(['feel','feeling','im',",","t","u","2","'","&amp;","-","...","s"])

    # Set X and y
    # tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words=stop_words)
    # tfidf_matrix = tf.fit_transform(df_dataK['text'])
    # X = tfidf_matrix
    df_dataK['text'] = df_dataK['text'].apply(lemmatizer)
    df_dataK['text'] = clean_str(df_dataK['text'])
    enc, X = model_used(df_dataK, 'BERT')
    Labelenc = LabelEncoder()
    df_dataK['emotion'] = Labelenc.fit_transform(df_dataK['emotion'])
    y = df_dataK['emotion']

    models = [LogisticRegression(),
          SGDClassifier(),
          RandomForestClassifier(),
          XGBClassifier(),
          Sequential()
              ]
    results = {}
    for modelset in models:
        if 'Sequential' in str(modelset):
            df_dataKKeras = df_dataK[['text']].copy()
            tk = preprocessing.text.Tokenizer()

            tk.fit_on_texts(df_dataKKeras['text'])
            X = tk.texts_to_matrix(df_dataKKeras['text'], mode='tfidf')

            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=None)
            model = modelset
            model.add(layers.Dense(16, activation='relu', input_shape=[X_train.shape[1]]))
            model.add(layers.Dense(8, activation='relu'))
            model.add(layers.Dense(4, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(optimizer="adam", loss="binary_crossentropy")
            model.fit(X_train, y_train, epochs=10, batch_size=10)
            y_test_pred = model.predict_classes(X_test)
            y_test_pred = np.array([x[0] for x in y_test_pred])
            y_test_proba = model.predict_proba(X_test)
            # results['NeuralNetwork'] = [model, y_test_proba, y_test_pred, y_test]
            results['NeuralNetwork'] = [None, y_test_proba, y_test_pred, y_test]

        else:
            try:
                model, y_test_proba, y_test_pred, y_test = get_metrix_v2(y, X, modelset)
            except:
                model, y_test_pred, y_test = get_metrix(y, X, modelset)
                y_test_proba = None
            if 'Logistic' in str(modelset):
                # results['LogisticRegression'] = [model, y_test_proba, y_test_pred, y_test]
                results['LogisticRegression'] = [None, y_test_proba, y_test_pred, y_test]
            elif 'SGD' in str(modelset):
                # results['SGDClassifier'] = [model, y_test_proba, y_test_pred, y_test]
                results['SGDClassifier'] = [None, y_test_proba, y_test_pred, y_test]
            elif 'Forest' in str(modelset):
                # results['RandomForest'] = [model, y_test_proba, y_test_pred, y_test]
                results['RandomForest'] = [None, y_test_proba, y_test_pred, y_test]
            elif 'XGB' in str(modelset):
                # results['Xgboost'] = [model, y_test_proba, y_test_pred, y_test]
                results['Xgboost'] = [None, y_test_proba, y_test_pred, y_test]
            else:
                # results[str(model)] = [model, y_test_proba, y_test_pred, y_test]
                results[str(model)] = [None, y_test_proba, y_test_pred, y_test]

    return results


results = model_results()

joblib.dump(results,"../dump/results_bin_BERT.joblib")