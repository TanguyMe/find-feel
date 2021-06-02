# Import Pandas
import pandas as pd
import streamlit as st

#Set Relative Path
import sys
sys.path.append("../src")

# Import data
df_dataK = pd.read_csv(r'../data/d03_cleaned_data/datall.csv')
df_dataK.head()

from nltk.stem import WordNetLemmatizer

df_display = pd.DataFrame()
df_display['brutText'] = df_dataK['text']

def lemmatizer(text):
    """Apply Wordnet lemmatizer to text (go to root word)"""
    wnl = WordNetLemmatizer()
    text = [wnl.lemmatize(word) for word in text.split()]
    return " ".join(text)


df_dataK['text'] = df_dataK['text'].apply(lemmatizer)
df_display['lemmatized'] = df_dataK['text']

# suppression des caratères spéciaux, formatage des contractions...
from nltk.corpus import stopwords
from emoji import demojize
import re

def clean_str(texts):
    from nltk.corpus import stopwords
    # Lowercasing
    texts = texts.str.lower()
    df_display['lowered'] = texts

    # Remove special chars
    texts = texts.str.replace(r"(http|@)\S+", "")
    texts = texts.apply(demojize)
    texts = texts.str.replace(r"::", ": :")
    texts = texts.str.replace(r"’", "'")
    texts = texts.str.replace(r"[^a-z\':_]", " ")
    df_display['removespecialchars'] = texts

    # Remove repetitions
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    texts = texts.str.replace(pattern, r"\1")
    df_display['removerepetitions'] = texts

    # Transform short negation form
    texts = texts.str.replace(r"(can't|cannot)", 'can not')
    texts = texts.str.replace(r"(ain't|wasn't|weren't)", 'be not')
    texts = texts.str.replace(r"(don't|didn't|didnt)", 'do not')
    texts = texts.str.replace(r"(haven't|hasn't)", 'have not')
    texts = texts.str.replace(r"(won't)", 'will not')
    texts = texts.str.replace(r"(im)", ' i am')
    texts = texts.str.replace(r"(ive)", ' i have')
    texts = texts.str.replace(r"(n't)", ' not')
    df_display['removeshortform'] = texts

    # Remove stop words
    stopwords = stopwords.words('english')
    stopwords.remove('not')
    stopwords.remove('nor')
    stopwords.remove('no')
    texts = texts.apply(lambda x: ' '.join([word for word in x.split() if (word not in stopwords and len(word) > 1 )]))
    df_display['removestopwords'] = texts
    return texts


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
def model_results_multi(df_dataK):
    # Set X and y
    enc, X = model_used(df_dataK, 'BERT')

    Labelenc = LabelEncoder()
    df_dataK['emotion'] = Labelenc.fit_transform(df_dataK['emotion'])
    y = df_dataK['emotion']

    models = [LogisticRegression(n_jobs=-1, multi_class='multinomial', penalty='l2', solver='saga', max_iter=500, C=1),
              SGDClassifier(n_jobs=-1, penalty='l2', max_iter=500, loss='hinge', alpha=0.001),
              RandomForestClassifier(n_jobs=-1, n_estimators=1000, min_samples_split=8, min_samples_leaf=2,
                                     criterion='entropy'),
              XGBClassifier(),
              Sequential()]
    results={}
    for modelset in models:
        if 'Sequential' in str(modelset):
            df_dataKKeras = df_dataK.sample(15000)
            y = df_dataKKeras['emotion']

            df_dataKKeras = df_dataKKeras[['text']]
            tk = preprocessing.text.Tokenizer()
            tk.fit_on_texts(df_dataKKeras['text'])
            X = tk.texts_to_matrix(df_dataKKeras['text'], mode='tfidf')

            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=None)
            model = modelset
            model.add(layers.Dense(160, activation='relu', input_shape=[X_train.shape[1]]))
            model.add(layers.Dense(80, activation='relu'))
            model.add(layers.Dense(40, activation='relu'))
            model.add(layers.Dense(6, activation='softmax'))
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=10, batch_size=100)

            y_train_pred = model.predict_classes(X_train)
            y_test_pred = model.predict_classes(X_test)
            y_test_proba = model.predict_proba(X_test)
            y_test_pred = Labelenc.inverse_transform([round(y_test_pred[0][0])])
            print(" Train accuracy = {}"
                  .format(round(accuracy_score(y_train, y_train_pred), 3)))

            print(" Test accuracy = {}"
                  .format(round(accuracy_score(y_test, y_test_pred), 3)))
            results['NeuralNetwork'] = [model, y_test_proba, y_test_pred, y_test]
        else:
            try:
                model, y_test_proba, y_test_pred, y_test = get_metrix_v2(y, X, modelset)
            except:
                model, y_test_pred, y_test = get_metrix(y, X, modelset)
                y_test_proba = None
            y_test_pred = Labelenc.inverse_transform(y_test_pred)
            if 'Logistic' in str(modelset):
                results['LogisticRegression'] = [model, y_test_proba, y_test_pred, y_test]
            elif 'SGD' in str(modelset):
                results['SGDClassifier'] = [model, y_test_proba, y_test_pred, y_test]
            elif 'Forest' in str(modelset):
                results['RandomForest'] = [model, y_test_proba, y_test_pred, y_test]
            elif 'XGB' in str(modelset):
                results['Xgboost'] = [model, y_test_proba, y_test_pred, y_test]
            else:
                results[str(model)] = [model, y_test_proba, y_test_pred, y_test]
    # del x,X,y,X_test,X_train,y_test,y_train,y_test_pred,y_train_pred,df_dataK,df_dataKKeras
    return results

# # Import data
# df_dataK = pd.read_csv(r'../data/d03_cleaned_data/datall_test.csv')
#
# df_dataK['text'] = df_dataK['text'].apply(lemmatizer)
# df_dataK['text'] = clean_str(df_dataK['text'])
# X_test = enc.encode(df_dataK['text'])
# y_test = Labelenc.transform(df_dataK['emotion'])
#
# count = 0
# place = 4
#
# for model in savedmodel:
#
#     print (model)
#     if count != place:
#         y_test_pred = model.predict(X_test)
#     if count == place:
#         X_test = tk.texts_to_matrix(df_dataK['text'], mode='tfidf')
#         y_test_pred = model.predict(X_test)
#     print (" Test accuracy = {}"
#             .format(round(accuracy_score(y_test, y_test_pred),3)))
#
#     count += 1


