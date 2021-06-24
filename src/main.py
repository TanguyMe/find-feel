
from fastapi import FastAPI

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from emoji import demojize
import re

from reg_lin import get_metrix, get_metrix_v2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from tensorflow.keras import layers, preprocessing
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

from nltk.stem import WordNetLemmatizer


def lemmatizer(text):
    """Apply Wordnet lemmatizer to text (go to root word)"""
    wnl = WordNetLemmatizer()
    text = [wnl.lemmatize(word) for word in text.split()]
    return " ".join(text)


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


df_dataK = pd.read_csv(r'../data/d03_cleaned_data/CleanKaggleBin.csv')
df_dataK.head()

df_dataK['text'] = df_dataK['text'].apply(lemmatizer)
df_dataK['text'] = clean_str(df_dataK['text'])

Labelenc = LabelEncoder()
df_dataK['emotion'] = Labelenc.fit_transform(df_dataK['emotion'])
y = df_dataK['emotion']

df_dataKKeras = df_dataK[['text']].copy()
tk = preprocessing.text.Tokenizer()

tk.fit_on_texts(df_dataKKeras['text'])
X = tk.texts_to_matrix(df_dataKKeras['text'], mode='tfidf')

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=None)
model = Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=[X_train.shape[1]]))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(X_train, y_train, epochs=10, batch_size=10)

    

app = FastAPI()


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}


@app.get("/{item}")
async def create_item(item: str):
    x = tk.texts_to_matrix([item], mode='tfidf')
    pred = model.predict(x)
    pred = Labelenc.inverse_transform([round(pred[0][0])])[0]
    print(pred)
    return {"Le texte entré a une connotation " : pred}