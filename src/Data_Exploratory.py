# Import Pandas
import pandas as pd

#Import Graphycal lybrary
import matplotlib.pyplot as plt

# Disable Warning
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

#Import Numberwork library
import numpy as np

# Import Personal library
from visualisation_tool import make_plot

from seaborn import barplot

#Import data
df_dataK = pd.read_csv(r'../data/d01_raw/Emotion_Kaggle.csv')

from wordcloud import WordCloud , STOPWORDS
STOPWORDS.update(['feel','feeling','im',",","t","u","2","'","&amp;","-","...","s"])



#Import data
df_dataW = pd.read_csv(r'../data/d01_raw/Emotion_Dataworld.csv')

df_dataW.drop(columns=["tweet_id","author"],inplace=True)
df_dataW.rename(columns={"sentiment": "Emotion", "content": "Text"}, inplace=True)
