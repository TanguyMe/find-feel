# ------------------------------------------ Import ------------------------------------------

# Importing the libraries

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from Machine_learningMVP import y_pred_proba, y_true, model_results
# import random as rd

# ------------------------------------------ Code ------------------------------------------

from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc, accuracy_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support   #, classification_report
from Data_Exploratory import df_dataW, df_dataK
from wordcloud import STOPWORDS
from collections import Counter
from Machine_learning_advanced import df_display  #, model_results_multi, df_dataK_ready
import scikitplot as skplt
import joblib
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import load_model
# import autokeras as ak
#
# load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)

classes=['anger', 'fear', 'happy', 'love', 'sadness', 'surprise']


def plot_results(y_true, y_probas, title, classes_to_plot=classes):
    """
    Given probabilities and true target, plot a roc curve for each category
    """
    fig, ax = plt.subplots()
    skplt.metrics.plot_roc(
        y_true, y_probas,
        classes_to_plot=classes_to_plot,
        ax=ax,
        title=title,
        cmap="tab20",
        plot_macro=False
    )
    return fig


@st.cache(allow_output_mutation=True)
def extract_most_used_word(text, stopwords=[], words=1, limit=False):
    """
    Return the wanted number of most used word of a given text and their number of occurences
    If limit is set to a value, it will return all words that appears more than limit value
    """
    words_list=[]
    # iterate through the csv file
    for val in text:
        # typecaste each val to string
        val = str(val)
        # split the value
        tokens = val.split()
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
            if not tokens[i] in stopwords:
                words_list.append(tokens[i])
    occurence_count = Counter(words_list)
    occ = np.array(occurence_count.most_common())
    if limit:
        if int(limit) == limit:
            return occ[occ[:,1].astype('int') > limit]
        try :
            return occ[occ[:,1].astype('int') > limit*text.shape[0]]
        except:
            return occ[occ[:,1].astype('int') > limit*len(occ)]
    return occ[:words]


def extract_most_used_words(df, text_column='Text', emotion_column='Emotion', max_occurences=3, limit=500):
    """
    Extract the words that are the most frequently used in several columns of the dataframe
    Max_occurences sets the number of columns in which the word must appear
    Limit sets the minimum number of time the word must appear in the column to be considered frequently used
    """
    wordlist = []
    for emotion in df[emotion_column].unique():
        wordlist.extend(extract_most_used_word(df[df[emotion_column] == emotion][text_column], limit=limit)[:, 0])
    return list(set([i for i in wordlist if wordlist.count(i) > max_occurences]))


def define_stopwords(stopwords, df, emotion_column='Emotion', name=''):
    """
    Define what list of stopwords will be used given an input (taken from streamlit)
    """
    if stopwords == 'Aucun':
        stop_words = []
    if stopwords == 'Wordcloud':
        stop_words = STOPWORDS
    if stopwords == 'Par répétition':
        with st.sidebar.beta_expander('Paramètres du stopword du ' + name):
            limit=st.slider('Limite', min_value=0., max_value=0.1, value=0.05, step=0.001, key=name)
            max_occurences=st.slider("Nombre limite d'occurences", min_value=2, max_value=len(df[emotion_column].unique()), value=2, key=name)
        stop_words = extract_most_used_words(df=df, max_occurences=max_occurences, limit=limit)
    return stop_words


def word_global_repartition_plot(df):
    """
    Plot the word frequency of a dataframe
    """
    data = df['Emotion'].value_counts()
    ax = sns.barplot(x=data.index, y=data.values)
    fig = plt.gcf()
    ax.tick_params('x', labelrotation=45)
    # Modifie la taille du graphique
    fig.set_size_inches(10, 4)
    # Ajout du titre
    fig.suptitle("Répartition des émotions", fontsize=18)
    # Ajout des labels pour les axes x et y
    plt.xlabel("Emotion", fontsize=20);
    plt.ylabel("Occurences", fontsize=20, rotation=90)
    st.pyplot(fig)
    plt.close()


def word_local_repartition_plot2(df, emotion_column='Emotion', text_column='Text', words=20, stopwords=None):
    """
    Plot the word repartition for every emotion, given a user choice
    """
    emotions = df[emotion_column].unique()
    chosen_emotions = st.multiselect('Emotions', list(emotions), default=list(emotions))
    count = 0
    list_cols=[]
    for val in range(len(chosen_emotions)//2):
        cols = plt.subplots(1, 2,  figsize=(12,12))
        list_cols.append(cols)
    if len(chosen_emotions)%2:
        cols = plt.subplots(1, 2,  figsize=(12,12))
        list_cols.append(cols)
    st.write(list_cols)
    for emotion in chosen_emotions:
        x = extract_most_used_word(df[df[emotion_column] == emotion][text_column], words=words, stopwords=stopwords)[:, 0]
        y = extract_most_used_word(df[df[emotion_column] == emotion][text_column], words=words, stopwords=stopwords)[:, 1].astype('int')

        ax = sns.barplot(x=x, y=y)
        fig = plt.gcf()
        # Modifie la taille du graphique
        fig.set_size_inches(10, 4)
        # Ajout du titre
        fig.suptitle("Répartition pour " + emotion, fontsize=18)
        # Ajout des labels pour les axes x et y
        plt.xlabel("Mots", fontsize=20);
        plt.ylabel("Occurences", fontsize=20, rotation=90)
        # list_cols[count//2][count%2].pyplot(fig,  use_column_width=True)
        list_cols[count // 2][count % 2].pyplot(fig)
        plt.close()


def word_local_repartition_plot(df, emotion_column='Emotion', text_column='Text', words=20, stopwords=[]):
    """
    Plot the word repartition for every emotion, given a user choice
    """
    emotions = df[emotion_column].unique()
    chosen_emotions = st.multiselect('Emotions', list(emotions), default=list(emotions))
    count = 0
    list_cols=[]
    figsize=(12,4)
    for val in range(len(chosen_emotions)//2):
        cols = plt.subplots(1, 2,  figsize=figsize)
        list_cols.append(cols)
    if len(chosen_emotions)%2:
        cols = plt.subplots(1, 1,  figsize=figsize)
        list_cols.append(cols)
    for emotion in chosen_emotions:
        x = extract_most_used_word(df[df[emotion_column] == emotion][text_column], words=words, stopwords=stopwords)[:,0]
        y = extract_most_used_word(df[df[emotion_column] == emotion][text_column], words=words, stopwords=stopwords)[:,1].astype('int')
        if not count%2:
            fig = list_cols[count//2][0]
        try:
            ax=list_cols[count//2][1][count%2]
        except:
            ax = list_cols[count // 2][1]
        sns.barplot(ax=ax, x=x ,y=y)
        ax.tick_params('x', labelrotation=90)
        # Ajout des labels pour les axes x et y
        ax.set_xlabel('Mots', fontsize=18)
        ax.set_ylabel('Occurences', fontsize=18)
        ax.set_title('Répartition pour ' + emotion, fontweight='bold', size=20)
        if count%2:
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        count=count+1
    if count%2:
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# Rappel et précision en fonction de la proba
@st.cache(allow_output_mutation=True)
def rappel_precision_f1(prob, ytrue, steps=101):
    """
    Return the recall, precision and f1_score and an array containing each probability for which the metrics where calculated
    Steps represents the number of probability taken
    """
    x = np.linspace(0,1,steps)
    rap = []
    pre = []
    f1 = []
    for val in x :
        ypred = [proba >= val for proba in prob]
        rap.append(recall_score(ytrue,ypred))
        pre.append(precision_score(ytrue,ypred))
        f1.append(f1_score(ytrue,ypred))
    return rap, pre, f1, x


def prediction(results):
    """Display the pridction page : metrics, matrix and tables"""
    models = st.sidebar.multiselect('Modèle', list(results.keys()), default=list(results.keys())[0])
    metrics = st.sidebar.multiselect('Métriques', ['f1', 'rappel', 'precision'], default=['f1', 'rappel', 'precision'])
    list_graphes = ['Métrique individuelle', 'Heatmap', 'Courbes Roc', 'Tableau de métriques']
    graphes = st.sidebar.multiselect('Graphes à afficher', list_graphes, default=list_graphes)
    list_metr = []
    list_keys = []
    for key, values in results.items():
        if key in models:
            st.header("Résultat de "+ str(key))
            y_pred = values[2]
            y_true = values[3].values
            if list_graphes[0] in graphes:
                if values[1] is not None:
                    y_pred_proba = values[1]
                    st.subheader("Courbe de rappel et précision")
                    try :
                        rappel, precision, f1, x = rappel_precision_f1(y_pred_proba[:, 1], y_true)
                    except:
                        rappel, precision, f1, x = rappel_precision_f1(y_pred_proba[:, 0], y_true)
                    d = {'x': x}
                    if 'f1' in metrics:
                        d['f1'] = f1
                    if 'rappel' in metrics:
                        d['rappel'] = rappel
                    if 'precision'in metrics:
                        d['precision'] = precision
                    # if metrics[0]:
                    #     d['f1'] = f1
                    # if metrics[1]:
                    #     d['rappel'] = rappel
                    # if metrics[2]:
                    #     d['precision'] = precision
                    chart_data = pd.DataFrame(data=d)
                    chart_data = chart_data.set_index('x')
                    st.line_chart(chart_data)
            if list_graphes[1] in graphes:
                st.subheader("Matrice de confusion")
                tabl = pd.DataFrame(confusion_matrix(y_pred, y_true), index = ['Negative', 'Positive'], columns=['Negative', 'Positive'])
                ax = sns.heatmap(tabl, annot=True, fmt='d')
                fig = plt.gcf()
                plt.close()
                st.pyplot(fig)
            list_keys.append(key)
            list_metr.append(pd.DataFrame(precision_recall_fscore_support(y_pred, y_true), columns = ['Negative', 'Positive'], index=['Precision', 'Rappel', 'f1', 'Occurences']).T)
    if list_graphes[2] in graphes:
        st.subheader("Courbes ROC")
        for model in models:
            y_true = results[model][3].values
            y_pred_proba = results[model][1]
            try :
                try:
                    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred_proba[:, 1])
                except:
                    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred_proba[:, 0])
                auc_keras = auc(fpr_keras, tpr_keras)
                plt.figure(1)
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(fpr_keras, tpr_keras, label=model + '(area = {:.3f})'.format(auc_keras))
            except:
                st.write(model + " n'a pas de prédiction de probabilité, on ne peut donc pas tracer sa courbe")
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close()
    if list_graphes[3] in graphes:
        try:
            metr = pd.concat(list_metr, keys=list_keys)
            st.subheader("Tableau de métriques")
            st.write(metr)
        except:
            pass

def prediction_multi(results):
    """Display the prediction with multiple classes : roc curves, accuracy table"""
    models_multi = st.selectbox('Modèle', list(results_multi.keys()), index=0)
    st.subheader("Courbes ROC")
    y_true = results[models_multi][3].values
    y_true = [classes[x] for x in y_true]
    y_pred_proba = results[models_multi][1]
    try :
        fig = plot_results(y_true,y_pred_proba, title=None)
        st.pyplot(fig)
        plt.close()
    except:
        st.write(models_multi + " n'a pas de prédiction de probabilité, on ne peut donc pas tracer sa courbe")
    list_accuracy = []
    list_mod = []
    for key, values in results_multi.items():
        list_mod.append(key)
        y_true = values[3]
        y_true = [classes[x] for x in y_true]
        y_pred = values[2]
        list_accuracy.append(accuracy_score(y_true, y_pred))
    dic = {'Modèle': list_mod, 'Accuracy': list_accuracy}
    chart_data_multi = pd.DataFrame(data=dic)
    chart_data_multi = chart_data_multi.set_index('Modèle')
    st.write(chart_data_multi)


def data_analysis():
    """
    Display the data analysis page : frequency of words for each dataset and each emotion, emotion spread
    """
    stopwords_choice = st.sidebar.selectbox('Choix du type de stopword', ['Aucun', 'Wordcloud', 'Par répétition'],
                                            index=2)
    st.header("Analyse de la donnée")
    st.subheader("Sur le premier jeu de donnée")
    words_number = st.sidebar.slider('Nombre de mots', min_value=1, max_value=40, value=20)
    list_stopwordsK = define_stopwords(stopwords=stopwords_choice, df=df_dataK, name='premier jeu de données')
    word_global_repartition_plot(df_dataK)
    word_local_repartition_plot(df_dataK, stopwords=list_stopwordsK, words=words_number)
    st.subheader("Sur le second jeu de donnée")
    list_stopwordsW = define_stopwords(stopwords=stopwords_choice, df=df_dataW, name='second jeu de données')
    word_global_repartition_plot(df_dataW)
    word_local_repartition_plot(df_dataW, stopwords=list_stopwordsW, words=words_number)


def text():
    """
    Display the raw text and the cleaned text, and any chosen step in between
    """
    st.header("Traitement du text")
    list_col = list(df_display.columns.values)
    st.write("(Le texte a subi les étapes de traitement dans l'ordre suivant : " + str(list_col) + ")")
    cols = st.multiselect('Etape de traitement', list_col, default=[list_col[0], list_col[-1]])
    st.write(df_display[cols].sample(50))


# results = model_results(stop_words=define_stopwords(stopwords=stopwords_choice, df=df_dataK))
st.title("Analyse d'émotions d'un document")
page = st.sidebar.radio('Page', ['Analyse de la donnée', "Traitement du texte", 'Résultats des prédictions binaires', "Résultat des prédictions avec 6 émotions"])


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def download():
    """Download the results from the models (predictions)"""
    empty = st.empty()
    empty.empty()
    with empty:
        st.info("Chargement des résultats")
    results_multi = joblib.load("../dump/results_multi.joblib")
    results_bin = joblib.load("../dump/results_bin_BERT.joblib")
    empty.empty()
    return results_multi, results_bin


results_multi, results_bin = download()


if page == "Analyse de la donnée":
    data_analysis()
if page == "Résultats des prédictions binaires":
    prediction(results=results_bin)
if page == "Traitement du texte":
    text()
if page == "Résultat des prédictions avec 6 émotions":
    prediction_multi(results=results_multi)
