# TP--Find_the_Feel

Structure du repository : 

data/ :   
    d01_raw/ : contient les données textes brutes issues de 2 jeux de données  
    03_cleaned_data/ : contient les données textes adaptées à différentes utilisations (datall : jeux fusionnés, CleanBin : jeux avec 2 émotions (positive et négativ, Clean :                               jeux avec 6 émotions  
  
notebook/ :  
    Cleaning : Nettoyage des données pour exporter les jeux de données nettoyés  
    Machine_learning : Mise en place et entrainement des modèles (regression logistique, xgboost, random forest, svm, neural network), export des résultats via joblib  
  
source/ :  
    app.py : Application streamlit permettant l'analyse des resultats des modèles et des jeux de données  
  

Contexte du projet

​

Vous devrez proposer plusieurs modèles de classification des émotions et proposer une analyse qualitative et quantitative de ces modèles en fonction de critères d'évaluation.. Vous devrez investiguer aux travers de librairies d'apprentissage automatique standards et de traitement automatique du langage naturel comment obtenir les meilleurs performance de prédiction possible en prenant en compte l'aspect multi-class du problème et en explorant l'impact sur la prédiction de divers prétraitement tel que la suppression des stop-words, la lemmatisation et l'utilisation de n-grams, et différente approche pour la vectorisation.

Vous devrez travailler dans un premier temps avec le jeu de données issue de Kaggle pour réaliser vos apprentissage et l'évaluation de vos modèles.

Dans l'objectif d'enrichir notre prédictions nous souhaitons augmenter notre jeux de donneés. Vous devrez donc travailler dans un deuxième temps avec le jeux de données fournie, issue de data.world afin de :

​

    Comparer d'une part si les résultats de classification sur votre premier jeux de données sont similaire avec le second. Commentez.
    Combiner les deux jeux données pour tenter d'améliorer vos résultats de prédiction.
    Prédire les nouvelles émotions présente dans ce jeux de données sur les message du premier, et observer si les résultats sont pertinent.

-- Vous devez en tout créer 5 classifiers différents et les comparer. Parmi ces 5 classifiers, l'un d'entre eux devra être un réseau de neurones prennant pour input des données ayant subit un embedding.

Vous devrez ensuite présenter vos résultats sous la forme d'un dashboard muli-pages Streamlit:

La première page du Dashboard sera dédiée à l'analyse et au traitement des données. Vous pourrez par exemple présenter les données "brut" sous la forme d'un tableau puis les données pré-traitées dans le même tableau avec un bouton ou menu déroulant permettant de passer d'un type de données à un autre (n'afficher qu'un échantillon des résultats, on dans une fenetre "scrollable"). Sur cette première page de dashboard seront accessibles vos graphiques ayant trait à votre première analyse de données (histogramme, bubble chart, scatterplot etc), notamment -l'histogramme représentant la fréquence d’apparition des mots (commentez) -l'histogramme des émotions (commentez)

Une deuxième page du Dashboard sera dédiée aux résultats issues des classifications . Il vous est demandé de comparer les résultats d'au moins 5 classifiers que vous présenterez dans un tableau permettant de visualiser vos mesures. Sur cette page de dashboard pourra se trouver par exemple, des courbes de rappel de précision (permette de tracer la précision et le rappel pour différents seuils de probabilité), un rapport de classification (un rapport de classification visuel qui affiche la precision, le recall, le f1-score, support, ou encore une matrice de confusion ou encore une graphique permettant de visualiser les mots les plus représentatif associé à chaque émotions.

Héberger le dashboard sur le cloud de visualisation de données Héroku (https://www.heroku.com/)

Vos travaux devront être “poussés” sur Github.

BONUS

Créer une application client/serveur permettant à un utilisateur d'envoyer du texte via un champs de recherche (ou un fichier sur le disque du client) et de lui renvoyer l'émotion du texte envoyé. L'application pourra également renvoyer la roue des émotions du document (exemple: quelle proportion de chacune des émotions contient le document ?)

