import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Modèles
modeles = {
  'Random Forest' : 'rfc_titanic',
  'SVM' : 'svc_titanic',
  'Logistic Regression' : 'lr_titanic',
}

# Fonction de chargement d'un modèle (avec mise en cache)
@st.cache_resource
def load_model(nom_modele):
    clf =  joblib.load(modeles[nom_modele])
    return clf

# Fonction de chargement du dataset (avec mise en cache)
@st.cache_data
def load_dataset():
  df = pd.read_csv("train.csv")
  return df

# Fonction de chargement des données de test prétraitées (avec mise en cache)
@st.cache_data
def load_test_data():
    X_test, y_test = joblib.load("test_titanic")
    return X_test, y_test

# Chargement du dataset et des données de test prétraitées
df = load_dataset()
X_test, y_test = load_test_data()

# Titre et navigation sur 3 pages
st.title("Projet de classification binaire Titanic")
pages = ["Exploration", "DataVizualization", "Modélisation"]
with st.sidebar:
    page = option_menu(
        menu_title="Sommaire",       # titre du menu
        options=pages,  # noms des pages
        icons=["search", "bar-chart", "cpu"],  # icônes Bootstrap
        menu_icon="cast",              # icône du menu
        default_index=0,               # page par défaut
    )

# Page Exploration
if page == pages[0] : 
  st.write("### Introduction")

  st.dataframe(df.head(10))
  st.write(df.shape)
  st.dataframe(df.describe())

  if st.checkbox("Afficher les NA") :
    st.dataframe(df.isna().sum())

# Page DataViz
if page == pages[1] : 
  st.write("### DataVizualization")

  fig = plt.figure()
  sns.countplot(x = 'Survived', data = df)
  st.pyplot(fig)

  fig = plt.figure()
  sns.countplot(x = 'Sex', data = df)
  plt.title("Répartition du genre des passagers")
  st.pyplot(fig)
  
  fig = plt.figure()
  sns.countplot(x = 'Pclass', data = df)
  plt.title("Répartition des classes des passagers")
  st.pyplot(fig)
  
  fig = sns.displot(x = 'Age', data = df)
  plt.title("Distribution de l'âge des passagers")
  st.pyplot(fig)

  fig = plt.figure()
  sns.countplot(x = 'Survived', hue='Sex', data = df)
  st.pyplot(fig)
  
  fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
  st.pyplot(fig)
  
  fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
  st.pyplot(fig)

  fig, ax = plt.subplots()
  sns.heatmap(df.select_dtypes(include=[np.number]).corr(), ax=ax)
  st.write(fig)

# Page modélisation
if page == pages[2] : 
   st.write("### Modélisation")
    
   # Choix du modèle
   option = st.selectbox('Choix du modèle', list(modeles.keys()))
   st.write('Le modèle choisi est :', option)

   # Chargement du modèle
   clf = load_model(option)

   # Evaluation
   from sklearn.metrics import confusion_matrix
   display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
   if display == 'Accuracy':
    score = clf.score(X_test, y_test)
    st.metric("Accuracy", round(score, 3))
   elif display == 'Confusion matrix':
    cm = confusion_matrix(y_test, clf.predict(X_test))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    st.pyplot(fig)