import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement du dataset
df=pd.read_csv("train.csv")

# Titre et navigation sur 3 pages
st.title("Projet de classification binaire Titanic")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

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

   # Suppression des variables non pertinentes
   df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

   # Séparation variables catégorielles, numériques et target
   y = df['Survived']
   X_cat = df[['Pclass', 'Sex',  'Embarked']]
   X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

   # Remplacement des valeurs manquantes et encodage des variables catégorielles 
   for col in X_cat.columns:
     X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
   for col in X_num.columns:
    X_num[col] = X_num[col].fillna(X_num[col].median())
   X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
   X = pd.concat([X_cat_scaled, X_num], axis = 1)

   # Séparation train test
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

   # Standardisation des variables numériques
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
   X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

   # Fonction d'entraînement
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.svm import SVC
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import confusion_matrix

   def prediction(classifier):
    if classifier == 'Random Forest':
        clf = RandomForestClassifier()
    elif classifier == 'SVC':
        clf = SVC()
    elif classifier == 'Logistic Regression':
        clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf
   
   # Fonction d'évaluation
   def scores(clf, choice):
    if choice == 'Accuracy':
        return clf.score(X_test, y_test)
    elif choice == 'Confusion matrix':
        return confusion_matrix(y_test, clf.predict(X_test))
    
   # Choix du modèle
   choix = ['Random Forest', 'SVC', 'Logistic Regression']
   option = st.selectbox('Choix du modèle', choix)
   st.write('Le modèle choisi est :', option)

   clf = prediction(option)
   display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
   if display == 'Accuracy':
    st.write(scores(clf, display))
   elif display == 'Confusion matrix':
    st.dataframe(scores(clf, display))
