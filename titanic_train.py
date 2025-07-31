import pandas as pd
import numpy as np

# Chargement du dataset
df = pd.read_csv("train.csv")

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

# Entraînement des modèles
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Evaluation
print ("Accuracy Random Forest : ",rfc.score(X_test,y_test))
print ("Accuracy SVM : ",svc.score(X_test,y_test))
print ("Accuracy Logistic Regression : ",lr.score(X_test,y_test))

# Sauvegarde des modèles et des données de test
import joblib
joblib.dump(rfc, "rfc_titanic")
joblib.dump(svc, "svc_titanic")
joblib.dump(lr, "lr_titanic")
joblib.dump((X_test, y_test), "test_titanic")