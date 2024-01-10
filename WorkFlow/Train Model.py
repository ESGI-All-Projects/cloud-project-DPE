# Databricks notebook source
# MAGIC %md
# MAGIC # I. Import Libs

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


# COMMAND ----------

# MAGIC %md
# MAGIC # II. Import Cleaned Data

# COMMAND ----------

# Chargement des données (adaptez le chemin selon vos fichiers)
train_processed_table = spark.table("train_processed_table").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC # III. Split Data

# COMMAND ----------

# Extraction de la variable cible et des caractéristiques
y = train_processed_table['Etiquette_DPE']
X = train_processed_table.drop(['Etiquette_DPE'], axis=1)

# Encode les variables pour le modèle
categorical_column = X.select_dtypes(include=['object', 'category']).columns
y = pd.get_dummies(y)
X = pd.get_dummies(X, columns=categorical_column)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC # Train model

# COMMAND ----------

def train_model(X_train, y_train, X_test, y_test):
    # model = xgb.XGBClassifier(num_class=7, n_estimators=150, learning_rate=0.1, max_depth=7)
    model = xgb.XGBClassifier(n_estimators=200, early_stopping_rounds=10)
    model = xgb.XGBClassifier(n_estimators=5, early_stopping_rounds=2) # =========================== 0 DEL ======================================
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    # model.fit(X_train, y_train)
    return model


# COMMAND ----------

model = train_model(X_train, y_train, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # VI Evaluate Model

# COMMAND ----------

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)


    y_test_indices = np.argmax(np.array(y_test), axis=1)
    y_pred_indices = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test_indices, y_pred_indices)

    # Affichage de la matrice de confusion avec seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=y_test.columns, yticklabels=y_test.columns)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return accuracy, report

# COMMAND ----------

accuracy, report = evaluate_model(model, X_test, y_test)
print(f"Précision améliorée sur les données de test : {accuracy:.2f}%")
print("\nRapport de classification amélioré :")
print(report)

# COMMAND ----------

# MAGIC %md
# MAGIC # Analyse de l'Importance des Caractéristiques

# COMMAND ----------

def plot_feature_importance(model, X_train, max_num_features=10):
    # Récupération de l'importance des fonctionnalités
    feature_importances = model.feature_importances_
    
    # Créer un DataFrame pour faciliter l'affichage
    features_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False).head(max_num_features)  # Limiter au top 10

    # Affichage de l'importance des fonctionnalités
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=features_df)
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.show()

plot_feature_importance(model, X_train)

# COMMAND ----------


