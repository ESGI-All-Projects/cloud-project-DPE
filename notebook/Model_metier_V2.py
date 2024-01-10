# Databricks notebook source
# MAGIC %md # I. Librairies
# MAGIC

# COMMAND ----------

import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report



# COMMAND ----------

# MAGIC %md # II. Preprocess function
# MAGIC
# MAGIC Cette fonction effectue les étapes suivantes :
# MAGIC
# MAGIC - Supprime les valeurs aberrantes.
# MAGIC - Sépare les données en numériques et catégorielles.
# MAGIC - Supprime les colonnes avec un pourcentage élevé de valeurs manquantes.
# MAGIC - Supprime des colonnes spécifiques.
# MAGIC - Combine les données numériques et catégorielles.
# MAGIC - Extrait la variable cible (y) et les caractéristiques (X).
# MAGIC - Encode les caractéristiques catégorielles.

# COMMAND ----------


def clean_data(data):
    # Comptage des valeurs NA et suppression des colonnes avec plus de 10% de NA
    na_percentage = data.isna().mean().round(4) * 100
    print("na_percentage : \n", na_percentage.sort_values())
    cols_to_drop_na = na_percentage[na_percentage > 10].index
    print("\n cols_to_drop : \n", cols_to_drop_na)
    data = data.drop(cols_to_drop_na, axis=1)
    
    # Suppression de colonnes spécifiques
    cols_to_drop = ["Code_postal_(BAN)", "Conso_5_usages_é_finale", "Emission_GES_éclairage",
                    "N°DPE", "Code_INSEE_(BAN)", "Qualité_isolation_plancher_bas", "Nom__commune_(Brut)",
                    "_c0", "Code_postal_(brut)", "Etiquette_GES"]
    
    data = data.drop(cols_to_drop, axis=1, errors='ignore')

    # Séparation en données numériques et catégorielles
    numerical_data = data.select_dtypes(include=['int64', 'float64'])

    # Suppression des valeurs aberrantes
    print("size row before cleaning outlier", len(data))
    def delete_outlier(df, column_name):
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    for col in numerical_data.columns:
        data = delete_outlier(data, col)
    print("size row after cleaning outlier", len(data))
    print("\n valeur de X : \n", data.columns)

    return data



# COMMAND ----------

# MAGIC %md
# MAGIC # III. Load Data - Clean and Saved

# COMMAND ----------

# Chargement des données (adaptez le chemin selon vos fichiers)
train_data = spark.read.csv("/mnt/data/train.csv", header=True, inferSchema=True)
test_data = spark.read.csv("/mnt/data/test.csv", header=True, inferSchema=True)

# Combinaison et prétraitement des données
data = train_data.union(test_data)
# data = all_data.sample(withReplacement=False, fraction=0.10, seed=42)

data = clean_data(data.toPandas())

# COMMAND ----------

spark_data = spark.createDataFrame(data)
spark_data.write.format("parquet").mode("overwrite").saveAsTable("train_processed_table")

# COMMAND ----------

# MAGIC %md
# MAGIC #IV. Split data

# COMMAND ----------

train_df = spark.table("train_processed_table").toPandas()
# Extraction de la variable cible et des caractéristiques
y = train_df['Etiquette_DPE']
X = train_df.drop(['Etiquette_DPE'], axis=1)

# Encode les variables pour le modèle
categorical_column = X.select_dtypes(include=['object', 'category']).columns
y = pd.get_dummies(y)
X = pd.get_dummies(X, columns=categorical_column)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# COMMAND ----------

# MAGIC %md
# MAGIC # V. 1/  Optimisation Hyperparameters / Train model (Optionnelle)

# COMMAND ----------

# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import uniform, randint

# # def optimize_and_train_model(X_train, y_train):
# #     xgb_model = xgb.XGBClassifier()

# #     params = {
# #         "n_estimators": randint(50, 200),
# #         "learning_rate": uniform(0.01, 0.3),
# #         "max_depth": randint(3, 10),
# #         "subsample": uniform(0.6, 0.4),
# #         "colsample_bytree": uniform(0.6, 0.4),
# #     }

# #     random_search = RandomizedSearchCV(xgb_model, param_distributions=params, n_iter=50, cv=3, verbose=1, n_jobs=-1, random_state=42)
# #     random_search.fit(X_train, y_train)

# #     return random_search.best_estimator_

# def optimize_and_train_model(X_train, y_train):
#     xgb_model = xgb.XGBClassifier()

#     params = {
#         "n_estimators": randint(80, 150),
#         "learning_rate": uniform(0.04, 0.2),
#         "max_depth": randint(5, 10),
#         "subsample": uniform(0.6, 0.4),
#         "colsample_bytree": uniform(0.6, 0.4),
#     }

#     random_search = RandomizedSearchCV(xgb_model, param_distributions=params, n_iter=20, cv=3, verbose=1, n_jobs=-1, random_state=42)
#     random_search.fit(X_train, y_train)

#     return random_search.best_estimator_


# model = optimize_and_train_model(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test avec un GridSearch mais modifier avec un tqdm, il s'agit d'une autre méthode qmais similaire du RandomizedSearchCV

# COMMAND ----------

# from sklearn.model_selection import ParameterGrid, StratifiedKFold, GridSearchCV
# from sklearn.metrics import make_scorer, accuracy_score
# from tqdm.notebook import tqdm
# import xgboost as xgb

# # Redéfinir tqdm pour une meilleure intégration avec GridSearchCV
# class TqdmGridSearchCV(GridSearchCV):
#     def _run_search(self, evaluate_candidates):
#         """Cette méthode remplace la méthode _run_search de GridSearchCV pour utiliser tqdm."""
#         evaluate_candidates(list(ParameterGrid(self.param_grid)))

# # Définir les hyperparamètres pour la recherche
# param_grid = {
#     "n_estimators": [100, 130],
#     "learning_rate": [0.1, 0.15],
#     "max_depth": [6, 9],
#     "subsample": [0.9]
# }

# # Créer et configurer le modèle XGBoost
# xgb_model = xgb.XGBClassifier(num_class=7, objective='binary:logistic')

# # Configurer GridSearchCV avec StratifiedKFold
# cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42) # Assurez-vous que le nombre de splits correspond à votre besoin

# # Configurer GridSearchCV
# grid_search = TqdmGridSearchCV(xgb_model, param_grid, scoring=make_scorer(accuracy_score), cv=cv, verbose=3)


# COMMAND ----------

# # Exécuter la recherche d'hyperparamètres
# model = grid_search.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC # V. 2/ Train model (Optionnelle depend if you launch Optimizer Hyperparams)

# COMMAND ----------

def train_model(X_train, y_train, X_test, y_test):
    # model = xgb.XGBClassifier(num_class=7, n_estimators=150, learning_rate=0.1, max_depth=7)
    model = xgb.XGBClassifier(n_estimators=200, early_stopping_rounds=10)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    # model.fit(X_train, y_train)
    return model


# COMMAND ----------

model = train_model(X_train, y_train, X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # VI. Evaluate Model

# COMMAND ----------

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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



# COMMAND ----------

# MAGIC %md
# MAGIC # Analyse de l'Importance des Caractéristiques

# COMMAND ----------

import matplotlib.pyplot as plt

def plot_feature_importance(model, X_train):
    # Récupération de l'importance des fonctionnalités
    feature_importances = model.feature_importances_
    
    # Création d'un DataFrame pour faciliter l'affichage
    features_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)

    # Affichage de l'importance des fonctionnalités
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=features_df)
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

plot_feature_importance(model, X_train)

# COMMAND ----------

import matplotlib as plt
# Récupérer l'importance des caractéristiques
importances = model.feature_importances_

# Trier les caractéristiques par importance
indices = np.argsort(importances)[::-1]

# Afficher le classement des caractéristiques
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print(f"{f + 1}. feature {X_train.columns[indices[f]]} ({importances[indices[f]]:.4f})")

# Tracer l'importance des caractéristiques dans un graphique à barres horizontales
plt.figure(figsize=(10, 10))
plt.title("Feature importances")
bars = plt.barh(range(X_train.shape[1]), importances[indices], color="r", align="center")

# Ajouter des étiquettes pour chaque barre
for bar in bars:
    plt.text(
        bar.get_width(),  # Position x de l'étiquette
        bar.get_y() + bar.get_height() / 2,  # Position y de l'étiquette
        f'{bar.get_width():.2f}',  # Valeur de l'étiquette
        va='center',  # Centrer verticalement
        ha='left'  # Aligner horizontalement à gauche
    )

plt.yticks(range(X_train.shape[1]), X_train.columns[indices])
plt.gca().invert_yaxis()  # Inverser l'axe y
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# COMMAND ----------


