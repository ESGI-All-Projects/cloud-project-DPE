# Databricks notebook source
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import joblib


# COMMAND ----------

csv_file_path_train = "/dbfs/mnt/data/train.csv"

# df = pd.read_csv("data/train.csv").sample(frac=0.1)
# df.to_csv("data/train_partial.csv", header=True, index=False)

df = pd.read_csv(csv_file_path_train)
display(df)

# df = pd.read_csv()
# df_test = pd.read_csv("data/test.csv")
# df_val = pd.read_csv("data/val.csv")
print("taille : ", len(df))

# COMMAND ----------

for column in df.columns:
    print(df[column].isna().sum()," : ",column)

features_selection_quantitatif = ["Hauteur_sous-plafond",
                                  "Emission_GES_éclairage",
                                  "Surface_habitable_logement",
                                  "Conso_5_usages/m²_é_finale",
                                  "Conso_5_usages_é_finale",
                                  "Année_construction",
                                  "Conso_5_usages_é_finale_énergie_n°2",
                                  "Surface_habitable_desservie_par_installation_ECS"]
features_selection_qualitative = ["Type_bâtiment",
                                  "Qualité_isolation_enveloppe",
                                  "Qualité_isolation_menuiseries",
                                  "Etiquette_GES",
                                  "Qualité_isolation_plancher_bas",
                                  "Qualité_isolation_murs",
                                  "Classe_altitude"]
id = ["N°DPE"]
target = ["Etiquette_DPE"]


# COMMAND ----------

# MAGIC %md # train
# MAGIC

# COMMAND ----------

df_filter = df[features_selection_quantitatif + features_selection_qualitative + id + target]
df_encoded = pd.get_dummies(df_filter, columns=features_selection_qualitative)
df_encoded = df_encoded.sample(frac=1)
y = df_encoded[target]
X = df_encoded.drop(id + target, axis=1)
y = pd.get_dummies(y)


# COMMAND ----------

# MAGIC %md # test
# MAGIC

# COMMAND ----------

# df_filter_val = df_test[features_selection_quantitatif + features_selection_qualitative + id + target]
# df_encoded_val = pd.get_dummies(df_filter_val, columns=features_selection_qualitative)
# df_encoded_val = df_encoded_val.sample(frac=1)
# X_test = df_encoded_val[features_selection_quantitatif + features_selection_qualitative]
# y_test = df_encoded_val[target]
# y_test = pd.get_dummies(y_val)


# COMMAND ----------

# k = 5
# kf = KFold(n_splits=k, shuffle=True, random_state=42)
# scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")

# print("Scores de validation croisée en k-fold :", scores)
# print("Moyenne des scores :", scores.mean())
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(n_estimators=200)
model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)], verbose=True)
joblib.dump(model, "model/xgboost_model.sav")


# COMMAND ----------

# MAGIC %md #Predict
# MAGIC

# COMMAND ----------

prediction = model.predict(X_val)
accuracy = accuracy_score(y_val, prediction)
print(accuracy)


# COMMAND ----------

#df_filter = df[features_selection_quantitatif].dropna()
#matrix_corr = df_filter.corr()
#plt.figure(figsize=(10, 8))
#sns.heatmap(matrix_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
#plt.title('Matrice de Corrélation')
#plt.show()

# COMMAND ----------


