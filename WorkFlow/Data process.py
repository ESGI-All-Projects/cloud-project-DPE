# Databricks notebook source
# MAGIC %md
# MAGIC # Import Lib

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC # II. Load Data & Process
# MAGIC
# MAGIC
# MAGIC Cette fonction effectue les étapes suivantes :
# MAGIC
# MAGIC - Supprime les valeurs aberrantes.
# MAGIC - Supprime les colonnes avec un pourcentage élevé de valeurs manquantes.
# MAGIC - Supprime des colonnes spécifiques.
# MAGIC - Combine les données train et test.

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

# Chargement des données (adaptez le chemin selon vos fichiers)
train_data = spark.read.csv("/mnt/data/train.csv", header=True, inferSchema=True)
test_data = spark.read.csv("/mnt/data/test.csv", header=True, inferSchema=True)

# Combinaison et prétraitement des données
data = train_data.union(test_data)
# data = all_data.sample(withReplacement=False, fraction=0.10, seed=42)

data = clean_data(data.toPandas())

# COMMAND ----------

# MAGIC %md
# MAGIC #III. Save file as Table

# COMMAND ----------

spark_data = spark.createDataFrame(data)
spark_data.write.format("parquet").mode("overwrite").saveAsTable("train_processed_table")
