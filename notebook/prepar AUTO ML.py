# Databricks notebook source
# Lecture des fichiers CSV
df1 = spark.read.csv("/mnt/data/train.csv", header=True, inferSchema=True)
df2 = spark.read.csv("/mnt/data/test.csv", header=True, inferSchema=True)
data_val = spark.read.csv("/mnt/data/val.csv", header=True, inferSchema=True)


# COMMAND ----------

# Liste des colonnes à conserver
colonnes_a_conserver = ['Classe_altitude', 'Conso_5_usages/m²_é_finale',
                       'Etiquette_DPE', 'Hauteur_sous-plafond', 'N°_département_(BAN)',
                       'Qualité_isolation_enveloppe', 'Qualité_isolation_menuiseries',
                       'Qualité_isolation_murs', 'Surface_habitable_logement',
                       'Type_bâtiment']

# Sélection des colonnes spécifiées
df1 = df1.select(colonnes_a_conserver)
df1.write.format("parquet").mode("overwrite").saveAsTable("train_table_clean_cols_V2")

df1_lim_lines = df1.limit(1250000)
df1_lim_lines.write.format("parquet").mode("overwrite").saveAsTable("train_table_clean_cols_1250000")

# COMMAND ----------

df = spark.read.table("train_table_clean_cols_1250000").toPandas()

# Obtenir le nombre de lignes
nombre_de_lignes = len(df)

print("Nombre de lignes dans le DataFrame :", nombre_de_lignes)

# COMMAND ----------


