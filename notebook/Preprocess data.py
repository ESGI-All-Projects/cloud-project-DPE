# Databricks notebook source
import pandas as pd

# COMMAND ----------

df = spark.read.table("train_250000rows").toPandas()
display(df)

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

df_filter = df[features_selection_quantitatif + features_selection_qualitative + id + target]
df_encoded = pd.get_dummies(df_filter, columns=features_selection_qualitative)
df_encoded = df_encoded.sample(frac=1)

display(df_encoded) #PRE_PROCESS_TRAIN_DATA
# y = df_encoded[target]
# X = df_encoded.drop(id + target, axis=1)
# y = pd.get_dummies(y)

# COMMAND ----------

sparkDF=spark.createDataFrame(df_encoded) 
sparkDF = sparkDF.withColumnRenamed("Qualité_isolation_enveloppe_très bonne","Qualité_isolation_enveloppe_très_bonne")
sparkDF = sparkDF.withColumnRenamed("Qualité_isolation_menuiseries_très bonne","Qualité_isolation_menuiseries_très_bonne")
sparkDF = sparkDF.withColumnRenamed("Qualité_isolation_plancher_bas_très bonne","Qualité_isolation_plancher_bas_très_bonne")
sparkDF = sparkDF.withColumnRenamed("Qualité_isolation_murs_très bonne","Qualité_isolation_murs_très_bonne")

sparkDF = sparkDF.withColumnRenamed("Classe_altitude_inférieur à 400m","Classe_altitude_inférieur_à_400m")
sparkDF = sparkDF.withColumnRenamed("Classe_altitude_supérieur à 800m","Classe_altitude_supérieur_à_800m:")



# COMMAND ----------

sparkDF.write.option("header", True).mode("overwrite").saveAsTable("preprocess_train")

# COMMAND ----------


