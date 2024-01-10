# Databricks notebook source
# MAGIC %md
# MAGIC # Pred Model Metier

# COMMAND ----------

import mlflow
import pandas as pd

# # URI du modèle enregistré dans MLflow
logged_model = 'runs:/2832c457dee545e085607ff008792c75/model'

# Charger le modèle comme un PyFuncModel
loaded_model = mlflow.pyfunc.load_model(logged_model)

# COMMAND ----------

# Charger et prétraiter les données de validation
val_data = spark.read.csv("/mnt/data/val.csv", header=True, inferSchema=True).toPandas()

# Supprimer les colonnes spécifiées (adaptez ces listes en fonction de vos besoins)
cols_to_drop = ["N°DPE", "Code_INSEE_(BAN)", "Code_postal_(BAN)", "Conso_5_usages_é_finale", "Qualité_isolation_plancher_bas", "Code_postal_(brut)", "Nom__commune_(Brut)", "Emission_GES_éclairage"]

cols_to_drop_cat = ["Configuration_installation_chauffage_n°2", "Cage_d'escalier", "Type_générateur_froid", "Type_émetteur_installation_chauffage_n°2", "Type_énergie_n°3", "Type_générateur_n°1_installation_n°2", "Description_générateur_chauffage_n°2_installation_n°2",
            "Qualité_isolation_plancher_haut_comble_aménagé", "Qualité_isolation_plancher_haut_comble_perdu", "Qualité_isolation_plancher_haut_toit_terrase"] 

cols_to_drop_num = ["Facteur_couverture_solaire_saisi", "Surface_habitable_desservie_par_installation_ECS", "Conso_5_usages_é_finale_énergie_n°2", "Surface_totale_capteurs_photovoltaïque", "Conso_chauffage_dépensier_installation_chauffage_n°1", "Coût_chauffage_énergie_n°2",              
            "Emission_GES_chauffage_énergie_n°2", "Facteur_couverture_solaire", "Année_construction", "Surface_habitable_immeuble"]  

# Appliquer le même prétraitement que pour les données d"entraînement
val_data = val_data.drop(cols_to_drop + cols_to_drop_cat + cols_to_drop_num, axis=1)

print(val_data.columns)



# COMMAND ----------

# Convertir les colonnes catégorielles
for column in val_data.columns:
    val_data[column] = val_data[column].astype("category").cat.codes

# Prédire sur les données prétraitées
prediction_val = loaded_model.predict(val_data)

# Convertir les prédictions en DataFrame Pandas
prediction = pd.DataFrame(prediction_val, columns=["Etiquette_DPE"])
prediction["Etiquette_DPE"] = prediction["Etiquette_DPE"].astype("object")

# Remplacer les valeurs numériques par les étiquettes correspondantes
mapping = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G"}
prediction["Etiquette_DPE"] = prediction["Etiquette_DPE"].replace(mapping)


display(prediction)

# COMMAND ----------

df_soumission = pd.DataFrame()


val_data = spark.read.csv("/mnt/data/val.csv", header=True, inferSchema=True)
data_soumission = val_data.toPandas()

# Créer le DataFrame de soumission
df_soumission = pd.DataFrame()
if "N°DPE" in data_soumission.columns:
    df_soumission["N°DPE"] = data_soumission["N°DPE"]
    df_soumission["Etiquette_DPE"] = prediction['Etiquette_DPE']
else:
    print("La colonne 'N°DPE' n'est pas présente dans les données de validation.")


display(df_soumission)

# COMMAND ----------

df_soumission["N°DPE"] = df_soumission["N°DPE"].astype(str)

# Enregistrer le DataFrame de soumission dans un fichier CSV
df_soumission.to_csv("../Data/Soumissions_model_metier_V2.csv", index=False)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Pred Model Perf (auto ML=)

# COMMAND ----------

import mlflow
import pandas as pd

logged_model = 'runs:/d6fbce926dbd4c06b01d7e85979db56f/model'

# Charger le modèle comme un PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# COMMAND ----------

# Charger et préparer les données de validation
val_data = spark.read.csv("/mnt/data/val.csv", header=True, inferSchema=True).toPandas()

# Sélectionner les colonnes nécessaires pour la prédiction
prediction_data = val_data[[
    "N°DPE", "Etiquette_GES", "Classe_altitude", "Conso_5_usages/m²_é_finale", 
    "Hauteur_sous-plafond", "N°_département_(BAN)", "Qualité_isolation_enveloppe",
    "Qualité_isolation_menuiseries", "Qualité_isolation_murs", 
    "Surface_habitable_logement", "Type_bâtiment"
]]

# COMMAND ----------

# Prédire sur les données prétraitées
prediction_val = loaded_model.predict(prediction_data)

# Convertir les prédictions en DataFrame Pandas
prediction = pd.DataFrame(prediction_val, columns=['Etiquette_DPE'])

display(prediction)

# COMMAND ----------

df_soumission = pd.DataFrame()


val_data = spark.read.csv("/mnt/data/val.csv", header=True, inferSchema=True)
data_soumission = val_data.toPandas()

# Créer le DataFrame de soumission
df_soumission = pd.DataFrame()
if "N°DPE" in data_soumission.columns:
    df_soumission["N°DPE"] = data_soumission["N°DPE"]
    df_soumission["Etiquette_DPE"] = prediction['Etiquette_DPE']
else:
    print("La colonne 'N°DPE' n'est pas présente dans les données de validation.")


display(df_soumission)

# COMMAND ----------

df_soumission["N°DPE"] = df_soumission["N°DPE"].astype(str)

# Enregistrer le DataFrame de soumission dans un fichier CSV
df_soumission.to_csv("../Data/Soumissions_model_perf.csv", index=False)

# COMMAND ----------


