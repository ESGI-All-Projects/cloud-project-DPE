import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score



#df = pd.read_csv("data/train.csv").sample(frac=0.1)
#df.to_csv("data/train_partial.csv", header=True, index=False)

df = pd.read_csv("data/train_partial.csv")
df_val = pd.read_csv("data/val.csv")
print("taille : ", len(df))
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


# train
df_filter = df[features_selection_quantitatif + features_selection_qualitative + id + target]
df_encoded = pd.get_dummies(df_filter, columns=features_selection_qualitative)
df_encoded = df_encoded.sample(frac=1)
X = df_encoded[features_selection_quantitatif + features_selection_qualitative]
y = df_encoded[target]
y = pd.get_dummies(y)

# val
df_filter_val = df_val[features_selection_quantitatif + features_selection_qualitative + id + target]
df_encoded_val = pd.get_dummies(df_filter_val, columns=features_selection_qualitative)
df_encoded_val = df_encoded_val.sample(frac=1)
X_val = df_encoded_val[features_selection_quantitatif + features_selection_qualitative]
y_val = df_encoded_val[target]
y_val = pd.get_dummies(y_val)

model = XGBClassifier()
model.fit(X, y, early_stopping_rounds=10, eval_set=[(X_val, y_val)], verbose=False)


#df_filter = df[features_selection_quantitatif].dropna()
#matrix_corr = df_filter.corr()
#plt.figure(figsize=(10, 8))
#sns.heatmap(matrix_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
#plt.title('Matrice de Corrélation')
#plt.show()
