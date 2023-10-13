import pandas as pd
import numpy as np
import joblib

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

df = pd.read_csv("data/val.csv")

id = ["N°DPE"]
df_filter = df[features_selection_quantitatif + features_selection_qualitative + id]
df_encoded = pd.get_dummies(df_filter, columns=features_selection_qualitative)

list_of_id = df_encoded[id]
X = df_encoded.drop(id, axis=1)

model = joblib.load("model/xgboost_model.sav")
prediction = model.predict(X)
col_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
prediction = [''.join(col_names[np.argmax(row)]) for row in prediction]
# build submission file
data = {
    "N°DPE": list_of_id["N°DPE"].to_list(),
    "Etiquette_DPE": prediction
}
df_soumission = pd.DataFrame(data=data)
df_soumission.to_csv("data/submission_xgboost.csv", index=False)