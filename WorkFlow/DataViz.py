# Databricks notebook source
# MAGIC %md
# MAGIC # I. Import Libs

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC # II. Import Data

# COMMAND ----------

train_df = spark.table("train_table").toPandas()


# COMMAND ----------

# MAGIC %md
# MAGIC # III. Visualisation raw_data

# COMMAND ----------

# MAGIC %md
# MAGIC ## valeurs aberrantes

# COMMAND ----------

# Liste des colonnes pour lesquelles générer l'histogramme individuellement
columns = ['Hauteur_sous-plafond', 'Conso_5_usages/m²_é_finale', 'Surface_habitable_logement']

for col in columns:
    # Créer la figure et l'axe
    plt.figure(figsize=(10, 6))
    
    # Générer l'histogramme avec une échelle logarithmique pour l'axe des y
    counts, bins, patches = plt.hist(train_df[col].dropna(), bins=50, log=True, color='skyblue', edgecolor='black')
    
    # Ajouter des annotations sur chaque bin
    for count, bin, patch in zip(counts, bins, patches):
        if count > 0:  # Annoter seulement si le compte est supérieur à 0
            # Positionner le texte au centre du patch (barre du histogramme)
            x_value = patch.get_x() + patch.get_width() / 2
            y_value = patch.get_height()
            plt.text(x_value, y_value, f'{int(count)}', ha='center', va='bottom', fontsize=8)

    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency (log scale)')
    plt.yscale('log')  # Définir l'échelle des y en logarithmique pour l'affichage
    plt.grid(True)
    plt.show()  # Afficher la figure

# COMMAND ----------

# MAGIC %md
# MAGIC ## Valeurs NaN

# COMMAND ----------

# Calculer le pourcentage de valeurs NaN dans chaque colonne
nan_percentage = train_df.isnull().mean().sort_values(ascending=False) * 100

# Transformer en DataFrame pour la visualisation
train_nan_df = pd.DataFrame(nan_percentage, columns=['Pourcentage de NaN'])

plt.figure(figsize=(12, 8))
sns.heatmap(train_nan_df, annot=True, fmt=".2f", cmap='viridis', cbar=True)
plt.title('Pourcentage de valeurs NaN par colonne dans train_df')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Répartition Etiquette_DPE

# COMMAND ----------

# Créer un countplot pour la colonne 'Etiquette_DPE'
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Etiquette_DPE', data=train_df)
plt.title('Répartition des valeurs dans la colonne Etiquette_DPE')

# Ajouter des annotations pour afficher le nombre de comptes pour chaque catégorie
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.show()


# COMMAND ----------



# COMMAND ----------

# MAGIC %md ==========================================================================================================================================================================
# MAGIC # IV. Cleaned Data

# COMMAND ----------

#Import Data

train_clean_df = spark.table("train_processed_table").toPandas()
display(train_clean_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Valeurs aberrantes

# COMMAND ----------

train_clean_df["Hauteur_sous-plafond"].unique()

# COMMAND ----------

# Liste des colonnes pour lesquelles générer l'histogramme individuellement
columns = ['Hauteur_sous-plafond', 'Conso_5_usages/m²_é_finale', 'Surface_habitable_logement']

for col in columns:
    # Créer la figure et l'axe
    plt.figure(figsize=(10, 6))
    
    # Générer l'histogramme avec une échelle logarithmique pour l'axe des y
    counts, bins, patches = plt.hist(train_clean_df[col].dropna(), bins=50, log=True, color='skyblue', edgecolor='black')
    
    # Ajouter des annotations sur chaque bin
    for count, bin, patch in zip(counts, bins, patches):
        if count > 0:  # Annoter seulement si le compte est supérieur à 0
            # Positionner le texte au centre du patch (barre du histogramme)
            x_value = patch.get_x() + patch.get_width() / 2
            y_value = patch.get_height()
            plt.text(x_value, y_value, f'{int(count)}', ha='center', va='bottom', fontsize=8)

    plt.title(f'Distribution de {col}')
    plt.xlabel(col)
    plt.ylabel('Fréquence (log scale)')
    plt.yscale('log')  # Définir l'échelle des y en logarithmique pour l'affichage
    plt.grid(True)
    plt.show()  # Afficher la figure

# COMMAND ----------

# MAGIC %md
# MAGIC ## Valeurs NaN

# COMMAND ----------

# Calculer le pourcentage de valeurs NaN dans chaque colonne
nan_percentage = train_clean_df.isnull().mean().sort_values(ascending=False) * 100

# Transformer en DataFrame pour la visualisation
train_nan_df = pd.DataFrame(nan_percentage, columns=['Pourcentage de NaN'])

plt.figure(figsize=(12, 8))
sns.heatmap(train_nan_df, annot=True, fmt=".2f", cmap='viridis', cbar=True)
plt.title('Pourcentage de valeurs NaN par colonne dans train_clean_df')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # IV. Univariate plot

# COMMAND ----------

# Séparation en données numériques et catégorielles
numerical_data = train_clean_df.select_dtypes(include=['int64', 'float64'])
categorical_data = train_clean_df.select_dtypes(include=['string', 'object'])
train_clean_df = spark.createDataFrame(train_clean_df)


# COMMAND ----------

from pyspark.sql.functions import col, lit, when

top_n = 10

for var in categorical_data.columns:
    top_modalities = train_clean_df.groupBy(var).count().orderBy('count', ascending=False).limit(top_n).select(var).rdd.flatMap(lambda x: x).collect()
    train_clean_df = train_clean_df.withColumn(var + "_top_n", when(col(var).isin(top_modalities), col(var)).otherwise(lit('Autres')))

# COMMAND ----------

# Configure the number of rows and columns based on the number of categorical variables
n_vars = len(categorical_data.columns)
n_cols = 2
n_rows = (n_vars + n_cols - 1) // n_cols  # Ceiling division to get enough rows

# Adjust the figure size dynamically based on the number of rows
plt.figure(figsize=(20, 5 * n_rows))

for i, var in enumerate(categorical_data.columns):
    ax = plt.subplot(n_rows, n_cols, i + 1)
    data_pd = train_clean_df.groupBy(var + "_top_n").count().orderBy('count', ascending=False).toPandas()
    
    sns.barplot(x='count', y=var + "_top_n", data=data_pd, palette='viridis')
    plt.title(f'Distribution of {var}', fontsize=14)
    
    ax.set_yticklabels([label.get_text()[:30] + '...' if len(label.get_text()) > 30 else label.get_text() for label in ax.get_yticklabels()], fontsize=10)
    
    # Formatting the count labels to include thousand separator and avoid overlap by adjusting text position
    for p in ax.patches:
        width = p.get_width()
        plt.text(width, p.get_y() + p.get_height() / 2, f'{int(width):,}', ha='left', va='center', fontsize=10)

# Adjust the layout
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## V. Bivariate Plots

# COMMAND ----------

from pyspark.sql.functions import col, when, lit, expr
from pyspark.sql import functions as F
import matplotlib.pyplot as plt

def plot_stacked_bar(data, variable_analyse, variable_cible, top_n=10):
    top_modalities = data.groupBy(variable_analyse).count().orderBy('count', ascending=False).limit(top_n).select(variable_analyse).rdd.flatMap(lambda x: x).collect()
    data_top_n = data.withColumn(variable_analyse, when(col(variable_analyse).isin(top_modalities), col(variable_analyse)).otherwise(lit('Autres')))
    pivot_data = data_top_n.groupBy(variable_analyse).pivot(variable_cible).count()
    pivot_data = pivot_data.fillna(0)

    columns_to_sum = [col(c) for c in pivot_data.columns if c != variable_analyse]

    pivot_data = pivot_data.select("*", expr("A + B + C + D + E + F + G as total"))

    for c in pivot_data.columns:
        if c != variable_analyse and c != 'total':
            pivot_data = pivot_data.withColumn(c, (col(c) / col('total')) * 100)       
    pivot_percent_pd = pivot_data.toPandas().set_index(variable_analyse)
    ax = pivot_percent_pd.drop('total', axis=1).plot(kind='bar', stacked=True, figsize=(10, 6))
    
    plt.title(f'Distribution de {variable_analyse} par {variable_cible} (en %)')
    plt.xlabel(f'{variable_analyse}')
    plt.ylabel('Pourcentage')
    plt.xticks(rotation=0)
    plt.legend(title=f'{variable_cible}', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy() 
        if height > 3:
            ax.text(x + width/2, y + height/2, '{:1.1f}%'.format(height), ha='center', va='center', fontsize=8)

    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Classe altitude

# COMMAND ----------

plot_stacked_bar(train_clean_df, "Classe_altitude", "Etiquette_DPE", top_n=10)

# COMMAND ----------

# MAGIC %md
# MAGIC  Qualité isolation enveloppe

# COMMAND ----------

plot_stacked_bar(train_clean_df, "Qualité_isolation_enveloppe", "Etiquette_DPE", top_n=10)

# COMMAND ----------

# MAGIC %md
# MAGIC  Qualité isolation murs

# COMMAND ----------

plot_stacked_bar(train_clean_df, "Qualité_isolation_murs", "Etiquette_DPE", top_n=10)

# COMMAND ----------

# MAGIC %md
# MAGIC  Qualité isolation menuiseries

# COMMAND ----------

plot_stacked_bar(train_clean_df, "Qualité_isolation_menuiseries", "Etiquette_DPE", top_n=10)

# COMMAND ----------

# MAGIC %md
# MAGIC  Type bâtiment

# COMMAND ----------

plot_stacked_bar(train_clean_df, "Type_bâtiment", "Etiquette_DPE", top_n=10)

# COMMAND ----------



# COMMAND ----------


