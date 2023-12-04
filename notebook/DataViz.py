# Databricks notebook source
import pandas as pd

# COMMAND ----------

csv_file_path_train = "/dbfs/mnt/data/train.csv"
df = pd.read_csv(csv_file_path_train)

# COMMAND ----------

count_by_label = df.groupby('Etiquette_DPE')['Etiquette_DPE'].count()
print(count_by_label)
print(count_by_label.sum())


# COMMAND ----------

df_sample = df.groupby('Etiquette_DPE', group_keys=False).apply(lambda x: x.sample(12470))
print(len(df_sample))

# COMMAND ----------


