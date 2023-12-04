# Databricks notebook source
dbutils.fs.mount(
source = "wasbs://data-dpe@esgigroupe3a.blob.core.windows.net", 
mount_point = "/mnt/data",
extra_configs = {"fs.azure.account.key.esgigroupe3a.blob.core.windows.net":dbutils.secrets.get(scope = "esgi-groupe3-scope", key = "esgi-groupe3-key")})

# COMMAND ----------

df = spark.read.csv("/mnt/data/train.csv", header=True)
df = df.drop("_c0")

df = df.withColumnRenamed("N°_département_(BAN)","N°_département_BAN")
df = df.withColumnRenamed("Code_postal_(BAN)","Code_postal_BAN")
df = df.withColumnRenamed("Nom__commune_(Brut)","Nom__commune_Brut")
df = df.withColumnRenamed("Code_INSEE_(BAN)","Code_INSEE_BAN")
df = df.withColumnRenamed("Code_postal_(brut)","Code_postal_brut")

print(df.columns)
# name_file = file.name.split(".")[0]
# df.write.option("header", True).mode("overwrite").saveAsTable(name_file)

# COMMAND ----------

files = dbutils.fs.ls("/mnt/data")

for file in files:
    # Vérifiez si le fichier est un fichier CSV par exemple
    if file.name.endswith('.csv'):
        # Lecture du fichier CSV dans un DataFrame Spark
        df = spark.read.option("header", "true").csv(file.path)
        df= df.drop("_c0")

        df = df.withColumnRenamed("N°_département_(BAN)","N°_département_BAN")
        df = df.withColumnRenamed("Code_postal_(BAN)","Code_postal_BAN")
        df = df.withColumnRenamed("Nom__commune_(Brut)","Nom__commune_Brut")
        df = df.withColumnRenamed("Code_INSEE_(BAN)","Code_INSEE_BAN")
        df = df.withColumnRenamed("Code_postal_(brut)","Code_postal_brut")

        name_file = file.name.split(".")[0]
        df.write.option("header", True).mode("overwrite").saveAsTable(name_file)


# COMMAND ----------

files = dbutils.fs.ls("/mnt/data")

for file in files:
    # Vérifiez si le fichier est un fichier CSV par exemple
    if file.name == "train.csv":
        # Lecture du fichier CSV dans un DataFrame Spark
        df = spark.read.option("header", "true").csv(file.path)
        df= df.drop("_c0")

        df = df.withColumnRenamed("N°_département_(BAN)","N°_département_BAN")
        df = df.withColumnRenamed("Code_postal_(BAN)","Code_postal_BAN")
        df = df.withColumnRenamed("Nom__commune_(Brut)","Nom__commune_Brut")
        df = df.withColumnRenamed("Code_INSEE_(BAN)","Code_INSEE_BAN")
        df = df.withColumnRenamed("Code_postal_(brut)","Code_postal_brut")

        name_file = file.name.split(".")[0]
        df.write.option("header", True).mode("overwrite").saveAsTable(name_file)
