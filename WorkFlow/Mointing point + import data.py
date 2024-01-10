# Databricks notebook source
# MAGIC %md
# MAGIC # Mounting Point

# COMMAND ----------

try:
    dbutils.fs.mount(
    source = "wasbs://data-dpe@esgigroupe3a.blob.core.windows.net", 
    mount_point = "/mnt/data",
    extra_configs = {"fs.azure.account.key.esgigroupe3a.blob.core.windows.net":dbutils.secrets.get(scope = "esgi-groupe3-scope", key = "esgi-groupe3-key")})
    
    print("Montage réussi.")
except Exception as e:
    print("Une erreur s'est produite lors du montage:", e)

# COMMAND ----------

# MAGIC %md
# MAGIC # Import Data

# COMMAND ----------

train_data = spark.read.csv("/mnt/data/train.csv", header=True, inferSchema=True)
test_data = spark.read.csv("/mnt/data/test.csv", header=True, inferSchema=True)
val_data = spark.read.csv("/mnt/data/val.csv", header=True, inferSchema=True)

# COMMAND ----------

train_data.write.format("parquet").mode("overwrite").saveAsTable("train_table")
test_data.write.format("parquet").mode("overwrite").saveAsTable("test_data")
val_data.write.format("parquet").mode("overwrite").saveAsTable("val_data")

# COMMAND ----------


