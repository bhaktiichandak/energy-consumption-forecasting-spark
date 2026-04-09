from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofmonth, to_date
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import os
import json
import shutil

# Create a Spark session for training with Windows Hadoop workaround
spark = SparkSession.builder \
    .appName("EnergyForecastTraining") \
    .master("local[*]") \
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
    .config("spark.sql.shuffle.partitions", "1") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("Loading data into Spark DataFrame...")
df = spark.read.option("header", "true").option("sep", ";").csv("data/energy.csv")

print("Cleaning and preprocessing data...")
clean_df = df.replace("?", None) \
    .withColumn("Voltage", col("Voltage").cast("double")) \
    .withColumn("Global_intensity", col("Global_intensity").cast("double")) \
    .withColumn("Sub_metering_1", col("Sub_metering_1").cast("double")) \
    .withColumn("Global_active_power", col("Global_active_power").cast("double"))

clean_df = clean_df.dropna()

print(f"Records after cleaning: {clean_df.count()}")

# Feature engineering with proper date parsing
print("Adding feature engineering...")
# Parse Date with D/M/YYYY format (handles both 4/4/2010 and 20/06/2007), then extract day
# Parse Time to extract hour
clean_df = clean_df.withColumn("Date_parsed", to_date(col("Date"), "d/M/yyyy")) \
    .withColumn("Hour", hour(col("Time"))) \
    .withColumn("Day", dayofmonth(col("Date_parsed")))

# Select relevant features for the model
feature_columns = ["Voltage", "Global_intensity", "Sub_metering_1", "Hour", "Day"]

# Assemble features into a vector column
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(clean_df).select("features", "Global_active_power")

# Train-test split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

print("Training Spark MLlib Linear Regression model...")
lr = LinearRegression(featuresCol="features", labelCol="Global_active_power")
model = lr.fit(train_data)

# Evaluate the model
predictions = model.transform(test_data)
evaluator = RegressionEvaluator(labelCol="Global_active_power", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Model trained! RMSE: {rmse:.4f}")

# Save model metadata for reference (coefficients and intercept)
model_dir = "spark_model"
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
os.makedirs(model_dir, exist_ok=True)

# Extract and save model parameters as JSON for reference
model_data = {
    "coefficients": model.coefficients.toArray().tolist(),
    "intercept": float(model.intercept),
    "features": feature_columns,
    "rmse": float(rmse)
}

with open(os.path.join(model_dir, "model_params.json"), "w") as f:
    json.dump(model_data, f, indent=2)

print("Model parameters saved to spark_model/model_params.json")

# Stop Spark session
spark.stop()