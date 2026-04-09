# ⚡ Energy Consumption Forecasting using Spark MLlib

A machine learning project that demonstrates **Apache Spark MLlib Linear Regression** for predicting household electricity consumption patterns. Built with a clean Streamlit UI for easy viva demonstration.

## 📋 Project Overview

**Topic:** Energy Consumption Forecasting using Spark MLlib  
**Algorithm:** Linear Regression (MLlib)  
**Dataset:** UCI Individual Household Electric Power Consumption (~2.05M records)  
**Framework:** Apache Spark  
**Language:** Python 3.11

### Key Concepts Demonstrated

- ✅ **Spark DataFrames** - Data loading, cleaning, and transformations using Spark SQL functions
- ✅ **Feature Engineering** - Extracting hour and day features from timestamp columns  
- ✅ **MLlib Linear Regression** - Supervised learning for continuous value prediction
- ✅ **Train-Test Split** - 80/20 split with model evaluation using RMSE metric
- ✅ **Data Preprocessing** - Handling missing values, type casting, and normalization

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or later
- Java 17 (automatically installed if not present)
- 8GB+ RAM recommended

### Installation

1. **Clone and navigate to project:**
   ```bash
   git clone https://github.com/bhaktiichandak/energy-consumption-forecasting-spark.git
   cd energy-consumption-forecasting-spark
   ```

2. **Download the dataset:**
   ```bash
   # Create data directory
   mkdir -p data
   
   # Download from UCI repository
   curl -o data/energy.csv.zip "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
   
   # Extract the dataset
   unzip data/energy.csv.zip -d data/
   mv data/household_power_consumption.txt data/energy.csv
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Run the Project

#### Step 1: Train the Model
```bash
python train_model.py
```

**Expected Output:**
```
Loading data into Spark DataFrame...
Cleaning and preprocessing data...
Records after cleaning: 2049280
Adding feature engineering...
Training Spark MLlib Linear Regression model...
Model trained! RMSE: 0.0479
Model parameters saved to spark_model/model_params.json
```

#### Step 2: Run the Streamlit UI
```bash
streamlit run ui/app.py
```

UI opens at `http://localhost:8502` (or similar).

## 📊 Data Description

### Dataset Details
- **Source:** UCI Machine Learning Repository
- **Records:** 2,049,280 (after cleaning)
- **Format:** CSV with semicolon separator
- **Date Range:** Dec 2006 - Nov 2010

### Features Used in Model
- **Voltage** (V): 223.2 - 254.0
- **Global_intensity** (A): 0.2 - 48.4
- **Sub_metering_1** (Wh): 0 - 30+
- **Hour** (engineered): 0-23
- **Day** (engineered): 1-31

### Target Variable
- **Global_active_power** (kW): Predicted electricity usage

## 🧮 Model Architecture

**Linear Regression Equation:**
```
ŷ = w₁·Voltage + w₂·Global_intensity + w₃·Sub_metering_1 + w₄·Hour + w₅·Day + b
```

**Model Performance:**
- Training RMSE: 0.0479 kW
- Train/Test Split: 80/20
- Test Set Size: 409,856 records

## 📁 Project Structure

```
energy-forecast/
├── train_model.py               # Spark MLlib training script
├── ui/
│   └── app.py                   # Streamlit prediction interface
├── data/
│   └── energy.csv               # Dataset (~2.05M records)
├── spark_model/
│   └── model_params.json        # Saved model coefficients
├── requirements.txt             # Dependencies
└── README.md
```

## 🔑 Key Concepts (for Viva)

### 1. Spark Session
```python
spark = SparkSession.builder.appName("EnergyForecastTraining").master("local[*]").getOrCreate()
```
Entry point for Spark DataFrames and MLlib APIs.

### 2. DataFrame Transformations
```python
clean_df = clean_df.replace("?", None) \
    .withColumn("Voltage", col("Voltage").cast("double")) \
    .dropna()
```
Chained operations for ETL (Extract, Transform, Load).

### 3. Date Parsing & Feature Engineering
```python
clean_df = clean_df.withColumn("Date_parsed", to_date(col("Date"), "d/M/yyyy")) \
    .withColumn("Hour", hour(col("Time"))) \
    .withColumn("Day", dayofmonth(col("Date_parsed")))
```
Spark SQL functions for time-series feature extraction. Format `d/M/yyyy` handles dates without leading zeros.

### 4. MLlib Feature Assembly
```python
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=["Voltage", "Global_intensity", ...], outputCol="features")
```
Combines multiple columns into a feature vector for MLlib.

### 5. Train-Test Split
```python
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
```
80% training, 20% test with reproducible random seed.

### 6. Linear Regression Training
```python
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol="features", labelCol="Global_active_power")
model = lr.fit(train_data)
```
MLlib class for regression. `.fit()` uses distributed computing.

### 7. Model Evaluation
```python
evaluator = RegressionEvaluator(labelCol="Global_active_power", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(model.transform(test_data))
```
Evaluates performance using RMSE metric on unseen test data.

## 💡 Viva Tips

**Q: Why use Spark instead of scikit-learn?**  
A: For distributed machine learning. Spark parallelizes across cores/machines, essential for large datasets. This project demonstrates Spark's distributed computing capabilities.

**Q: Explain the date parsing fix?**  
A: CSV had variable-format dates like "4/4/2010" and "20/6/2007". Format `dd/MM/yyyy` expected leading zeros, failing on "4/4/2010". Solution: Use `d/M/yyyy` to match variable-length months/days.

**Q: What does RMSE 0.0479 mean?**  
A: Model predictions deviate by ±0.0479 kW on average. Lower RMSE = better predictions.

**Q: How does the UI predict without retraining?**  
A: Saves model coefficients & intercept as JSON. UI loads these and applies: `y = Σ(coefficient × feature) + intercept`.

**Q: Why 80/20 split?**  
A: 80% trains the model with enough data to learn patterns. 20% test set is unseen data to measure generalization and prevent overfitting.

## 🛠️ Troubleshooting

### "Java not found"
Java 17 is at C:\Users\bhakt\.jdk. Verify:
```bash
echo %JAVA_HOME%
java -version
```

### "Model not found" in UI
Run training first:
```bash
python train_model.py
```

### Port already in use
Streamlit auto-assigns next port (8502, 8503, etc.). Check displayed URL.

## 📚 Study Topics

1. **Distributed Computing** - How Spark parallelizes data & computation
2. **DataFrame API** - SQL-like operations on distributed data
3. **MLlib Algorithms** - Linear regression with gradient descent on Spark
4. **Feature Engineering** - Transforming raw data into predictive features
5. **Model Evaluation** - Understanding regression metrics (RMSE)
6. **ML Deployment** - Serving models via web interfaces

---

**Status:** ✅ Ready for Viva Demonstration  
**Framework:** Apache Spark 3.x + Streamlit  
**Dataset:** UCI Energy Consumption (~2M records)

