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
streamlit run app.py
```

UI opens at `http://localhost:8501` (or similar).

### 🌐 Deploy to Streamlit Community Cloud

1. **Make Repository Public** (if not already):
   - Go to your GitHub repository
   - Settings → General → Danger Zone → Change repository visibility
   - Set to **Public**

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select repository: `bhaktiichandak/energy-consumption-forecasting-spark`
   - Main file path: `app.py`
   - Click **Deploy**

3. **Alternative: Deploy from Local**:
   - The repository is now configured for Streamlit Cloud
   - `app.py` is in the root directory
   - `requirements.txt` and `packages.txt` are ready
   - Repository is connected to GitHub

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
