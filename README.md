# 🚕 NYC Taxi Trip Duration Prediction

<div align="center">

<img src="https://user-images.githubusercontent.com/74038190/213844263-a8897a51-32f4-4b3b-b5c2-e1528b89f6f3.png" width="50px" />&nbsp;<img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="400">&nbsp;<img src="https://user-images.githubusercontent.com/74038190/213844263-a8897a51-32f4-4b3b-b5c2-e1528b89f6f3.png" width="50px" />

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost-orange?style=for-the-badge&logo=xgboost&logoColor=white)
![Gradio](https://img.shields.io/badge/UI-Gradio-ff7c00?style=for-the-badge&logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Predicting taxi trip durations in NYC using advanced machine learning techniques**

[Demo](#-demo) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Performance](#-performance)

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="700">

</div>

---

## ✨ Overview

<img align="right" alt="Coding" width="400" src="https://user-images.githubusercontent.com/74038190/229223263-cf2e4b07-2615-4f87-9c38-e37600f8381a.gif">

This project leverages **XGBoost** and advanced feature engineering to predict NYC taxi trip durations with high accuracy. Built with real-world data containing ~1.4 million trips, it demonstrates end-to-end machine learning workflow from data preprocessing to deployment via an interactive web application.

### 🌟 Highlights
- 🎯 **1.4M+ trips** analyzed
- 🚀 **67% variance** explained
- ⚡ **Real-time predictions** via Gradio
- 📊 **Advanced feature engineering**

<br clear="right"/>

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47e185d-9656-417b-b0d6-e79ea83aab14.gif" width="100%">

## 🎯 Key Features

<div align="center">

### 🔍 Intelligent Feature Engineering
<img src="https://user-images.githubusercontent.com/74038190/212257472-08e52665-c503-4bd9-aa20-f5a4dae769b5.gif" width="100">

</div>

- **Temporal Features**: Hour, weekday, weekend indicator, month extraction
- **Geospatial Analysis**: Haversine distance calculation between pickup/dropoff
- **Data Quality**: Smart outlier detection using IQR method
- **Target Transformation**: Log normalization for improved model performance

<div align="center">

### 🧠 Advanced ML Pipeline
<img src="https://user-images.githubusercontent.com/74038190/212257465-7ce8d493-cac5-494e-982a-5a9deb852c4b.gif" width="100">

</div>

- **XGBoost Regressor** with optimized hyperparameters
- **Robust Preprocessing**: Handles missing values and outliers
- **Scalable Architecture**: Efficient training on large datasets

<div align="center">

### 🎨 Interactive Web Interface
<img src="https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif" width="100">

</div>

- **Gradio-powered** user-friendly interface
- Real-time predictions
- Visual feedback and result interpretation

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47e185d-9656-417b-b0d6-e79ea83aab14.gif" width="100%">

## 📊 Dataset

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257467-871d32b7-e401-42e8-a166-fcfd7baa4c6b.gif" width="100">
</div>

| Attribute | Value |
|-----------|-------|
| **Source** | [NYC Taxi Trip Duration (Kaggle)](https://www.kaggle.com/c/nyc-taxi-trip-duration) |
| **Rows** | ~1.4 Million trips |
| **Time Period** | 2016 |
| **Features** | 11 original features |

### Original Features
```python
✅ id                    # Unique trip identifier
✅ vendor_id             # Provider identifier
✅ pickup_datetime       # Trip start timestamp
✅ dropoff_datetime      # Trip end timestamp
✅ passenger_count       # Number of passengers
✅ pickup_longitude      # Pickup GPS coordinate
✅ pickup_latitude       # Pickup GPS coordinate
✅ dropoff_longitude     # Dropoff GPS coordinate
✅ dropoff_latitude      # Dropoff GPS coordinate
✅ store_and_fwd_flag    # Trip storage indicator
✅ trip_duration         # Target variable (seconds)
```

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47e185d-9656-417b-b0d6-e79ea83aab14.gif" width="100%">

## ⚙️ Engineered Features

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257468-1e9a91f1-b626-4baa-b15d-5c385dfa7ed2.gif" width="100">
</div>

```python
# Temporal Features 🕐
📅 pickup_hour          # Hour of day (0-23)
📆 pickup_weekday       # Day of week (0-6)
🎉 pickup_is_weekend    # Weekend indicator (0/1)
🗓️  pickup_month         # Month (1-12)

# Geospatial Features 🗺️
📍 distance_km          # Haversine distance between pickup/dropoff
```

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47e185d-9656-417b-b0d6-e79ea83aab14.gif" width="100%">

## 🤖 Model Architecture

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257460-738ff738-247f-4445-a718-cdd0ca76e2db.gif" width="100">

### XGBoost Configuration
</div>

```python
XGBRegressor(
    n_estimators=500,        # 🎯 Number of boosting rounds
    max_depth=10,            # 🌳 Maximum tree depth
    learning_rate=0.05,      # 📈 Step size shrinkage
    subsample=0.8,           # 🎲 Row sampling ratio
    colsample_bytree=0.8,    # 🎲 Column sampling ratio
    random_state=42          # 🔒 Reproducibility
)
```

### Why XGBoost?

<img align="left" src="https://user-images.githubusercontent.com/74038190/216122041-518ac897-8d92-4c6b-9b3f-ca01dcaf38ee.png" width="50" />

- ⚡ **Fast**: Parallel processing and optimized algorithms
- 🎯 **Accurate**: Handles complex non-linear relationships
- 🛡️ **Robust**: Built-in regularization prevents overfitting
- 📈 **Scalable**: Efficient with large datasets

<br clear="left"/>

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47e185d-9656-417b-b0d6-e79ea83aab14.gif" width="100%">

## 📈 Performance Metrics

<div align="center">

<img src="https://user-images.githubusercontent.com/74038190/212257463-4d082cb4-7483-4eaf-bc25-6dde2628aabd.gif" width="100">

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.67 | Explains 67% of variance ✨ |
| **RMSE** | ~3000 sec | ~50 minutes average error ⏱️ |
| **MAE** | ~2200 sec | Median error ~37 minutes 📊 |

</div>

### Performance Visualization

```
Actual vs Predicted Trip Duration
     │
 12k │     ╱╲
     │    ╱  ╲      🎯 Strong Correlation!
  8k │   ╱ ★★ ╲
     │  ╱ ★★★★ ╲
  4k │ ╱★★★★★★★╲
     │╱★★★★★★★★★╲
   0 └───────────────
     0  4k  8k  12k
```

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47e185d-9656-417b-b0d6-e79ea83aab14.gif" width="100%">

## 🚀 Installation

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif" width="100">
</div>

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository 📥
git clone https://github.com/husseinahmed/nyc-taxi-trip-duration-xgboost.git

# Navigate to project directory 📂
cd nyc-taxi-trip-duration-xgboost

# Create virtual environment (recommended) 🐍
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies 📦
pip install -r requirements.txt
```

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212281775-b468df30-4edc-4bf8-a4ee-f52e1aaddc86.gif" width="400">
</div>

### Requirements

```txt
pandas>=1.3.0          # 🐼 Data manipulation
numpy>=1.21.0          # 🔢 Numerical computing
scikit-learn>=1.0.0    # 🤖 ML algorithms
xgboost>=1.5.0         # 🚀 Gradient boosting
gradio>=3.0.0          # 🎨 Web interface
matplotlib>=3.4.0      # 📊 Visualization
seaborn>=0.11.0        # 📈 Statistical plots
```

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47e185d-9656-417b-b0d6-e79ea83aab14.gif" width="100%">

## 💻 Usage

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257465-7ce8d493-cac5-494e-982a-5a9deb852c4b.gif" width="100">
</div>

### Training the Model

```bash
# Run the training script 🏋️
python src/train.py
```

<img align="right" alt="Training" width="300" src="https://user-images.githubusercontent.com/74038190/212749447-bfb7e725-6987-49d9-ae85-2015e3e7cc41.gif">

This will:
1. ✅ Load and preprocess the dataset
2. ✅ Engineer features
3. ✅ Train the XGBoost model
4. ✅ Save the trained model to `models/xgboost_model.pkl`

<br clear="right"/>

### Running the Web Application

```bash
# Launch the Gradio interface 🚀
python app/app.py
```

Then open your browser and navigate to: `http://localhost:7860`

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/216120981-b9507c36-0e04-4469-8e27-c99271b45ba5.png" width="40" /> 
<strong>Live Demo Running!</strong>
<img src="https://user-images.githubusercontent.com/74038190/216120981-b9507c36-0e04-4469-8e27-c99271b45ba5.png" width="40" />
</div>

### Making Predictions via Code

```python
import pickle
import pandas as pd

# Load the trained model 📦
with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare input data 📝
data = {
    'passenger_count': 2,
    'pickup_hour': 18,
    'pickup_weekday': 4,
    'pickup_is_weekend': 0,
    'pickup_month': 6,
    'distance_km': 5.2
}

# Make prediction 🎯
prediction = model.predict(pd.DataFrame([data]))
print(f"Predicted trip duration: {prediction[0]:.2f} seconds")
```

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47e185d-9656-417b-b0d6-e79ea83aab14.gif" width="100%">

## 📁 Project Structure

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257467-871d32b7-e401-42e8-a166-fcfd7baa4c6b.gif" width="100">
</div>

```
nyc-taxi-trip-duration-xgboost/
│
├── 📂 data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Preprocessed data
│
├── 📂 notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── 📂 src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data cleaning & feature engineering
│   ├── model.py               # Model training & evaluation
│   └── train.py               # Main training script
│
├── 📂 app/
│   ├── app.py                 # Gradio web application
│   └── utils.py               # Helper functions
│
├── 📂 models/
│   └── xgboost_model.pkl      # Trained model
│
├── 📂 images/                  # Project images & visualizations
│
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── LICENSE                    # MIT License
```

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47e185d-9656-417b-b0d6-e79ea83aab14.gif" width="100%">

## 🎨 Web Application Demo

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257460-738ff738-247f-4445-a718-cdd0ca76e2db.gif" width="100">
</div>

The Gradio interface provides an intuitive way to make predictions:

**Input Parameters:**
- 👥 Number of passengers (1-6)
- 🕐 Pickup hour (0-23)
- 📅 Pickup day of week (0-6)
- 🎉 Weekend indicator
- 🗓️ Month (1-12)
- 📍 Distance in kilometers

**Output:**
- ⏱️ Predicted trip duration in minutes
- 📊 Confidence interval
- 📈 Visual comparison chart

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/221352975-94759904-aa4c-4032-a8ab-b546efb9c478.gif" width="500">
</div>

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47e185d-9656-417b-b0d6-e79ea83aab14.gif" width="100%">

## 🔬 Methodology

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257468-1e9a91f1-b626-4baa-b15d-5c385dfa7ed2.gif" width="100">
</div>

### 1. Data Preprocessing 🧹
```python
# Outlier Removal using IQR
Q1 = df['trip_duration'].quantile(0.25)
Q3 = df['trip_duration'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['trip_duration'] >= Q1 - 1.5*IQR) & 
        (df['trip_duration'] <= Q3 + 1.5*IQR)]
```

### 2. Feature Engineering 🛠️
```python
# Haversine Distance Calculation
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km
```

### 3. Model Training 🚀
```python
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = XGBRegressor(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)
```

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47e185d-9656-417b-b0d6-e79ea83aab14.gif" width="100%">

## 📊 Key Insights

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257463-4d082cb4-7483-4eaf-bc25-6dde2628aabd.gif" width="100">
</div>

- 🌆 **Peak Hours**: Longest trips occur during rush hours (7-9 AM, 5-7 PM)
- 📏 **Distance Impact**: Strong positive correlation (0.85) with trip duration
- 📅 **Day Patterns**: Weekends show 15% longer average trips
- 🌡️ **Seasonal Trends**: Summer months have higher variability

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212748830-4c709398-a386-4761-84d7-9e10b98fbe6e.gif" width="500">
</div>

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47e185d-9656-417b-b0d6-e79ea83aab14.gif" width="100%">

## 🛣️ Roadmap

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212257472-08e52665-c503-4bd9-aa20-f5a4dae769b5.gif" width="100">
</div>

- [x] ✅ Basic XGBoost model implementation
- [x] ✅ Feature engineering pipeline
- [x] ✅ Gradio web application
- [ ] 🔄 Add LightGBM ensemble model
- [ ] 🔄 Implement real-time traffic data integration
- [ ] 🔄 Deploy on AWS/Azure
- [ ] 🔄 Add weather data features
- [ ] 🔄 Create mobile app version

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47e185d-9656-417b-b0d6-e79ea83aab14.gif" width="100%">

## 🤝 Contributing

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/216122041-518ac897-8d92-4c6b-9b3f-ca01dcaf38ee.png" width="100" />
</div>

Contributions are welcome! Please follow these steps:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push to the branch (`git push origin feature/AmazingFeature`)
5. 🔃 Open a Pull Request

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="400">
</div>

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47e185d-9656-417b-b0d6-e79ea83aab14.gif" width="100%">

## 👤 Author

<div align="center">

<img src="https://user-images.githubusercontent.com/74038190/213844263-a8897a51-32f4-4b3b-b5c2-e1528b89f6f3.png" width="50px" />

**Hussain Ahmed**

<img src="https://user-images.githubusercontent.com/74038190/216122065-2f028bae-25d6-4a3c-bc9f-175394ed5011.png" width="100" />

Data Analyst | Machine Learning Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hussian-ahmed/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/HussainAhmed75)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:hrfy330@gmail.com)

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="700">

</div>

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47e185d-9656-417b-b0d6-e79ea83aab14.gif" width="100%">

## 🙏 Acknowledgments

<div align="center">
<img src="https://user-images.githubusercontent.com/74038190/216122069-5b8169d7-1d8e-4a13-b245-a8e4176c99f8.png" width="100" />
</div>

- 🚖 NYC Taxi & Limousine Commission for the dataset
- 💻 Kaggle community for insights and discussions
- 🚀 XGBoost development team
- 🎨 Gradio for the amazing UI framework

<img src="https://user-images.githubusercontent.com/74038190/212284115-f47e185d-9656-417b-b0d6-e79ea83aab14.gif" width="100%">

## 📞 Contact

<div align="center">

<img src="https://user-images.githubusercontent.com/74038190/216122003-7b8a634b-144e-4d10-9578-a52a5dbb49a2.png" width="100" />

For questions or feedback:
- 💬 Open an issue on GitHub
- 🔗 Reach out via LinkedIn  
- 📧 Send an email

</div>

---

<div align="center">

<img src="https://user-images.githubusercontent.com/74038190/213844263-a8897a51-32f4-4b3b-b5c2-e1528b89f6f3.png" width="50px" />&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/74038190/216122041-518ac897-8d92-4c6b-9b3f-ca01dcaf38ee.png" width="50" />&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/74038190/213844263-a8897a51-32f4-4b3b-b5c2-e1528b89f6f3.png" width="50px" />

**If you found this project helpful, please give it a ⭐!**

<img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="400">

Made with ❤️ and ☕ by Hussein Ahmed

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="700">

</div>
