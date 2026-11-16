# Food Delivery Time Prediction with Machine Learning

A comprehensive machine learning and web application project that predicts food delivery times using real-world order data. The project covers the complete data science workflow—from exploratory data analysis and preprocessing to model training, evaluation, and deployment in a small Flask app.

Live Demo: https://estimate-time-for-food-delivery.onrender.com/

## Project Overview

This project analyzes 45,593 delivery orders to estimate the time required to complete a food delivery in minutes. It uses key factors such as:

- delivery distance
- traffic density
- weather conditions
- order type
- vehicle type
- city type
- festival status
- delivery person age and rating
- number of deliveries in progress

The repository includes both the machine learning notebook and a deployed web interface that lets users estimate delivery times from input values.

## Highlights

- End-to-end ML pipeline from CSV to prediction
- Feature engineering and cleaning pipeline
- Model training and evaluation with regression algorithms
- Flask-based frontend for prediction
- REST API endpoint for integration
- Deployed live demo on Render

## Tech Stack

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Flask
- Joblib

## Dataset

- Total records: 45,593
- Target variable: Time_taken(min)
- Features include restaurant and destination coordinates, order timing, traffic, weather, city, vehicle, festival, and rider metadata.

## Model Workflow

1. Load and clean the dataset
2. Handle missing and inconsistent values
3. Engineer distance and time-based features
4. Encode categorical variables
5. Train regression models
6. Evaluate using RMSE and R²
7. Save the trained model
8. Serve predictions through a Flask app

## Repository Structure

```bash
Food-Delivery-Time-Prediction-With-Machine-Learning/
├── README.md
├── Food_delivery_time_prediction.ipynb
├── Food delivery.csv
├── train.py
├── app.py
├── requirements.txt
├── model/
│   └── model.joblib
├── templates/
│   └── index.html
└── .gitignore
```

## Live Demo

Open the deployed app here:

https://estimate-time-for-food-delivery.onrender.com/

## Local Setup

### 1) Clone the repository

```bash
git clone https://github.com/soumensen411/Food-Delivery-Time-Prediction-With-Machine-Learning.git
cd Food-Delivery-Time-Prediction-With-Machine-Learning
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Train the model

```bash
python train.py "Food delivery.csv"
```

### 4) Run the app

```bash
python app.py
```

Then open:

```bash
http://127.0.0.1:5000/
```

## API Usage

The Flask app exposes a prediction endpoint:

```bash
POST /api/predict
```

Example JSON body:

```json
{
  "distance": 8.5,
  "hour": 18,
  "traffic": "High",
  "weather": "Sunny",
  "order_type": "Meal",
  "city": "Metropolitian",
  "vehicle": "motorcycle",
  "vehicle_condition": 2,
  "age": 28,
  "rating": 4.6,
  "multiple_deliveries": 1,
  "festival": "No"
}
```

## Notes

- If no trained model file is present, the app runs in demo mode with rule-based fallback estimates.
- The project is designed for practical ML learning and deployment demos, making it useful for portfolios, coursework, and prediction-based web applications.

## License

This project is for educational and demonstration purposes.
