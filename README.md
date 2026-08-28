# Food Delivery Time Prediction with Machine Learning

A machine learning project that predicts food delivery times using various features such as delivery person information, restaurant location, weather conditions, traffic density, and vehicle type.

## 🎯 Project Overview

This project aims to build predictive models to estimate delivery times for food orders. By analyzing patterns in historical delivery data, we can help delivery platforms optimize logistics and provide customers with more accurate delivery time estimates.

### Dataset

- **Total Records**: 45,593 delivery orders
- **Features**: 20 columns including delivery person details, location coordinates, weather conditions, traffic density, vehicle type, and more
- **Target Variable**: `Time_taken(min)` - Delivery time in minutes

## 📊 Features

The dataset includes the following features:

| Feature | Description |
|---------|-------------|
| **ID** | Unique identifier for each delivery |
| **Delivery_person_ID** | ID of the delivery person |
| **Delivery_person_Age** | Age of the delivery person |
| **Delivery_person_Ratings** | Customer rating of the delivery person |
| **Restaurant_latitude** | Latitude coordinate of the restaurant |
| **Restaurant_longitude** | Longitude coordinate of the restaurant |
| **Delivery_location_latitude** | Latitude of delivery destination |
| **Delivery_location_longitude** | Longitude of delivery destination |
| **Order_Date** | Date when the order was placed |
| **Time_Orderd** | Time when order was placed |
| **Time_Order_picked** | Time when order was picked up |
| **Weatherconditions** | Weather at time of delivery (Sunny, Cloudy, Stormy, Fog, Sandstorms, Windy) |
| **Road_traffic_density** | Traffic density (Low, Medium, High, Jam) |
| **Vehicle_condition** | Condition of the vehicle (0-3 scale) |
| **Type_of_order** | Type of order (Snack, Meal, Drinks, Buffet) |
| **Type_of_vehicle** | Vehicle type (motorcycle, scooter, bicycle, electric_scooter) |
| **multiple_deliveries** | Number of deliveries person is making |
| **Festival** | Whether delivery is during a festival (Yes/No) |
| **City** | City type (Urban, Metropolitan) |
| **Time_taken(min)** | **TARGET** - Delivery time in minutes |

## 🔍 Data Quality

- **Missing Values**: No missing values in the dataset
- **Duplicates**: No duplicate records
- **Data Consistency Issues**: 
  - 616 rows with weather conditions marked as "conditions NaN"
  - Some rows have missing delivery person age/ratings
  - Cleaned and handled during preprocessing

## 🛠️ Tech Stack

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **Scikit-learn** - Machine learning algorithms
- **Google Colab** - Development environment (with GPU support)

## 📋 Project Workflow

1. **Data Loading & Exploration**
   - Load dataset from CSV
   - Explore data structure and statistics
   - Check for missing values and duplicates

2. **Data Preprocessing**
   - Clean data (handle missing values, fix data types)
   - Feature extraction (extract temporal features)
   - Feature engineering (calculate distance, handle categorical variables)
   - Remove outliers if necessary

3. **Exploratory Data Analysis (EDA)**
   - Statistical analysis of features
   - Correlation analysis
   - Distribution analysis
   - Weather and traffic impact analysis

4. **Feature Engineering**
   - Create new features (e.g., distance between restaurant and delivery location)
   - Encode categorical variables
   - Scale numerical features

5. **Model Development**
   - Train multiple regression models
   - Hyperparameter tuning
   - Model evaluation and comparison

6. **Model Evaluation**
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - R² Score

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Project

1. Clone the repository:
```bash
git clone https://github.com/soumensen411/Food-Delivery-Time-Prediction-With-Machine-Learning.git
cd Food-Delivery-Time-Prediction-With-Machine-Learning
```

2. Open the Jupyter notebook:
```bash
jupyter notebook Food_delivery_time_prediction.ipynb
```

3. Run all cells to execute the complete analysis and model training pipeline.

**Note**: This project was developed in Google Colab. To run it locally, ensure all required libraries are installed.

## 📈 Key Insights

- Delivery time varies significantly based on weather conditions
- Traffic density is a major factor affecting delivery duration
- Vehicle type impacts delivery efficiency
- Delivery person ratings may correlate with delivery speed
- Distance between restaurant and delivery location is a strong predictor

## 🎓 Model Algorithms

The project explores and compares various regression algorithms:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- (Additional algorithms as implemented in the notebook)

## 📊 Performance Metrics

Models are evaluated using:
- **MAE** (Mean Absolute Error) - Average absolute difference
- **MSE** (Mean Squared Error) - Penalizes larger errors
- **RMSE** (Root Mean Squared Error) - Interpretable in same units as target
- **R² Score** - Model variance explanation

## 🔧 Data Preprocessing Steps

### Handling Missing Values
```python
# Remove rows with null weather conditions or delivery person info
# Impute or remove rows with missing critical features
```

### Feature Engineering
```python
# Calculate distance using latitude/longitude
# Extract day, month, year from dates
# Extract hour from time features
# Encode categorical variables (weather, traffic, vehicle type)
```

### Data Normalization
```python
# Scale numerical features for better model performance
# Apply appropriate encoding for categorical variables
```

## 📁 Project Structure

```
Food-Delivery-Time-Prediction-With-Machine-Learning/
├── README.md                              # Project documentation
├── Food_delivery_time_prediction.ipynb    # Main Jupyter notebook
├── .gitignore                             # Git ignore file
└── [Data files - if included]
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Share ideas for feature engineering

## 📝 License

This project is open source and available for educational and research purposes.

## 👤 Author

**Soumendra** - [GitHub Profile](https://github.com/soumensen411)

## 🙏 Acknowledgments

- Dataset source and inspiration
- Open-source libraries and communities
- Machine learning best practices and tutorials

## 📞 Contact

For questions or collaboration opportunities, feel free to reach out via GitHub.

---

**Happy Predicting!** 🚀

*Last Updated: 2026*