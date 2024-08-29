Name: Agastinraaj A

Company:CODTECH IT SOLUTIONS

ID:CTO8DS6242

Domain:Machine Learning

Duration:July to August 2024

Mentor:Muzammil Ahmed


Overview

The script is a machine learning pipeline designed to predict housing prices based on features like square footage, number of bedrooms, and location. It reads a dataset, preprocesses the data, trains a linear regression model, and makes predictions.

Key Features

1. Data Loading and Cleaning
    - Loads a dataset from a CSV file named synthetic_housing_data_10k.csv.
    - Removes outliers from square_footage, num_bedrooms, and price columns using the interquartile range (IQR) method.

2. Feature Selection and Target Variable
    - Independent variables (features): square_footage, num_bedrooms, and location.
    - Target variable: price.

3. Data Preprocessing
    - Uses ColumnTransformer to preprocess data:
        - Standardizes numerical features (square_footage and num_bedrooms) using StandardScaler.
        - One-hot encodes the categorical feature (location) using OneHotEncoder.
    - Combines preprocessing steps into a Pipeline with a LinearRegression model.

4. Model Training
    - Splits dataset into training and test sets using an 80-20 split (train_test_split).
    - Trains the model on the training data using the pipeline.

5. Model Evaluation
    - Evaluates the model's performance using the test set.
    - Metrics used: Mean Squared Error (MSE) and R-squared value.
    - Generates a scatter plot and residuals distribution plot to visualize the model's performance.

6. Prediction Function
    - Defines a function predict_price to predict the price of a house based on user inputs for square footage, number of bedrooms, and location.
    - Preprocesses the input and uses the trained pipeline to make a prediction.

7. Interactive Command-Line Interface
    - Includes an interactive command-line interface that prompts the user to input square footage, number of bedrooms, and location, and outputs the predicted price.
