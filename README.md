# ML-Regression-Projects

# Machine Learning Regression Projects

This repository contains a collection of machine learning projects that focus on regression techniques and their application in various datasets. The projects include both traditional regression models and advanced machine learning techniques, with a focus on understanding model performance and accuracy.

---

## Table of Contents
1. Overview
2. Project 1: Lasso & Ridge Regression for Car Mileage Dataset
3. Project 2: Multiple Linear Regression (MLR) for Digital Marketing Investment Dataset
4. Project 3: Multiple Linear Regression (MLR) for Housing Dataset
5. Project 4: Employee Salary Prediction APP
6. Project 5: Experience Candidates Salary Prediction App
7. Project 6: Car Sales Prediction App
8. Installation Instructions

---

## Overview

This repository contains three main regression projects based on different datasets. The goal was to explore the use of regression models (Lasso, Ridge, MLR, and Simple Linear Regression) to make predictions and evaluate the performance of these models using various metrics. All models are implemented in Python using popular machine learning libraries like sklearn, pandas, and numpy.

---

### Project 1: Lasso & Ridge Regression for Car Mileage Dataset

In this project, I used the *Lasso and Ridge Regression* models to predict the mileage of cars based on various features (like weight, engine size, etc.). These models help in regularizing and reducing overfitting by introducing penalties for large coefficients.

*Key Features:*
- Implemented Lasso and Ridge regression models
- Compared performance using *Mean Squared Error (MSE)*
- Dataset: Car Mileage dataset with features like weight, engine size, horsepower, etc.

---

### Project 2: Multiple Linear Regression (MLR) for Digital Marketing Investment Dataset

This project uses *Multiple Linear Regression (MLR)* to predict the effectiveness of digital marketing investment (TV, radio, and newspaper spendings) on sales.

*Key Features:*
- Applied MLR to understand the relationship between marketing spendings and sales
- Compared the performance of MLR with other models
- Dataset: Digital marketing investment dataset

---

### Project 3: Multiple Linear Regression (MLR) for Housing Dataset

In this project, MLR was applied to predict house prices based on different features like square footage, number of rooms, and location.

*Key Features:*
- Explored the impact of multiple features on house pricing
- Compared model performance with other regression techniques
- Dataset: Housing price prediction dataset

**Project 4: Employee Salary Prediction Project**

Project Overview :
This project involves predicting employee salaries based on various features using different regression models. The goal is to compare the accuracy of multiple regression models and identify which one provides the best predictions for employee salaries.

The dataset used contains employee-related features such as experience, education, job role, etc., and the target variable is the salary of the employee. We have applied and compared multiple regression models, including Linear Regression, Polynomial Regression, Decision Tree Regression, Random Forest Regression, Support Vector Regression, and K-Nearest Neighbors Regression.

Additionally, a user-friendly frontend interface has been created using Streamlit to allow for easy interaction with the model, enabling users to input data and get salary predictions.

Models Used :
The following regression models have been implemented and compared for accuracy:

Linear Regression: A basic linear model to predict salary based on linear relationships between features.
Polynomial Regression: A more advanced form of linear regression to handle non-linear relationships.
Decision Tree Regression: A model that splits the data into subgroups to make predictions based on decision-making processes.
Random Forest Regression: An ensemble of decision trees that improves prediction accuracy by averaging multiple trees' predictions.
Support Vector Regression (SVR): A model that tries to find a hyperplane that best fits the data and can handle both linear and non-linear relationships.
K-Nearest Neighbors Regression (KNN): A model that predicts the value based on the average of the k-nearest data points.

Frontend - Streamlit Interface : 
A frontend interface has been developed using Streamlit to make the model accessible to users without requiring any technical knowledge. This interface allows users to:

- Input employee-related features such as experience, education level, job role, etc.
- Select the regression model they wish to use.
- Get real-time salary predictions based on the input data.
  
---

**Project 5: Experience Candidates Salary Prediction App**

Project Overview
This project predicts an employee’s salary based on their years of experience using a Simple Linear Regression model. The main objective of this project is to create a model that can predict salary values for employees based on their experience and present it through a Streamlit app interface for easy interaction.

The Streamlit app provides an easy-to-use interface where users can upload their own datasets or input years of experience to receive salary predictions. It also visualizes the regression line and evaluates the model's performance.

Features
Simple Linear Regression: The app uses a basic linear regression model that assumes a linear relationship between experience and salary.
User Input for Prediction: Users can input their years of experience to get a salary prediction.
Dataset Upload: Users can upload their own dataset to train the model and see how the prediction works on their data.
Visualization: A plot showing the regression line along with the data points is displayed, illustrating the model’s prediction.

Model Evaluation
The Simple Linear Regression model is evaluated on the dataset using metrics such as R-squared (R²) and Mean Squared Error (MSE). The app displays these metrics to help you assess the model's performance. Higher R² values indicate a better fit of the model to the data.

---

**Project 6: Car Sales Prediction App**

Project Overview :
This project is designed to predict whether a person will buy a car based on various features using two different machine learning models: K-Nearest Neighbors (KNN) and Logistic Regression. The goal is to predict a binary outcome where:

1 means the person will buy a car.
0 means the person will not buy a car.
We have also developed a user-friendly Streamlit frontend that allows users to input relevant features (such as age, income, previous purchases, etc.) and get real-time predictions from both the KNN and Logistic Regression models.

Features :
K-Nearest Neighbors (KNN): A machine learning algorithm that classifies individuals based on their similarity to other data points.
Logistic Regression: A statistical model that estimates the probability of a binary outcome (whether the person will buy a car or not).
Interactive Frontend: The app allows users to input personal and demographic data and predicts whether the user will buy a car.
Real-Time Predictions: The user gets predictions instantly, either showing 1 for buying or 0 for not buying.
Model Comparison: The frontend allows users to compare results between KNN and Logistic Regression models for the same data.

Model Evaluation:
The performance of both models is evaluated based on accuracy and confusion matrix.
The models provide predictions in binary format: 1 for a person who will buy a car, and 0 for a person who will not.

Usage :
Once the Streamlit app is running:

You will be asked to enter features such as Age, Income, and Previous Car Purchases.
After entering the data, click the Predict button.
The app will display predictions from both the KNN and Logistic Regression models.
You will see whether the user is likely to buy a car (1) or not (0).

Conclusion :
This project demonstrates the use of K-Nearest Neighbors (KNN) and Logistic Regression for predicting car sales based on customer features. By developing a simple and interactive Streamlit frontend, users can easily interact with both models to predict whether they will buy a car based on their input data.


## Installation Instructions

To run the code in this repository locally, follow these steps:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/jansaidaA2/ML-Regression-Projects.git
