# Boston House Pricing Prediction

Problem Statement:
The Boston housing dataset contains various features about houses in Boston, such as crime rates, average number of rooms, property tax rates, and more, along with their corresponding house prices. The objective of this project is to predict the median value of owner-occupied homes in Boston using a set of predictors (features) available in the dataset.

This is a supervised regression problem where the goal is to build a model that can accurately predict housing prices based on input features. The model will be trained on historical data and tested on unseen data to evaluate its performance.

### Key Steps:
Data Preprocessing:

Load and inspect the dataset.

Clean the data (check for missing values, remove or handle outliers, etc.).

Rename columns for clarity (e.g., renaming the target variable to price).

Exploratory Data Analysis (EDA):

Explore relationships between features and the target variable (price).

Visualize correlations and distributions of features to understand their impact on house prices.

Use tools like correlation matrices and pair plots to find the most influential features.

Feature Engineering:

Select relevant independent variables (features) for the model.

Transform data as needed (e.g., scaling numerical values).

Model Building:

Split the data into training and testing sets.

Choose a machine learning algorithm (Linear Regression in this case).

Train the model on the training data.

Model Evaluation:

Evaluate the model using various metrics, including Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).

Analyze the model's performance on test data and adjust if necessary.

Model Deployment (Optional):

Save the trained model using pickle or joblib for future use or deployment.

## Software and Tools

1. [Github Account](https://github.com/)
2. [Heroku Account](https://www.heroku.com/)
3. [VS Code IDE](https://code.visualstudio.com/)
4. [Git CLI](https://git-scm.com/)

#### Create new environment

""" python3 -m venv evn_name """

"""evn_name\Scripts\activate"""
"""evn_name\Scripts\deactivate"""

### Install requirements.txt
""" pip install -r requirements.txt

### Config git global username and email
""" git config --global user.name "name"
""" git config --global user.email "email"



