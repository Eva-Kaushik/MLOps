#!/usr/bin/env python
# coding: utf-8

# In[2]:


import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("numpy")
install("pandas")
install("matplotlib")
install("scikit-learn")
install("mlflow")

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn

# In[3]:


TARGET_COL = "cost"

NUMERIC_COLS = [
    "distance",
    "dropoff_latitude",
    "dropoff_longitude",
    "passengers",
    "pickup_latitude",
    "pickup_longitude",
    "pickup_weekday",
    "pickup_month",
    "pickup_monthday",
    "pickup_hour",
    "pickup_minute",
    "pickup_second",
    "dropoff_weekday",
    "dropoff_month",
    "dropoff_monthday",
    "dropoff_hour",
    "dropoff_minute",
    "dropoff_second",
]

CAT_NOM_COLS = [
    "store_forward",
    "vendor",
]

CAT_ORD_COLS = [
]


# In[5]:


import os  # Import the os module

# Define Arguments for this step
class MyArgs:
    def __init__(self, /, **kwargs):
        self.__dict__.update(kwargs)

args = MyArgs(
    train_data = "/tmp/prep/train",
    model_output = "/tmp/train",
    regressor__n_estimators = 500,
    regressor__bootstrap = 1,
    regressor__max_depth = 10,
    regressor__max_features = "auto", 
    regressor__min_samples_leaf = 4,
    regressor__min_samples_split = 5
)

os.makedirs(args.model_output, exist_ok=True)


# In[6]:



def main(args):
    '''Read train dataset, train model, save trained model'''

    # Read train data
    train_data = pd.read_parquet(Path(args.train_data))

    # Split the data into input(X) and output(y)
    y_train = train_data[TARGET_COL]
    X_train = train_data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS]

    # Train a Random Forest Regression Model with the training set
    model = RandomForestRegressor(n_estimators = args.regressor__n_estimators,
                                  bootstrap = args.regressor__bootstrap,
                                  max_depth = args.regressor__max_depth,
                                  max_features = args.regressor__max_features,
                                  min_samples_leaf = args.regressor__min_samples_leaf,
                                  min_samples_split = args.regressor__min_samples_split,
                                  random_state=0)

    # log model hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.regressor__n_estimators)
    mlflow.log_param("bootstrap", args.regressor__bootstrap)
    mlflow.log_param("max_depth", args.regressor__max_depth)
    mlflow.log_param("max_features", args.regressor__max_features)
    mlflow.log_param("min_samples_leaf", args.regressor__min_samples_leaf)
    mlflow.log_param("min_samples_split", args.regressor__min_samples_split)

    # Train model with the train set
    model.fit(X_train, y_train)

    # Predict using the Regression Model
    yhat_train = model.predict(X_train)

    # Evaluate Regression performance with the train set
    r2 = r2_score(y_train, yhat_train)
    mse = mean_squared_error(y_train, yhat_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, yhat_train)
    
    # log model performance metrics
    mlflow.log_metric("train r2", r2)
    mlflow.log_metric("train mse", mse)
    mlflow.log_metric("train rmse", rmse)
    mlflow.log_metric("train mae", mae)

    # Visualize results
    plt.scatter(y_train, yhat_train,  color='black')
    plt.plot(y_train, y_train, color='blue', linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.savefig("regression_results.png")
    mlflow.log_artifact("regression_results.png")

    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)


# In[9]:


import mlflow
import os

# Define Arguments for this step
class MyArgs:
    def __init__(self, /, **kwargs):
        self.__dict__.update(kwargs)

args = MyArgs(
    train_data="/tmp/prep/train",
    model_output="/tmp/train",
    regressor__n_estimators=500,
    regressor__bootstrap=1,
    regressor__max_depth=10,
    regressor__max_features="auto",
    regressor__min_samples_leaf=4,
    regressor__min_samples_split=5
)

# Check if there's an active run, and end it if there is
if mlflow.active_run():
    mlflow.end_run()

# Start MLflow run
mlflow.start_run()

# Log parameters
mlflow.log_params(vars(args))

# Log custom information
lines = [
    f"Train dataset input path: {args.train_data}",
    f"Model output path: {args.model_output}",
    f"n_estimators: {args.regressor__n_estimators}",
    f"bootstrap: {args.regressor__bootstrap}",
    f"max_depth: {args.regressor__max_depth}",
    f"max_features: {args.regressor__max_features}",
    f"min_samples_leaf: {args.regressor__min_samples_leaf}",
    f"min_samples_split: {args.regressor__min_samples_split}"
]

for line in lines:
    print(line)

# Call main function (replace this with your actual main functionality)
def main(args):
    # Your main functionality here
    pass

main(args)

# End MLflow run
mlflow.end_run()


# In[10]:


ls "/tmp/train" 
