
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #sampling helper
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import GridSearchCV #more pipeline
from sklearn.metrics import mean_squared_error, r2_score #eval metrics
from sklearn.externals import joblib #persists model, pickle package to store large numpy arrays

#edit
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url) #pandas gets csv data from the web? neat
print(data)
print(data.head())
