from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def train_test_split(data,period):
    train, test = data.iloc[:-period], data.iloc[-period:]
    return train, test

def eval_model_performance(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    return {'MSE': mse, 'RMSE': rmse, 'MAPE': mape}

def plot_forecast(actual, forecast, title='Forecast vs Actual', figsize=(16,4)):
    plt.figure(figsize=figsize)
    plt.plot(actual, label='Actual')
    plt.plot(forecast, label='Forecast')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_residuals(residuals, figsize=(12,4)):
    plt.figure(figsize=figsize)
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.show()


