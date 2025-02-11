# Linear Regression (https://www.coursera.org/learn/machine-learning/home/week/1)
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt


dataset_name = "umernaeem217/synthetic-house-prices-univariate"
path = kagglehub.dataset_download(dataset_name) + "/house_prices_dataset.csv"

# initial values
w = 1
b = 0

def linear_prediction(x, w, b):
    return w*x + b


def cost_function(df, x_col, y_col):
    n = len(df)
    cost = 0
    for _, row in df.iterrows():
        x = row[x_col]
        y = row[y_col]
        y_ = linear_prediction(x, w, b)
        cost += (y_ - y)**2
    return cost/(2*n)

def compute_dowJ_w(df, x_col, y_col):
    n = len(df)
    to_sum = 0
    for _, row in df.iterrows():
        x = row[x_col]
        y = row[y_col]
        y_ = linear_prediction(x, w, b)
        to_sum += ((y_ - y)*(x))
    return to_sum/n

def compute_dowJ_b(df, x_col, y_col):
    n = len(df)
    to_sum = 0
    for _, row in df.iterrows():
        x = row[x_col]
        y = row[y_col]
        y_ = linear_prediction(x, w, b)
        to_sum += (y_ - y)
    return to_sum/n

def gradient_descent(df, x_col, y_col):
    global w, b
    alpha = 0.0000001
    dowJ_w = 1000
    dowJ_b = 1000
    epsilon = 1e-6
    iter_count = 0
    max_iterations = 1000

    while (abs(dowJ_w) > epsilon or abs(dowJ_b) > epsilon) and iter_count < max_iterations:
        # descending
        iter_count += 1
        dowJ_w = compute_dowJ_w(df, x_col, y_col)
        dowJ_b = compute_dowJ_b(df, x_col, y_col)
        w = w - alpha * dowJ_w
        b = b - alpha * dowJ_b
        cost = cost_function(df, x_col, y_col)

        print("w: ", w, " b: ", b)
        print("dowJ_w: ", dowJ_w, " dowJ_b: ", dowJ_b)
        print("cost: ", cost)
        print("--------------------------------")

x_col = "area"
y_col = "price"

df = pd.read_csv(path)
gradient_descent(df, x_col, y_col)