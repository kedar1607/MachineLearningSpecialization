# In this lab we are going to see how different python 
# libraries can be used to plot data points. 
# We will be using the matplotlib library to plot the data points.
# We will also be using the numpy library to create arrays of data points.

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
# print(f"x_train = {x_train}")
# print(f"y_train = {y_train}")

# m is the number of training examples
# print(f"x_train.shape: {x_train.shape}")
# m = x_train.shape[0]
# print(f"Number of training examples is: {m}")

# m is the number of training examples
# m = len(x_train)
# print(f"Number of training examples is: {m}")

# i = 0 # Change this to 1 to see (x^1, y^1)

# x_i = x_train[i]
# y_i = y_train[i]
# print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# # Plot the data points
# plt.scatter(x_train, y_train, marker='x', c='red')
# # Set the title
# plt.title("Housing Prices")
# # Set the y-axis label
# plt.ylabel('Price (in 1000s of dollars)')
# # Set the x-axis label
# plt.xlabel('Size (1000 sqft)')
# plt.show()



w = 100
b = 100
# print(f"w: {w}")
# print(f"b: {b}")

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb


tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()