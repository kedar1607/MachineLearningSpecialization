import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import  plot_data, sigmoid, dlc
plt.style.use('./deeplearning.mplstyle')

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])                                           #(m,)


# The data points with label  𝑦=1
#   are shown as red crosses, while the data points with label  𝑦=0
#   are shown as blue circles.
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)

# Set both axes to be from 0-4
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()



# In a previous lab, you developed the logistic loss function. Recall, loss is defined to apply to one example. Here you combine the losses to form the cost, which includes all the examples.

# Recall that for logistic regression, the cost function is of the form

# 𝐽(𝐰,𝑏)=1/𝑚∑[𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),𝑦(𝑖))] (from 0 to m-1)
# where

# 𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),𝑦(𝑖))
#   is the cost for a single data point, which is:

# 𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),𝑦(𝑖))=−𝑦(𝑖)log(𝑓𝐰,𝑏(𝐱(𝑖)))−(1−𝑦(𝑖))log(1−𝑓𝐰,𝑏(𝐱(𝑖)))
# where m is the number of training examples in the data set and:
# f(x_i) = g(z_i)
# z_i = (w * x_i) + b
# g(z_i) = 1 / (1 + e^(-z_i)) .. the sigmoid function
def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        # f_wb_i = 1 / (1 + np.exp(-z_i)) # sigmoid function
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost


# Test your function above
w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))



# Now, let's see what the cost function output is for a different value of  𝑤
#  .

# In a previous lab, you plotted the decision boundary for  𝑏=−3,𝑤0=1,𝑤1=1
#  . That is, you had b = -3, w = np.array([1,1]).

# Let's say you want to see if  𝑏=−4,𝑤0=1,𝑤1=1
#  , or b = -4, w = np.array([1,1]) provides a better model.

# Let's first plot the decision boundary for these two different  𝑏
#   values to see which one fits the data better.

# For  𝑏=−3,𝑤0=1,𝑤1=1
#  , we'll plot  −3+𝑥0+𝑥1=0
#   (shown in blue)
# For  𝑏=−4,𝑤0=1,𝑤1=1
#  , we'll plot  −4+𝑥0+𝑥1=0
#   (shown in magenta)

# Choose values between 0 and 6
x0 = np.arange(0,6)

# Plot the two decision boundaries
x1 = 3 - x0
x1_other = 4 - x0

fig,ax = plt.subplots(1, 1, figsize=(4,4))
# Plot the decision boundary
ax.plot(x0,x1, c=dlc["dlblue"], label="$b$=-3")
ax.plot(x0,x1_other, c=dlc["dlmagenta"], label="$b$=-4")
ax.axis([0, 4, 0, 4])

# Plot the original data
plot_data(X_train,y_train,ax)
ax.axis([0, 4, 0, 4])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.legend(loc="upper right")
plt.title("Decision Boundary")
# plt.show()


w_array1 = np.array([1,1])
b_1 = -3
w_array2 = np.array([1,1])
b_2 = -4

# You can see the cost function behaves as expected and the cost for b = -4, w = np.array([1,1]) is indeed higher than the cost for b = -3, w = np.array([1,1])
print("Cost for b = -3 : ", compute_cost_logistic(X_train, y_train, w_array1, b_1))
print("Cost for b = -4 : ", compute_cost_logistic(X_train, y_train, w_array2, b_2))