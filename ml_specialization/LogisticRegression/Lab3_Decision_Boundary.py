import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import plot_data, sigmoid, draw_vthresh
plt.style.use('./deeplearning.mplstyle')

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 

fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X, y, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
# plt.show()

# Refresher on logistic regression and decision boundary
# Recall that for logistic regression, the model is represented as

# 𝑓𝐰,𝑏(𝐱(𝑖))=𝑔(𝐰⋅𝐱(𝑖)+𝑏)(1)
# where  𝑔(𝑧)
#   is known as the sigmoid function and it maps all input values to values between 0 and 1:

# 𝑔(𝑧)=11+𝑒−𝑧(2)
# and  𝐰⋅𝐱
#   is the vector dot product:

# 𝐰⋅𝐱=𝑤0𝑥0+𝑤1𝑥1
 
# We interpret the output of the model ( 𝑓𝐰,𝑏(𝑥)
#  ) as the probability that  𝑦=1
#   given  𝐱
#   and parameterized by  𝐰
#   and  𝑏
#  .

# Therefore, to get a final prediction ( 𝑦=0
#   or  𝑦=1
#  ) from the logistic regression model, we can use the following heuristic -
# if  𝑓𝐰,𝑏(𝑥)>=0.5
#  , predict  𝑦=1
 
# if  𝑓𝐰,𝑏(𝑥)<0.5
#  , predict  𝑦=0
 
# Let's plot the sigmoid function to see where  𝑔(𝑧)>=0.5

# Generate an array of evenly spaced values between -10 and 10
z = np.arange(-10,11)
fig,ax = plt.subplots(1,1,figsize=(5,3))
# Plot z vs sigmoid(z)
ax.plot(z, sigmoid(z), c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
# plt.show()
 
# As you can see,  𝑔(𝑧)>=0.5
#   for  𝑧>=0
 
# For a logistic regression model,  𝑧=𝐰⋅𝐱+𝑏
#  . Therefore,

# if  𝐰⋅𝐱+𝑏>=0
#  , the model predicts  𝑦=1
 
# if  𝐰⋅𝐱+𝑏<0
#  , the model predicts  𝑦=0
 
# Plotting decision boundary
# Now, let's go back to our example to understand how the logistic regression model is making predictions.

# Our logistic regression model has the form

# 𝑓(𝐱)=𝑔(−3+𝑥0+𝑥1)
 
# From what you've learnt above, you can see that this model predicts  𝑦=1
#   if  −3+𝑥0+𝑥1>=0
 
# Let's see what this looks like graphically. We'll start by plotting  −3+𝑥0+𝑥1=0
#  , which is equivalent to  𝑥1=3−𝑥0
#  .

# Choose values between 0 and 6
x0 = np.arange(0,6)

x1 = 3 - x0
fig,ax = plt.subplots(1,1,figsize=(5,4))
# Plot the decision boundary
ax.plot(x0,x1, c="b")
ax.axis([0, 4, 0, 3.5])

# Fill the region below the line
ax.fill_between(x0,x1, alpha=0.2)

# Plot the original data
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()