
# adapted from https://pytorch.org/tutorials/beginner/pytorch_with_examples.html 

import numpy as np
import matplotlib.pyplot as plt

# Create random input and output data
x = np.linspace(-np.pi, np.pi, 2000)
y_true = np.sin(x)

# Random initialization
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

def func_approx(x):
    return a + b * x + c * x ** 2 + d * x ** 3

plt.ion() # Enable interactive plotting so we can update the figure without blocking the code
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y_true, 'b')
line2, = ax.plot(x, func_approx(x), 'r')
ax.set_ylim(-2,2)

learning_rate = 1e-6
for step in range(2000):
    # Forward pass: compute predicted y
    y_approx = func_approx(x)

    # Compute and print loss
    loss = np.square(y_approx - y_true).sum()

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_approx = 2.0 * (y_approx - y_true)
    grad_a = grad_y_approx.sum()
    grad_b = (grad_y_approx * x).sum()
    grad_c = (grad_y_approx * x ** 2).sum()
    grad_d = (grad_y_approx * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

    if step % 100 == 0:
        print("Loss:", loss)
        # Update the figure
        line2.set_ydata(y_approx)
        fig.canvas.draw()
        fig.canvas.flush_events()

print(a, b, c, d)

plt.ioff()
plt.show()
