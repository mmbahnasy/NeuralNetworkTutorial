
# adapted from https://pytorch.org/tutorials/beginner/pytorch_with_examples.html 

import torch
import matplotlib.pyplot as plt

# Create random itorchut and output data
x = torch.linspace(-torch.pi, torch.pi, 2000)
y_true = torch.sin(x)

# preparing the tensor (x, x^2, x^3).
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

loss_fn = torch.nn.MSELoss(reduction='sum')

plt.ion() # Enable interactive plotting so we can update the figure without blocking the code
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y_true, 'b')
line2, = ax.plot(x, model(xx).detach().numpy(), 'r')
ax.set_ylim(-2,2)


learning_rate = 1e-6
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for step in range(2000):
    # Forward pass: compute predicted y
    y_approx = model(xx)

    # Compute and print loss
    loss = loss_fn(y_approx, y_true)

    # Zero the gradients before running the backward pass.
    model.zero_grad()
    # Backward pass: compute gradient.
    loss.backward()

    # Update weights
    # with torch.no_grad():
    #     for param in model.parameters():
    #         param -= learning_rate * param.grad
    optimizer.step()

    if step % 100 == 0:
        print("Loss:", loss)
        # Update the figure
        line2.set_ydata(y_approx.detach().numpy())
        fig.canvas.draw()
        fig.canvas.flush_events()

linear_layer = model[0]

print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')


plt.ioff()
plt.show()
