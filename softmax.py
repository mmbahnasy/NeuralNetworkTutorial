import torch
import matplotlib.pyplot as plt


x= torch.linspace(-5, 5, 10)
X = torch.zeros((len(x) ** 2, 2))
for i in range(len(x) ** 2):
    X[i][0] = x[int(i/len(x))]
    X[i][1] = x[i%len(x)]
# X = torch.outer(x, torch.ones(3))
print(x)
print(X)
# a = torch.randn(6, 9)
# print(a)
logits = torch.softmax(X, dim=1)
print(logits)

X, Y = torch.meshgrid(x, x)
# Z1 = torch.zeros_like(X)
# Z2 = torch.zeros_like(X)
# for i in range(len(x) ** 2):
#     Z1[int(i/len(x))][i%len(x)] = logits[i, 0]
#     Z2[int(i/len(x))][i%len(x)] = logits[i, 1]
Z1 = logits[:,0].reshape(X.shape)
Z2 = logits[:,1].reshape(X.shape)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')
ax.set_zlabel('Probability')

ax.plot_surface(X, Y, Z1, cmap='viridis', edgecolor='none')
# ax.plot_surface(X, Y, Z2, cmap='plasma', edgecolor='none')
# ax.plot_surface(X, Y, -torch.log(Z1), cmap='viridis', edgecolor='none')

ax.set_title('Surface plot')
ax.view_init(20, 220)
plt.show()
