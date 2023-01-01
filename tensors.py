import torch
import numpy as np 

data = [[1., 2.], [3., 4.]]
# print(torch.tensor(data, dtype=torch.float))

np_d = np.array(data)

t_d = torch.from_numpy(np_d)

# np_d[0,0] = 10
# print(np_d)
# print(t_d)

t = torch.rand_like(t_d)
# print(t)

t0 = torch.ones(t_d.shape)
# print(t0)

# print(t_d * t_d)


# print(t_d.T)

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2
# print(a, b)

Q.backward(gradient=torch.tensor([1., 1.]))
# print(a, b)
# print(a.grad, b.grad)

data = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
# print(data)

# print(data.reshape((2,3)))

# print(torch.cat((data, data), dim=0 ))

