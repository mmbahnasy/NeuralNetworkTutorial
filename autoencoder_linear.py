import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

mnist_data = datasets.MNIST(root="data", train=True, download=True, transform=transform,)
batch_size = 64
dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

class Autoencoder_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3) # encode the image into 3 features
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def decode(self, encoded_image):
        return self.decoder(encoded_image)

model = Autoencoder_Linear()
MODEL_PATH = "./autoencoder_model1.pt"
try:
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Model parameters were loaded from:", MODEL_PATH)
except:
    pass

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


for epoch in range(10):
    for batch, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.reshape(-1, 28*28)
        recon = model(imgs)
        loss = criterion(recon, imgs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 99:
            print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
            torch.save(model.state_dict(), MODEL_PATH)

train_features, _ = next(iter(dataloader))
print("train_features.shape:", train_features.shape)
reconstructedImage = model(train_features.reshape(-1, 28*28)).reshape(train_features.shape).detach().numpy()
print("reconstructedImage.shape:", reconstructedImage.shape)

fig, axes = plt.subplots(2, 8)
for i in range(8):
    axes[0][i].imshow(train_features[i,0])
    axes[1][i].imshow(reconstructedImage[i,0])
plt.show()

if False:
    # Generating image from random features
    reconstructedImage = model.decode(torch.rand(batch_size,3)).reshape(-1, 28, 28).detach().numpy()
    print("reconstructedImage.shape:", reconstructedImage.shape)

    fig, axes = plt.subplots(1, 8)
    for i in range(8):
        axes[i].imshow(reconstructedImage[i])
    plt.show()
