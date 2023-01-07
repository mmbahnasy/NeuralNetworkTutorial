import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))
    ])

mnist_data = datasets.MNIST(root="data", train=True, download=True, transform=transform,)
batch_size = 64
dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # N, 1, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # -> N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # -> N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 16, 7) # -> N, 64, 1, 1
        )
        
        # N , 64, 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 7), # -> N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # N, 16, 14, 14 (N,16,13,13 without output_padding)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), # N, 1, 28, 28  (N,1,27,27)
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = Autoencoder()
MODEL_PATH = "./autoencoder_model2.pt"
try:
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Model parameters were loaded from:", MODEL_PATH)
except:
    pass

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-3, 
                             weight_decay=1e-5)

for epoch in range(0):
    for batch, (imgs, labels) in enumerate(dataloader):
        recon = model(imgs)
        loss = criterion(recon.view(-1), imgs.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 99:
            print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
            torch.save(model.state_dict(), MODEL_PATH)

train_features, train_labels = next(iter(dataloader))
print("train_features.shape:", train_features.shape)
reconstructedImage = model(train_features).reshape(train_features.shape).detach().numpy()
print("reconstructedImage.shape:", reconstructedImage.shape)

fig, axes =  plt.subplots(2, 8)
for i in range(8):
    axes[0][i].imshow(train_features[i, 0])
    axes[1][i].imshow(reconstructedImage[i, 0])
plt.show()
