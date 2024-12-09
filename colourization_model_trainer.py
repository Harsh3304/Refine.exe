import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt

class GrayscaleColorDataset(Dataset):
    def __init__(self, grayscale_dir, color_dir, transform=None):
        self.grayscale_dir = grayscale_dir
        self.color_dir = color_dir
        self.transform = transform
        self.grayscale_files = os.listdir(grayscale_dir)
        self.color_files = os.listdir(color_dir)
        
    def __len__(self):
        return len(self.grayscale_files)
    
    def __getitem__(self, idx):
        grayscale_path = os.path.join(self.grayscale_dir, self.grayscale_files[idx])
        color_path = os.path.join(self.color_dir, self.color_files[idx])
        
        gray_img = Image.open(grayscale_path).convert("L")  
        color_img = Image.open(color_path).convert("RGB")  

        if self.transform:
            gray_img = self.transform(gray_img)
            color_img = self.transform(color_img)

        return gray_img, color_img

class ColorizationCNN(nn.Module):
    def __init__(self):
        super(ColorizationCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: [128, 128, 128]
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: [256, 64, 64]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: [128, 128, 128]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # Output: [64, 256, 256]
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),  # Output: [3, 256, 256]
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# Load dataset
def get_data_loaders(grayscale_dir, color_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    dataset = GrayscaleColorDataset(grayscale_dir=grayscale_dir, color_dir=color_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (gray_images, color_images) in enumerate(dataloader):
            gray_images, color_images = gray_images.to(device), color_images.to(device)
            optimizer.zero_grad()

            outputs = model(gray_images)
            loss = criterion(outputs, color_images)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")
    
    return model

def colorize_random_image(model, grayscale_dir):
    model.eval()
    random_image = random.choice(os.listdir(grayscale_dir))
    img_path = os.path.join(grayscale_dir, random_image)

    img = Image.open(img_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    gray_img = transform(img).unsqueeze(0).to(device) 
    
    with torch.no_grad():
        colorized_img = model(gray_img).cpu().squeeze(0) 

    colorized_img = transforms.ToPILImage()(colorized_img)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original Grayscale")
    axes[1].imshow(colorized_img)
    axes[1].set_title("Colorized")
    plt.show()

if __name__ == "__main__":
    num_epochs = 200
    batch_size = 32
    learning_rate = 1e-3
    grayscale_dir = 'grayscale'  
    color_dir = 'clean_images' 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ColorizationCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = get_data_loaders(grayscale_dir, color_dir, batch_size)

    trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs)

    torch.save(trained_model.state_dict(), 'colorization_model.pth')

    colorize_random_image(trained_model, grayscale_dir)
