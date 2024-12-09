import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vgg16
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


class ImageDenoiseDataset(Dataset):
    def __init__(self, clean_dir, noisy_dirs, transform=None):
        self.clean_images = glob.glob(os.path.join(clean_dir, "*"))
        self.noisy_images = []

        for noisy_dir in noisy_dirs:
            self.noisy_images.extend(glob.glob(os.path.join(noisy_dir, "*")))

        self.transform = transform

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        clean_image_path = self.clean_images[idx]
        clean_image_name = os.path.basename(clean_image_path)

        noisy_image_path = os.path.join(noisy_dirs[np.random.randint(len(noisy_dirs))], f"noisy_{clean_image_name}")

        clean_image = Image.open(clean_image_path).convert("RGB")
        noisy_image = Image.open(noisy_image_path).convert("RGB")

        if self.transform:
            clean_image = self.transform(clean_image)
            noisy_image = self.transform(noisy_image)

        return noisy_image, clean_image

class MultiScaleDenoisingCNN(nn.Module):
    def __init__(self):
        super(MultiScaleDenoisingCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def perceptual_loss(output, target):
    vgg = vgg16(weights="IMAGENET1K_V1").features[:16].eval().to(output.device) 
    with torch.no_grad():
        output_features = vgg(output)
        target_features = vgg(target)
    loss = nn.MSELoss()(output_features, target_features)
    return loss

def calculate_psnr_batch(clean_images, denoised_images):
    batch_psnr = 0
    for i in range(clean_images.size(0)):
        clean_image = clean_images[i].cpu().numpy().transpose(1, 2, 0)
        denoised_image = denoised_images[i].cpu().detach().numpy().transpose(1, 2, 0)
        
        clean_image = np.clip(clean_image, 0, 1)
        denoised_image = np.clip(denoised_image, 0, 1)
        
        psnr = compare_psnr(clean_image, denoised_image, data_range=1)
        batch_psnr += psnr
    
    return batch_psnr / clean_images.size(0)

def train_model_with_advanced_techniques(model, dataloader, device, num_epochs=150, learning_rate=1e-3, patience=10):
    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    early_stopping_patience = patience
    best_loss = float('inf')
    best_psnr = 0.0
    epochs_no_improve = 0
    
    model.train()

    for epoch in tqdm(range(num_epochs), desc="Epochs Progress"):
        running_loss = 0.0
        total_psnr = 0.0
        batch_progress = tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for i, (noisy_imgs, clean_imgs) in batch_progress:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

            optimizer.zero_grad()

            outputs = model(noisy_imgs)
            loss_mse = criterion_mse(outputs, clean_imgs)
            loss_perceptual = perceptual_loss(outputs, clean_imgs) 
            total_loss = loss_mse + loss_perceptual 

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            psnr_value = calculate_psnr_batch(clean_imgs, outputs)
            total_psnr += psnr_value
            batch_progress.set_postfix(loss=total_loss.item(), psnr=psnr_value)

        avg_loss = running_loss / len(dataloader)
        avg_psnr = total_psnr / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.4f}')

        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_psnr = avg_psnr
            epochs_no_improve = 0

            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == early_stopping_patience:
            print(f"Early stopping triggered. Best PSNR: {best_psnr:.4f}")
            break

if __name__ == "__main__":
    clean_image_dir = "clean_images"
    noisy_dirs = ["noisy_images/gaussian", "noisy_images/salt_pepper", "noisy_images/poisson", "noisy_images/speckle"]  # List of noisy directories
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = ImageDenoiseDataset(clean_image_dir, noisy_dirs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MultiScaleDenoisingCNN().to(device)

    train_model_with_advanced_techniques(model, dataloader, device)
 