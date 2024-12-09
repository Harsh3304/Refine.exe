import os
import torch
from torchvision import transforms
from PIL import Image
from model import MultiScaleDenoisingCNN
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import matplotlib.pyplot as plt
import numpy as np

# Function to denoise an image using the trained model
def denoise_image(model, noisy_image, device):
    model.eval()
    with torch.no_grad():
        noisy_image = noisy_image.unsqueeze(0).to(device)  # Add batch dimension
        denoised_image = model(noisy_image)
        denoised_image = denoised_image.squeeze(0).cpu()  # Remove batch dimension
    return denoised_image

# Function to compute PSNR between clean and denoised images
def calculate_psnr(clean_image, denoised_image):
    clean_np = clean_image.cpu().numpy().transpose(1, 2, 0)
    denoised_np = denoised_image.cpu().numpy().transpose(1, 2, 0)
    
    clean_np = np.clip(clean_np, 0, 1)
    denoised_np = np.clip(denoised_np, 0, 1)
    
    psnr = compare_psnr(clean_np, denoised_np, data_range=1)
    return psnr

# Function to load and transform the image
def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image)

# Function to display images side by side with PSNR
def display_images(noisy_image, denoised_image, clean_image, psnr_value):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert tensors to numpy format for display
    noisy_image_np = noisy_image.cpu().numpy().transpose(1, 2, 0)
    denoised_image_np = denoised_image.cpu().numpy().transpose(1, 2, 0)
    clean_image_np = clean_image.cpu().numpy().transpose(1, 2, 0)
    
    ax[0].imshow(np.clip(noisy_image_np, 0, 1))
    ax[0].set_title("Noisy Image")
    ax[0].axis("off")

    ax[1].imshow(np.clip(denoised_image_np, 0, 1))
    ax[1].set_title(f"Denoised Image\nPSNR: {psnr_value:.2f} dB")
    ax[1].axis("off")
    
    ax[2].imshow(np.clip(clean_image_np, 0, 1))
    ax[2].set_title("Clean Image")
    ax[2].axis("off")
    
    plt.show()


# Main function to test the model
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories for test images
    clean_image_dir = "test_clean"  # Directory containing clean images
    noisy_image_dir = "test_noisy"  # Directory containing noisy images

    # Define transformations (matching those used in training)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Load the trained model
    model = MultiScaleDenoisingCNN().to(device)
    model.load_state_dict(torch.load("best_model.pth"))

    # Loop through images in the noisy directory and find corresponding clean images
    for image_name in os.listdir(noisy_image_dir):
        noisy_image_path = os.path.join(noisy_image_dir, image_name)
        clean_image_path = os.path.join(clean_image_dir, image_name)

        if os.path.exists(noisy_image_path) and os.path.exists(clean_image_path):
            # Load images
            noisy_image = load_image(noisy_image_path, transform)
            clean_image = load_image(clean_image_path, transform)

            # Denoise the noisy image
            denoised_image = denoise_image(model, noisy_image, device)

            # Calculate PSNR
            psnr_value = calculate_psnr(clean_image, denoised_image)
            print(f"PSNR for {image_name}: {psnr_value:.2f} dB")

            # Display images for visual comparison with PSNR
            display_images(noisy_image, denoised_image, clean_image, psnr_value)
