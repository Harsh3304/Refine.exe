import os
from PIL import Image

# Function to convert images to grayscale
def convert_to_grayscale(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each image in the input folder
    for img_file in os.listdir(input_folder):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            # Open the image
            img_path = os.path.join(input_folder, img_file)
            img = Image.open(img_path)
            
            # Convert the image to grayscale
            grayscale_img = img.convert("L")
            
            # Save the grayscale image to the output folder
            grayscale_img.save(os.path.join(output_folder, img_file))

    print(f"All images have been converted to grayscale and saved to {output_folder}")

# Example usage
input_folder = 'test_clean'
output_folder = 'test_grayscale'
convert_to_grayscale(input_folder, output_folder)
