# Refine.exe 🖼️✨🎨🔧

## Overview

Refine.exe is a powerful desktop application for image processing, offering two key functionalities:
- Image Denoising
- Image Colorization

Built using Python, PyTorch, and Tkinter, this application provides an intuitive graphical interface for enhancing and transforming images.

## ✨ Features

### 1. Image Denoising
- Remove noise from color images
- Uses a custom Convolutional Neural Network (CNN) for denoising
- Apply median filtering for preliminary noise reduction
- Preview and download denoised images

### 2. Image Colorization
- Convert grayscale images to color
- Utilizes a deep learning model to add realistic color
- Supports various image types
- Easy preview and download of colorized images

## 📦 Prerequisites

- Python 3.7+
- PyTorch
- NumPy
- Pillow (PIL)
- SciPy
- Tkinter

## 🚀 Installation

1. Clone the repository
```bash
git clone https://github.com/Harsh3304/Refine.exe.git
cd refine-exe
```

2. Install required dependencies
```bash
pip install torch torchvision numpy pillow scipy
```

3. Ensure you have the pre-trained models:
- `models/denoising_cnn_150.pth`
- `models/colorization_model_100eph.pth`

## 💻 Usage

Run the application:
```bash
python main.py
```
## 🖼️ Screenshots

1. **Home Page (for new user)**  
   <img src="/Dependencies/readme_images/home_screen.png" alt="Home Screen Page" width="700">  
    

2. **Denoising Demonstration**  
   <img src="/Dependencies/readme_images/noise_removal.jpg" alt="Denoising" width="700">  
    

2. **Colourization Demonstration**  
   <img src="/Dependencies/readme_images/colourization.jpg" alt="SignIn Page" width="700">  

### Denoising Workflow
1. Click "Add Image" button
2. Select a noisy image
3. Click "Start" to process
4. Preview the denoised image
5. Download if satisfied

### Colorization Workflow
1. Click "Add Image" button
2. Select a grayscale image
3. Click "Start" to process
4. Preview the colorized image
5. Download if satisfied

## 🔍 Model Details

### Denoising Model
- Architecture: Custom Convolutional Neural Network
- Trained on: 150 epochs
- Preprocessing: Median filtering
- Input: RGB images
- Output: Noise-reduced images

### Colorization Model
- Architecture: Custom Convolutional Neural Network
- Trained on: 100 epochs
- Input: Grayscale images
- Output: Colorized images

## 📂 Project Structure

```
refine-exe/

├── colour_to_gray.py
├── colourization_model_trainer.py
├── denoising_model_trainer.py
├── main.py
├── model.py
├── models/
│   ├── denoising_cnn_150.pth
│   └── colorization_model_100eph.pth
├── Dependencies/
│   ├── images/
│   └── UI assets
|   └── Readme_images
├── requirements.txt
└── tester_for_denoising.py
```

## Dependencies Folder

The `Dependencies` folder contains UI assets:
- Background images
- Button images
- UI design elements

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

Harsh Patel - harshp3304@gmail.com

Project Link: [https://github.com/Harsh3304/Refine.exe](https://github.com/Harsh3304/Refine.exe)