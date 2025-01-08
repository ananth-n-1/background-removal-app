# Background Removal App

A Streamlit-based web application that removes backgrounds from images using the U^2-Net deep learning model.

## Overview

This application provides a user-friendly interface for removing backgrounds from images using the U^2-Net (U square net) model, which is specifically designed for salient object detection. The app features a simple upload interface, image rotation capabilities, and the ability to save processed images.

## Features

- Upload images (supports JPG, JPEG, PNG formats)
- Automatic background removal using U^2-Net
- Image rotation controls (-180° to 180°)
- Save processed images with transparent backgrounds
- Support for both CPU and CUDA processing
- Responsive web interface

## Prerequisites

- Python 3.7 or higher
- Streamlit
- PyTorch
- NumPy
- Pillow (PIL)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/ananth-n-1/background-removal-app.git
cd background-removal-app
```

2. Install the required packages:
```bash
pip install streamlit torch torchvision numpy Pillow scikit-image
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL 

3. Upload an image using the sidebar interface

4. Wait for the background removal processing to complete

5. Use the rotation slider if needed to adjust the image orientation

6. Click the "Save Image" button to save the processed image

## Model Information

This application uses the U^2-Net model for salient object detection and background removal. U^2-Net is a deep neural network that was specifically designed to detect salient objects in images with high precision.

- Original U^2-Net Repository: [https://github.com/xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)
- Paper: [U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/abs/2005.09007)

## Project Structure

```
background-removal-app/
├── app.py                # Main Streamlit application
├── u2net                 # U^2-Net model implementation
├── u2net.pth             # Pre-trained model weights (needs to be downloaded)(https://drive.google.com/file/d/1u_6j8rVn4xZCHA1yymYbf97Earoie8vu/view?usp=sharing)
├── output/               # Directory for saved processed images
└── README.md             # Project documentation
```

## Acknowledgments

- U^2-Net model and pre-trained weights by [xuebinqin](https://github.com/xuebinqin/U-2-Net)
- Built with [Streamlit](https://streamlit.io/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
