import streamlit as st
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from u2net import utils, model

# Set page configuration
st.set_page_config(page_title="Background Removal App", layout="wide")

# Initialize session state variables
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'rotation_angle' not in st.session_state:
    st.session_state.rotation_angle = 0

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_name: str = "u2net"):
    if model_name == "u2netp":
        net = model.U2NETP(3, 1)
    elif model_name == "u2net":
        net = model.U2NET(3, 1)
    else:
        st.error("Choose between u2net or u2netp")
        return None

    try:
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_name + ".pth"))
            net.to(torch.device("cuda"))
        else:
            net.load_state_dict(torch.load(model_name + ".pth", map_location="cpu"))
    except FileNotFoundError:
        st.error(f"Model file {model_name}.pth not found!")
        return None

    net.eval()
    return net

def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3

    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose([utils.RescaleT(320), utils.ToTensorLab(flag=0)])
    sample = transform({"imidx": np.array([0]), "image": image, "label": label})
    return sample

def predict(net, item):
    sample = preprocess(item)

    with torch.no_grad():
        if torch.cuda.is_available():
            inputs_test = torch.cuda.FloatTensor(sample["image"].to(device).unsqueeze(0).float())
        else:
            inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        pred = d1[:, 0, :, :]
        predict = norm_pred(pred)

        predict = predict.squeeze()
        predict_np = predict.cpu().detach().numpy()
        img = Image.fromarray(predict_np * 255).convert("RGB")

        del d1, d2, d3, d4, d5, d6, d7, pred, predict, predict_np, inputs_test, sample

        return img

def process_image(image, net):
    # Convert uploaded file to PIL Image
    img_array = np.array(image)
    
    # Predict mask
    output = predict(net, img_array)
    output = output.resize((image.size), resample=Image.BILINEAR)
    
    # Create transparent background
    empty_img = Image.new("RGBA", (image.size), 0)
    # Convert original image to RGBA
    img_rgba = image.convert("RGBA")
    # Create final composite
    new_img = Image.composite(img_rgba, empty_img, output.convert("L"))
    
    return new_img

def main():
    st.title("Background Removal App")
    
    with st.spinner("Loading model..."):
        net = load_model("u2net")

    if 'success' not in st.session_state:
        st.session_state.success=False

    if net is None:
        st.error("Failed to load model. Please ensure the model file exists.")
        return
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.file_name = uploaded_file.name
            if st.session_state.processed_image is None:
                with st.spinner("Processing image..."):
                    st.session_state.processed_image = process_image(image, net)

            st.subheader("Processed Image")
            st.session_state.success=True
            st.session_state.rotation_angle = st.slider("Rotate Image", -180, 180, 0)
    if st.session_state.success:
        if st.session_state.rotation_angle != 0:
            rotated_image = st.session_state.processed_image.rotate(st.session_state.rotation_angle, expand=True)
            st.image(rotated_image, width=800)
        else:
            st.image(st.session_state.processed_image, width=800)
    if st.session_state.success:
        with st.sidebar:
            if st.button("Save Image"):
                os.makedirs('output', exist_ok=True)
                save_path = f"output/{st.session_state.file_name}"
                if st.session_state.rotation_angle != 0:
                    rotated_image.save(save_path,'PNG')
                else:
                    st.session_state.processed_image.save(save_path,'PNG')
                st.success(f"Image saved as {save_path}")

if __name__ == "__main__":
    main()