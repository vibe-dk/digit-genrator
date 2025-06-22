import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# Define the Generator class again
class Generator(nn.Module):
    def __init__(self, nz=100, n_classes=10, img_size=28):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz + n_classes, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        labels_emb = self.label_emb(labels)
        gen_input = torch.cat((noise, labels_emb), -1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("generator.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_images(model, digit, n=5):
    noise = torch.randn(n, 100)
    labels = torch.full((n,), digit, dtype=torch.long)
    with torch.no_grad():
        imgs = model(noise, labels).cpu()
    imgs = (imgs + 1) / 2  # Scale from [-1, 1] to [0, 1]
    return imgs

# --- Streamlit UI ---
st.markdown("<h1 style='text-align: center;'>Handwritten Digit Image Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Generate synthetic MNIST-like images using your trained model.</p>", unsafe_allow_html=True)

# Dropdown instead of slider
digit = st.selectbox("Choose a digit to generate (0–9):", options=list(range(10)), index=0)

if st.button("Generate Images"):
    gen = load_model()
    imgs = generate_images(gen, digit)

    st.markdown(f"### Generated images of digit {digit}")

    cols = st.columns(5)
    for i in range(5):
        img_np = imgs[i].squeeze().numpy()
        cols[i].image(img_np, use_container_width=True, clamp=True, caption=f"Sample {i+1}")
