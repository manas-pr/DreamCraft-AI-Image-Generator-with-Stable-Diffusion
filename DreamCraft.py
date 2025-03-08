import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Model setup
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Streamlit UI
st.title("🚀 Stable Diffusion Image Generator")
st.write("Generate stunning AI art directly from text prompts!")

# Input form
prompt = st.text_input("Enter your prompt:", "a photo of a dog flying in Mars")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)
        image.save("generated_image.png")
        st.success("✅ Image successfully generated!")
        st.download_button("Download Image", open("generated_image.png", "rb"), "generated_image.png")

