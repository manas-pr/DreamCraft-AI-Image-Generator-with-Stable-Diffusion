import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import torch.backends.cudnn as cudnn

# Model setup
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

if device == "cuda":
    cudnn.benchmark = True

pipe.enable_attention_slicing()
pipe.to(memory_format=torch.channels_last)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Streamlit UI
st.title("🚀 Stable Diffusion Image Generator")
st.write("Generate stunning AI art directly from text prompts!")

# Input form
prompt = st.text_input("Enter your prompt:", "a photo of a dog flying in Mars")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image = pipe(prompt, height=384, width=384, num_inference_steps=25).images[0]
        st.image(image, caption="Generated Image", use_column_width=True)
        image.save("generated_image.png")
        st.success("✅ Image successfully generated!")
        st.download_button("Download Image", open("generated_image.png", "rb"), "generated_image.png")
