import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import torch.backends.cudnn as cudnn

# Page Configuration for faster loading
st.set_page_config(page_title="Stable Diffusion", layout="wide")

# Model setup
model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float32
    ).to(device)
    
    if device == "cuda":
        cudnn.benchmark = True
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()

    pipe.to(memory_format=torch.channels_last)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    return pipe

pipe = load_model()

# Streamlit UI
st.title("🚀 Stable Diffusion Image Generator")
st.write("Generate stunning AI art directly from text prompts!")

# Input form
prompt = st.text_input("Enter your prompt:")

if st.button("Generate Image", use_container_width=True):
    with st.status("🖌️ Generating... Please wait...", expanded=True) as status:
        torch.cuda.empty_cache()  # Clear GPU cache before generation
        image = pipe(prompt, height=256, width=256, num_inference_steps=15).images[0]
        image.save("generated_image.jpg", format="JPEG", quality=85)
        st.image("generated_image.jpg", caption="Generated Image", use_container_width=True)
        st.success("✅ Image successfully generated!")
        st.download_button("Download Image", open("generated_image.jpg", "rb"), "generated_image.jpg")
