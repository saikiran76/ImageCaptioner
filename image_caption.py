import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

# Load the pre-trained model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_vector = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set max length and number of beams for generation
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams" : num_beams}

# Define function to predict captions
def predict(image_path):
    # Open the image and convert to RGB if necessary
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    # Extract image features and generate captions
    pixel_values = feature_vector(images=[image], return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds[0]

# Define Streamlit app
st.title("Image Captioning")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict caption
    caption = predict(uploaded_file)
    st.write("Caption:", caption)
