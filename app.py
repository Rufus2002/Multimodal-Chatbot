import os
import streamlit as st
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel
import groq
from groq import Groq

# Initialize Groq client (ensure to replace with your API key)
client = Groq(api_key=("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"))

#%% Image Model (for handling image inputs)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def process_image(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs

#%% Conversational Model (for text-based queries)
def converse_with_model(query):
    """Converse with the Llama model using Groq's API."""
    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a friendly assistant. Answer the user's question."},
            {"role": "user", "content": f"{query}"}
        ]
    )
    content = chat_completion.choices[0].message.content
    return content

#%% Streamlit Web Interface (multimodal)
st.header("Multimodal Conversational Chatbot")

# Dropdown to select mode (Text or Image)
input_mode = st.selectbox("Select Input Mode", ["Text", "Image"])

if input_mode == "Text":
    # Text Input Mode
    user_query = st.text_input(label="", help="Ask here", placeholder="What do you want to ask?")
    
    if user_query:
        response = converse_with_model(user_query)
        st.header("Chatbot Response")
        st.write(response)

elif input_mode == "Image":
    # Image Input Mode
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(io.BytesIO(uploaded_image.read()))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process the image with CLIP model
        probabilities = process_image(image)
        st.write("Image Analysis Probabilities:", probabilities)

st.write("Switch between text and image modes to interact with the chatbot.")
