import os
import streamlit as st
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel
import groq
from groq import Groq

# Initialize Groq client
client = Groq(api_key=("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"))

#%% Initialize Models
# Load CLIP Model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def process_image(image):
    """Process the image using CLIP to generate probabilities."""
    try:
        # Process the image through CLIP processor
        inputs = clip_processor(images=image, text=[""], return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        return probs
    except Exception as e:
        raise RuntimeError(f"Error in processing image: {e}")

def converse_with_model(query, role):
    """Converse with the Llama model using Groq's API."""
    try:
        chat_completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": f"You are a {role}. Answer the user's question."},
                {"role": "user", "content": query}
            ]
        )
        content = chat_completion.choices[0].message.content
        return content
    except Exception as e:
        return f"Error in conversation: {e}"

#%% Streamlit Web Interface (multimodal)
st.header("Multimodal Conversational Chatbot")

# Dropdown to select mode (Text or Image)
input_mode = st.selectbox("Select Input Mode", ["Text", "Image"])

# Dropdown to select role
role_selection = st.selectbox("Select Role", ["Joker", "HR Manager", "Stock Researcher", "Scientist", "Teacher"])

if input_mode == "Text":
    # Text Input Mode
    st.subheader("Text Mode")
    user_query = st.text_input(label="", help="Ask your question here", placeholder="What do you want to ask?")

    if user_query:
        response = converse_with_model(user_query, role_selection)
        st.header("Chatbot Response")
        st.write(response)

elif input_mode == "Image":
    # Image Input Mode
    st.subheader("Image Mode")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        try:
            # Display uploaded image
            image = Image.open(io.BytesIO(uploaded_image.read()))
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Process image and display probabilities
            probabilities = process_image(image)
            st.header("Image Analysis")
            st.write("Image Analysis Probabilities:", probabilities)
        except Exception as e:
            st.error(f"Error processing image: {e}")

st.write("Switch between Text and Image modes to interact with the chatbot.")

