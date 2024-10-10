#%% packages
import os
import streamlit as st
import sys
import chromadb
from pypdf import PdfReader
import re
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import groq
from groq import Groq

# Initialize Groq client (ensure to replace with your API key)
client = Groq(api_key=("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"))

#%% data prep (same as before)
ipcc_report_file = "IPCC_AR6_WGII_TechnicalSummary.pdf"
reader = PdfReader(ipcc_report_file)
ipcc_texts = [page.extract_text().strip() for page in reader.pages]
ipcc_texts_filt = ipcc_texts[5:-5]
ipcc_wo_header_footer = [re.sub(r'\d+\nTechnicalSummary', '', s) for s in ipcc_texts_filt]
ipcc_wo_header_footer = [re.sub(r'\nTS', '', s) for s in ipcc_wo_header_footer]
ipcc_wo_header_footer = [re.sub(r'TS\n', '', s) for s in ipcc_wo_header_footer]

char_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0.2
)
texts_char_splitted = char_splitter.split_text('\n\n'.join(ipcc_wo_header_footer))

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0.2,
    tokens_per_chunk=256
)

texts_token_splitted = []
for text in texts_char_splitted:
    try:
        texts_token_splitted.extend(token_splitter.split_text(text))
    except:
        print(f"Error in text: {text}")
        continue

#%% Vector Database
chroma_client = chromadb.PersistentClient(path="db")
chroma_collection = chroma_client.get_or_create_collection("ipcc")
ids = [str(i) for i in range(len(texts_token_splitted))]
chroma_collection.add(
    ids=ids,
    documents=texts_token_splitted
)

#%% Image Model (for handling image inputs)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def process_image(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs

#%% RAG Function (for text query)
def rag(query, n_results=5):
    print("RAG")
    res = chroma_collection.query(query_texts=[query], n_results=n_results)
    docs = res["documents"][0]
    joined_information = ';'.join([f'{doc}' for doc in docs])
    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a climate specialist. Answer the user's question based on the provided document."},
            {"role": "user", "content": f"Question: {query}. \n Information: {joined_information}"}
        ]
    )
    content = chat_completion.choices[0].message.content
    print(content)
    return content, docs

#%% Streamlit Web Interface (multimodal)
st.header("Multimodal Climate Change Chatbot")

# Dropdown to select mode (Text or Image)
input_mode = st.selectbox("Select Input Mode", ["Text", "Image"])

if input_mode == "Text":
    # Text Input Mode
    user_query = st.text_input(label="", help="Ask here to learn about Climate Change", placeholder="What do you want to know about climate change?")
    
    if user_query:
        rag_response, raw_docs = rag(user_query)
        st.header("Raw Information")
        for i in range(len(raw_docs)):
            st.text(f"Raw Response {i}: {raw_docs[i]}")

        st.header("RAG Response")
        st.write(rag_response)

elif input_mode == "Image":
    # Image Input Mode
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(io.BytesIO(uploaded_image.read()))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process the image with CLIP model
        probabilities = process_image(image)
        st.write("Image Analysis Probabilities:", probabilities)
        
        # (Optional) You could also further process the image for more analysis depending on use case

st.write("Switch between text and image modes to ask questions or analyze climate-related images.")
