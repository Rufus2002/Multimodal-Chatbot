import streamlit as st
import requests
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load the Llama model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained("huggingface/llama-7b")
model = LlamaForCausalLM.from_pretrained("huggingface/llama-7b")

# Define the Groq API URL and the API key
GROQ_API_URL = "https://api.groq.com/v1/generate"  # Replace with your Groq API endpoint
GROQ_API_KEY = "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"  # Replace with your actual Groq API key

# Function to get response from Groq API
def get_groq_response(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "max_tokens": 150
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    response_data = response.json()
    return response_data.get("text", "Sorry, I could not generate a response.")

# Function to get response from Llama model
def get_llama_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.title("Multimodal Chatbot")
st.write("Select a role and ask your question:")

# Dropdown for selecting chatbot role
roles = ["General", "Tech Support", "AI Assistant"]
selected_role = st.selectbox("Choose a role for the chatbot:", roles)

user_input = st.text_input("Your question:")
if st.button("Submit"):
    if user_input:
        st.write(f"You: {user_input}")

        # Combine the role with user input
        prompt = f"As a {selected_role} role, {user_input}"

        # Get responses from both APIs
        groq_response = get_groq_response(prompt)
        llama_response = get_llama_response(prompt)

        # Display the responses
        st.write(f"Groq Response: {groq_response}")
        st.write(f"Llama Response: {llama_response}")
    else:
        st.write("Please enter a question.")
