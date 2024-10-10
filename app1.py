import streamlit as st
from transformers import LlamaForCausalLM, LlamaTokenizer

# Initialize Llama model and tokenizer
model_name = "llama-13b"  # Change this to the LLaMA model you are using
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Define chatbot roles
roles = {
    "assistant": "You are a helpful assistant.",
    "mentor": "You are a knowledgeable mentor guiding the user.",
    "friend": "You are a friendly and supportive companion.",
    "customer_support": "You are a professional customer support representative."
}

# Chatbot function using LLaMA model
def generate_response(user_input, role):
    prompt = f"{roles[role]} {user_input}"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate response using the LLaMA model
    outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Streamlit UI
st.title("Multimodal Chatbot with LLaMA")

# Dropdown to select role
selected_role = st.selectbox("Select Chatbot Role", options=list(roles.keys()), format_func=lambda x: roles[x])

# Text area to input the user's message
user_input = st.text_area("Enter your message:")

# Button to send the message
if st.button("Send"):
    if user_input:
        response = generate_response(user_input, selected_role)
        st.write(f"**Chatbot ({selected_role}):** {response}")
    else:
        st.write("Please enter a message.")
