import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import time

# ==========================================
# 1. CUSTOM CSS (The UI Magic)
# ==========================================
custom_css = """
<style>
    /* Center the main title and add a gradient */
    .main-title {
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Make the chat input look nicer */
    .stChatInput {
        border-radius: 20px !important;
        border: 1px solid #4d4d4f !important;
        background-color: #1e1e24 !important;
    }
    
    /* Custom Thinking Animation (Bouncing Dots) */
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        height: 20px;
    }
    .typing-indicator span {
        background-color: #92FE9D;
        border-radius: 50%;
        display: inline-block;
        height: 8px;
        width: 8px;
        margin-right: 5px;
        animation: bounce 1.4s infinite ease-in-out both;
    }
    .typing-indicator span:nth-of-type(1) { animation-delay: -0.32s; }
    .typing-indicator span:nth-of-type(2) { animation-delay: -0.16s; }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# 2. PAGE CONFIG & HEADER
# ==========================================
st.set_page_config(page_title="AaritGPT", page_icon="🤖", layout="centered")

# Render the centered headers
st.markdown('<h1 class="main-title">AaritGPT</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Your custom-trained 1.1B AI Assistant</p>', unsafe_allow_html=True)

# ==========================================
# 3. LOAD MODEL (Cached)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path = "aarit-lora-adapter"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer

# Show a nice loading message on first run
with st.spinner("Waking up AaritGPT (this happens only once)..."):
    model, tokenizer = load_model()

# ==========================================
# 4. CHAT INTERFACE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history with custom avatars
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask AaritGPT anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant", avatar="🤖"):
        # 1. Render the custom bouncing dots animation
        message_placeholder = st.empty()
        message_placeholder.markdown(
            '<div class="typing-indicator"><span></span><span></span><span></span></div>', 
            unsafe_allow_html=True
        )
        
        # 2. Generate the text
        messages = [{"role": "system", "content": "You are AaritGPT, a helpful AI created by Aarit."}]
        for msg in st.session_state.messages:
            messages.append(msg)
            
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        outputs = model.generate(
            **inputs, max_new_tokens=150, temperature=0.7, 
            top_p=0.9, repetition_penalty=1.15, do_sample=True, 
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 3. Replace the bouncing dots with the actual response
        message_placeholder.markdown(response)
            
    # Save response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
