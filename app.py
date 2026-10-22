import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Set the title here (Removed the 1.1B text)
st.set_page_config(page_title="AaritGPT", page_icon="🤖")
st.title("🤖 AaritGPT")

# 1. Load Model (Cached so it only loads once)
@st.cache_resource
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path = "aarit-lora-adapter"

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# 2. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Chat Input
if prompt := st.chat_input("Ask AaritGPT anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            messages = [{"role": "system", "content": "You are AaritGPT, a helpful AI created by Aarit."}]
            for msg in st.session_state.messages:
                messages.append(msg)

            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

            outputs = model.generate(
                **inputs, max_new_tokens=500, temperature=0.7,
                top_p=0.9, repetition_penalty=1.15, do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
