import streamlit as st
from transformers import pipeline
import os

# 1. Page Configuration (Browser Tab Title & Icon)
st.set_page_config(
    page_title="Smart Offline AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# 2. Sidebar Setup (Project Info & Controls)
with st.sidebar:
    st.title("🤖 Project Menu")
    st.markdown("**Project Name:** Offline AI Chatbot")
    st.markdown("**Tech Stack:** Python, Streamlit, HuggingFace Transformers")
    st.markdown("**Mode:** Local / Offline Ready ⚡")
    st.divider()
    
    # Clear Chat Button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Download Chat History Button (Extra Project Feature)
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        chat_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button(
            label="📥 Download Chat (.txt)",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )

# 3. Main Heading
st.title("🤖 Personal AI Assistant")
st.caption("Powered by Local Open-Source LLM (Works Offline after initial setup)")

# 4. Load AI Model (Cached so it loads only once and saves locally)
@st.cache_resource
def load_ai_model():
    # Yeh model size mein chhota (~1GB) aur fast hai, bina GPU ke CPU par aaram se chalta hai
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # './local_model' folder mein save ho jayega taaki offline chal sake
    os.makedirs("./local_model", exist_ok=True)
    
    chatbot = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"cache_dir": "./local_model"}
    )
    return chatbot

with st.spinner("AI Model load ho raha hai... (Pehli baar mein 2-3 minute lagenge)"):
    ai_pipe = load_ai_model()

# 5. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Main aapka AI Assistant hoon. Aaj main aapki kya madad kar sakta hoon?"}
    ]

# 6. Display Previous Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7. Handle User Input
if user_prompt := st.chat_input("Yahan apna sawal likhein..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            # System instructions + conversation history
            prompt_messages = [
                {"role": "system", "content": "You are a helpful, smart, and polite AI assistant. Answer clearly and concisely."}
            ] + st.session_state.messages

            outputs = ai_pipe(
                prompt_messages,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            # Extract generated reply
            bot_reply = outputs[0]["generated_text"][-1]["content"]
            st.markdown(bot_reply)

    # Save AI response in memory
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})