import os
import json
import streamlit as st
import speech_recognition as sr
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ---------------- CONFIG ----------------
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not set. Add it to your .env file.")

# ---------------- VECTORSTORE ----------------
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/paraphrase-xlm-r-multilingual-v1'
    )
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# ---------------- PROMPT TEMPLATE ----------------
CUSTOM_PROMPT_TEMPLATE = """"
You are a knowledgeable legal assistant specialized in Indian law. 
Answer concisely and only give the precise point relevant to the user's question.
Do NOT include extra context or full document text.

User question: {question}

Context (retrieved documents):
{context}

Instructions:
- Reply only with the exact answer.
- Use clear and professional language.
- If you don't know, say "I don't know".
- Keep it 2-6 sentences.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# ---------------- SPEECH RECOGNITION ----------------
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        st.warning("Sorry, I couldn't understand.")
    except sr.RequestError as e:
        st.error(f"Speech Recognition error: {e}")
    return None

# ---------------- TEXT TO SPEECH ----------------
def text_to_speech(text):
    try:
        from gtts import gTTS
        tts = gTTS(text)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        st.info("Listen to your answer below:")
        return mp3_fp.getvalue()
    except Exception:
        import pyttsx3
        import tempfile
        engine = pyttsx3.init()
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        engine.save_to_file(text, tmp_file.name)
        engine.runAndWait()
        with open(tmp_file.name, "rb") as f:
            audio_data = f.read()
        st.info("Listen to your answer below:")
        return audio_data

# ---------------- QUERY PROCESS ----------------
def process_query(user_query):
    vectorstore = get_vectorstore()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            groq_api_key=GROQ_API_KEY,
        ),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=False,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    response = qa_chain.invoke({'query': user_query})
    return response["result"]

# ---------------- SAVE CHAT ----------------
def save_conversation(messages):
    with open("conversation.json", "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

# ---------------- MAIN APP ----------------
def main():
    st.set_page_config(page_title="Legal Chatbot", layout="wide")
    st.title("JurisAI - Your Legal Assistant")

    # --- Initialize session state ---
    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}
    if 'current_chat' not in st.session_state:
        st.session_state.current_chat = None

    # --- Sidebar: chat history ---
    st.sidebar.header("Chat History")
    if st.sidebar.button("New Chat"):
        chat_name = f"Chat {len(st.session_state.conversations)+1}"
        st.session_state.conversations[chat_name] = []
        st.session_state.current_chat = chat_name

    if st.session_state.conversations:
        chat_names = list(st.session_state.conversations.keys())
        selected = st.sidebar.radio(
            "Select a chat:",
            chat_names,
            index=chat_names.index(st.session_state.current_chat) if st.session_state.current_chat in chat_names else 0
        )
        st.session_state.current_chat = selected

    if not st.session_state.current_chat:
        chat_name = f"Chat {len(st.session_state.conversations)+1}"
        st.session_state.conversations[chat_name] = []
        st.session_state.current_chat = chat_name

    # --- Load current chat ---
    messages = st.session_state.conversations[st.session_state.current_chat]

    # Display previous messages
    for msg in messages:
        role = "user" if msg["role"] == "user" else "assistant"
        st.chat_message(role).markdown(msg["content"])

    # --- Text input ---
    user_text = st.chat_input("Type your legal question...")

    # --- Mic button ---
    if st.button("Speak your question"):
        spoken_text = recognize_speech()
        if spoken_text:
            messages.append({"role": "user", "content": spoken_text})
            st.chat_message("user").markdown(spoken_text)

            answer = process_query(spoken_text)
            messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").markdown(answer)

            audio_bytes = text_to_speech(answer)
            st.audio(audio_bytes, format='audio/mp3')

            save_conversation(messages)

    # --- Handle typed text ---
    if user_text:
        messages.append({"role": "user", "content": user_text})
        st.chat_message("user").markdown(user_text)

        answer = process_query(user_text)
        messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)

        audio_bytes = text_to_speech(answer)
        st.audio(audio_bytes, format='audio/mp3')

        save_conversation(messages)

if __name__ == "__main__":
    main()
