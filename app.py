import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
import os
import tempfile
import speech_recognition as sr
import pyttsx3

# Initialize session state
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    if 'embeddings' not in st.session_state:
        st.session_state['embeddings'] = None  # Initialize embeddings
        st.session_state['vector_store'] = None  # Initialize vector store
        st.session_state['chain'] = None  # Initialize chain

# Initialize pyttsx3 for text-to-speech
def init_tts():
    engine = pyttsx3.init()
    return engine

# Function to use TTS to read the chatbot's response
def speak_text(text, tts_engine):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Function to capture voice input using speech recognition
def capture_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening for your question...")
        audio = r.listen(source)

    try:
        st.write("Recognizing speech...")
        voice_input = r.recognize_google(audio)
        st.write(f"You said: {voice_input}")
        return voice_input
    except sr.UnknownValueError:
        st.write("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        st.write(f"Could not request results; {e}")
    
    return None

# Function to handle the conversation with the chatbot
def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

# Display the chat history and manage user input (text and voice)
def display_chat_history(chain, tts_engine):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
            voice_button = st.form_submit_button(label='Speak')  # Button to trigger voice input
            submit_button = st.form_submit_button(label='Send')  # Button to send text input

        if voice_button:
            # Capture voice input and set it as the user_input
            user_input = capture_voice_input()

        # Check if either text or voice input is provided, and process it
        if (submit_button or user_input) and user_input:  # Ensure either button works with valid input
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])  # Use the input for chat

            # Store chat history for display
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # Display chat history with a speaker icon for TTS
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                user_message = st.session_state["past"][i]
                bot_response = st.session_state['generated'][i]

                # Display user and bot messages
                message(user_message, is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(bot_response, key=str(i), avatar_style="fun-emoji")

                # Add speaker icon for TTS output
                if st.button("🔊", key=f"tts_button_{i}"):
                    speak_text(bot_response, tts_engine)  # Use TTS to read the response

# Create a conversational chain for the chatbot
def create_conversational_chain(vector_store):
    model_path = "C:\mistrall_llm\mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # Updated model path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file does not exist at: {model_path}")

    try:
        llm = LlamaCpp(
            streaming=True,
            model_path=model_path,
            temperature=0.75,
            top_p=1,
            verbose=True,
            n_ctx=4096
        )
    except Exception as e:
        raise

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )
    return chain

# Add custom CSS styling
def add_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background-image: url('https://i.pinimg.com/236x/63/cb/e4/63cbe4026e6697c7ff238c58803d49c4.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }
    input {
        background-color: rgba(255, 255, 255, 0.8);
        color: black;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
    }
    .streamlit_chat {
        font-family: "Arial", sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Main function
def main():
    initialize_session_state()
    add_custom_css()  # Add custom CSS for "The Boys" theme
    st.title("Multi-PDF ChatBot using Mistral-7B-Instruct :books:")
    
    tts_engine = init_tts()  # Initialize the TTS engine

    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(text)

        if st.session_state['embeddings'] is None:
            st.session_state['embeddings'] = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device': 'cpu'})

        if st.session_state['vector_store'] is None:
            st.session_state['vector_store'] = FAISS.from_documents(text_chunks, embedding=st.session_state['embeddings'])

        if st.session_state['chain'] is None:
            st.session_state['chain'] = create_conversational_chain(st.session_state['vector_store'])

        display_chat_history(st.session_state['chain'], tts_engine)  # Pass TTS engine to the chat history display

if __name__ == "__main__":
    main()

