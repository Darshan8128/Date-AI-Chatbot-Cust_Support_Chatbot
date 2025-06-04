import streamlit as st
import os
from dotenv import load_dotenv
from ui import setup_ui, display_chat_messages
from utils import (
    load_vectorstore,
    get_ai_response,
    extract_text_from_file,
    whisper_transcribe,
    elevenlabs_tts
)
# Removed: from streamlit.runtime.scriptrunner import RerunException # <--- This import is no longer needed

load_dotenv()

# API keys loaded from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

VECTOR_STORE_DIR = "vectorstore"
VECTOR_STORE_PATH = os.path.join(VECTOR_STORE_DIR, "company_vectorstore")
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Initialize Streamlit session state variables with consistent keys
for key, default in [
    ('chat_history', []),
    ('conversation_history', []),  # For LangChain's internal chat history
    ('vectorstore', None),
    ('uploaded_files', []), # Stores filenames of uploaded documents (not content)
    ('knowledge_updated', False), # Flag to trigger vectorstore reload
    ('user_input', ""), # For the text input box
    ('recording_state', "ready"), # State for microphone recording
    ('speech_text', ""), # For transcribed speech
    ('files', []),  # To store the *content* and names of uploaded files for knowledge base
    ('language_code', 'en'),  # Default language for translations/TTS
]:
    if key not in st.session_state:
        st.session_state[key] = default

def reset_chat():
    """Resets chat history and triggers a rerun."""
    st.session_state.chat_history = []
    st.session_state.conversation_history = []
    st.session_state.user_input = ""
    st.rerun() # Correct way to trigger rerun

def start_recording():
    """Sets recording state and triggers a rerun."""
    st.session_state.recording_state = "Listening"
    st.rerun() # Correct way to trigger rerun

def record_and_transcribe():
    """Records audio, transcribes it, and updates user input."""
    status_area = st.empty()
    try:
        status_area.markdown('Listening...', unsafe_allow_html=True)
        transcript = whisper_transcribe()
        st.session_state.user_input = transcript
        status_area.markdown('Done! Click "Send" to submit.', unsafe_allow_html=True)
        st.session_state.recording_state = "ready"
    except Exception as e:
        status_area.error(f"Error: {e}")
        st.session_state.recording_state = "ready"
    # No rerun here, as this function is called inside the UI loop and user needs to click Send

def send_query():
    """Processes user query, gets AI response, and updates chat history."""
    if st.session_state.user_input.strip(): # Check for non-empty input
        query = st.session_state.user_input.strip()
        st.session_state.chat_history.append({"role": "user", "content": query})
        language_code = st.session_state.get('language_code', 'en')

        with st.spinner("Thinking..."):
            answer, sources = get_ai_response(query, lang=language_code)
            # Only attempt TTS if ELEVENLABS_API_KEY is available and answer is not empty
            audio_file = elevenlabs_tts(answer, lang=language_code) if answer and ELEVENLABS_API_KEY else None

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "audio": audio_file,
                "sources": sources
            })

        st.session_state.user_input = "" # Clear input field
        st.rerun() # Correct way to trigger rerun

def remove_file(filename):
    """Removes a file from uploads and updates session state."""
    filepath = os.path.join("uploads", filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    # Remove from st.session_state.uploaded_files (list of names)
    st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f != filename]

    # Remove from st.session_state.files (list of dicts containing content)
    st.session_state.files = [f_info for f_info in st.session_state.files if f_info.get("name") != filename]


    st.session_state.knowledge_updated = True # Flag vectorstore for rebuilding
    st.rerun() # Correct way to trigger rerun

def main():
    """Main function to set up and run the Streamlit application."""
    setup_ui(
        reset_chat=reset_chat,
        remove_file=remove_file,
        extract_text_from_file=extract_text_from_file
    )

    # Display logo - update path if needed. Consider a relative path like "Logo.jpg"
    # if the image is in the same directory as this script.
    st.image(r"C:\Users\darsh\OneDrive\Desktop\Chatbot 2\Date AI.png", width=200)

    st.title("Date AI Chatbot")
    st.markdown("Ask questions about Date AI's products and services.")

    chat_container = st.container()
    with chat_container:
        display_chat_messages()

    st.markdown("#### Ask a question")
    col1, col2, col3 = st.columns([6, 1, 1])

    with col1:
        st.session_state.user_input = st.text_input(
            "Type your question here...",
            value=st.session_state.user_input,
            label_visibility="collapsed"
        )

    with col2:
        if st.session_state.recording_state == "ready":
            if st.button("🎤", help="Click to speak"):
                start_recording()
        else:
            # This function doesn't trigger a rerun; the 'Send' button or user interaction will.
            record_and_transcribe()

    with col3:
        if st.button("Send"):
            send_query()

    with st.expander("How to use Date AI chatbot", expanded=False):
        st.markdown("""
        ### Instructions
        1. Ask questions by typing or speaking.
        2. Select text from responses as needed.
        3. Change language settings in the sidebar.
        4. Upload Documents to update AI knowledge.
        5. Reset the chat anytime from the sidebar.
        """)

if __name__ == "__main__":
    main()