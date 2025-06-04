import streamlit as st

#page configuration
def setup_ui(reset_chat,remove_file,extract_text_from_file):
    # set page config
    st.set_page_config(
        page_title="Date AI Chatbot",
        page_icon=":robot:",
        layout="wide" 
    )
    
    #apply  custom css
    apply_custom_css()
    
    # sidebar elements
    with st.sidebar:
        st.title("Date AI Chatbot")
        st.markdown("---")

        
        # language selection
        languages = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Russian": "ru",
            "Hindi": "hi" 
        }
        selected_language = st.selectbox("Select Language", list(languages.keys()),index=0)
        
        st.session_state.language = languages[selected_language]
        
        # file upload
        uploaded_file = st.file_uploader("Upload additional documents", 
                                         type=["pdf", "txt","md","html", "docx"],
                                         help="Upload files to enhance the AI's knowledge base.")
        
        if uploaded_file:
            file_text = extract_text_from_file(uploaded_file)
            if file_text:
                file_info = {
                    "name": uploaded_file.name,
                    "content": file_text
                }
                if file_info not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(file_info)
                    st.session_state.knowledge_updated = True
                    st.success(f"Added '{uploaded_file.name}' to knowledge base")
                    
                    
                # show uploaded files
                if st.session_state.uploaded_files:
                    st.markdown("### Uploaded Files")
                    for i,file_info in enumerate(st.session_state.uploaded_files):
                        st.markdown(f"**{i+1}. {file_info['name']}**")
                        remove_button_key = f"remove_file_{i}"
                        if st.button(f"Remove {file_info['name']}",
                        key=remove_button_key):
                            remove_file(i)
                            
                # reset chat button
                if st.button("Reset chat",key="reset_chat"):
                    reset_chat()

# avatar images
def get_avatar_html(avatar_type):
    if avatar_type == "user":
        avatar_url = "https://api.dicebear.com/7.x/personas/svg?seed=user"
    else:
        avatar_url = "https://api.dicebear.com/7.x/bottts/svg?seed=techease" 
        
    return f'<img src="{avatar_url}" alt="{avatar_type} avatar">'

# Display chat messages with selectable text
def display_chat_messages():
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            col1,col2 = st.columns([1,9])
            with col1:
                st.markdown(f'<div class="avatar">{get_avatar_html("user")}</div', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="chat-message user"><div class="message">{message["content"]}</div></div>', unsafe_allow_html=True)
        else:
            col1,col2 = st.columns([1,9])
            with col1:
                st.markdown(f'<div class="avatar">{get_avatar_html("bot")}</div>', unsafe_allow_html=True)
            with col2:
                # Make sure the text is properly escaped but still selectable
                st.markdown(f'<div class="chat-message bot"><div class="message">{message["content"]}</div></div>', unsafe_allow_html=True)
                
                # Add audio player if available
                if "audio" in message and message["audio"]:
                    st.audio(message["audio"])
                    
                    
# Apply custom CSS
def apply_custom_css():
    st.markdown("""
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }

            .chat-message {
                padding: 15px;
                border-radius: 12px;
                margin-bottom: 15px;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
                font-size: 16px;
                line-height: 1.6;
            }

            .user {
                background-color: #d0f0fd;
                text-align: right;
                color: #00334e;
            }

            .bot {
                background-color: #f9e4ec;
                text-align: left;
                color: #4a0033;
            }

            .avatar {
                display: flex;
                justify-content: center;
                align-items: center;
                width: 42px;
                height: 42px;
                border-radius: 50%;
                overflow: hidden;
                box-shadow: 0 1px 4px rgba(0,0,0,0.2);
            }

            .avatar img {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
        </style>
    """, unsafe_allow_html=True)