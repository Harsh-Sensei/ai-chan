import streamlit as st
import hmac
import random
import time
import os
import pickle
from datetime import datetime

import chatbots.mistral as mistral
import retreival_system.retriever as retriever

ASSISTANT_AVATAR = "assets/images/assistant.png"
USER_AVATAR = "assets/images/me.png"
STATE_PATH = "conversations/"
INIT_TEXT = "Below are some previous conversations to help you answer the questions that I will ask."
SEP_TEXT = f"Now let's continue with the current conversation. You can use the previous conversations to assistant me in the present session timestamped as {str(datetime.fromtimestamp(time.time()).strftime("%B %d, %Y %I:%M"))}: "

st.set_page_config(page_title="AI-Chan", page_icon=ASSISTANT_AVATAR)

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()


@st.cache_resource(show_spinner=False)
def get_chatbot():
    return mistral.Mistral()


@st.cache_resource(show_spinner=False)
def get_retriever():
    return retriever.BM25ColbertRetriever()

with st.spinner('AI-chan is getting ready...'):
    rag = get_retriever()
    chatbot = get_chatbot()

def new_chat():
    st.session_state.messages = []
    

st.sidebar.title("AI-Chan")
st.sidebar.button("New Chat", on_click=new_chat)
enable_memory = st.sidebar.checkbox("Enable Memory")
st.sidebar.image(ASSISTANT_AVATAR, caption="Chatbot", use_column_width=True)
st.sidebar.title("Chat History")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "new_state" not in st.session_state:
    st.session_state.new_state = True
    st.session_state.file_name = f"{time.time()}.pkl"
    st.cache_data.clear()
    
    
@st.cache_resource(show_spinner=False)
def get_sorted_files():
    # Fetch names of all files in the directory
    files = os.listdir(STATE_PATH)

    # Sort the file names in lexicographical order
    sorted_files = sorted(files)[::-1]
    if len(sorted_files) > 10 : 
        sorted_files = sorted_files[:10]
    
    return sorted_files

def load_chat(index):
    st.session_state.messages = st.session_state.chat_history[index]

sorted_files = get_sorted_files()

st.session_state.chat_history = []
for idx, file in enumerate(sorted_files):
    with open(os.path.join(STATE_PATH, file), "rb") as f:
        chat = pickle.load(f)
        st.session_state.chat_history.append(chat)
        with st.sidebar:
                st.button(label=str(datetime.fromtimestamp(int(file.split(".")[0]))
                                    .strftime("%B %d, %Y %I:%M")) + "  \n" + chat[0]["content"],
                        on_click=load_chat,
                        args=[idx],
                        use_container_width=True
                        )


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=ASSISTANT_AVATAR if message["role"]=="assistant" else USER_AVATAR):
        st.markdown(message["content"])

def refresh_chats():
    get_sorted_files.clear()
    rag.load_new_docs(path=STATE_PATH)

# Add a button to the sidebar
st.sidebar.button("Refresh Chats", on_click=refresh_chats)

# Accept user input
if prompt := st.chat_input("Hey, chat here...."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    retrieved_passage = ""
    if enable_memory:
        retrieved_passage = rag.get_retrieved_docs(prompt)
    if len(retrieved_passage) > 0:
        retrieved_passage = f"\n{INIT_TEXT}\n" + retrieved_passage + f"\n{SEP_TEXT}\n" + prompt
    else :
        retrieved_passage = prompt
    print("Retrieved Passage :", retrieved_passage)
    
    # Display assistant response in chat message container
    with st.chat_message('assistant',avatar=ASSISTANT_AVATAR):
        with st.spinner("Umm..."):
            last_content = st.session_state.messages[-1]["content"]
            st.session_state.messages[-1]["content"] = retrieved_passage
            response = chatbot.get_response(st.session_state.messages)
            st.session_state.messages[-1]["content"] = last_content
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    ## saving the conversation
    if st.session_state.new_state:
        st.session_state.new_state = False
        with open(os.path.join(STATE_PATH, st.session_state.file_name), "wb") as f:
            pickle.dump(st.session_state.messages, f)
    else :
        with open(os.path.join(STATE_PATH, st.session_state.file_name), "wb") as f:
            pickle.dump(st.session_state.messages, f)       

