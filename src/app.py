import streamlit as st
from model import (create_conversational_rag_chain, get_session_history, load_pdf_documents)

# Streamlit UI
st.markdown("<h1 style='color: blue;'>Conversation With PDF</h1>", unsafe_allow_html=True)
st.write("Upload PDF's and chat with content")

# store chat history
if 'store' not in st.session_state:
    st.session_state.store = {}

# File upload
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = load_pdf_documents(uploaded_files)

    # Create embeddings
    vectorstore, retriever = documents['vectorstore'], documents['retriever']

    # conversational chain 
    conversational_rag_chain, handle_unrelated_query = create_conversational_rag_chain(retriever)

    # User input
    session_id = st.text_input("Session ID", value="default_session")
    
    if session_id.strip() == "":
        session_id = "default_session"

    user_input = st.text_input("Your question:")

    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        # Handle unrelated
        response = handle_unrelated_query(response, user_input)
        st.write("Assistant:", response['answer'])
        st.write("Chat History:", session_history.messages)
