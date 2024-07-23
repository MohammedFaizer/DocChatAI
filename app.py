# import streamlit as st
# import os
# from langchain_community.document_loaders import PDFMinerLoader
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain_community.llms import HuggingFacePipeline
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
# import torch
# st.set_page_config(page_title="Cyces.AI", page_icon="🤖")
# st.title("Cyces PDF Chatbot.ai")

# # Custom CSS for chat messages
# st.markdown("""
#     <style>
#         .user-message {
#             text-align: right;
#             background-color: #4a4a4a;
#             color: white;
#             padding: 10px;
#             border-radius: 10px;
#             margin-bottom: 10px;
#             display: inline-block;
#             width: fit-content;
#             max-width: 70%;
#             margin-left: auto;
#             box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
#         }
#         .assistant-message {
#             text-align: left;
#             background-color: #c2860e;
#             color: white;
#             padding: 10px;
#             border-radius: 10px;
#             margin-bottom: 10px;
#             display: inline-block;
#             width: fit-content;
#             max-width: 70%;
#             margin-right: auto;
#             box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
#         }
#     </style>
# """, unsafe_allow_html=True)

# def get_file_size(file):
#     file.seek(0, os.SEEK_END)
#     file_size = file.tell()
#     file.seek(0)
#     return file_size

# # Add a sidebar for model selection and user details

# model_options = ["MBZUAI/LaMini-T5-738M", "google/flan-t5-base", "google/flan-t5-small"]
# selected_model = st.sidebar.selectbox("Choose Model", model_options)
# st.sidebar.write("---")
# st.sidebar.write("\n")
# uploaded_file = st.sidebar.file_uploader("Upload file", type=["pdf"])


# @st.cache_resource
# def initialize_qa_chain(filepath, CHECKPOINT):
#     loader = PDFMinerLoader(filepath)
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
#     splits = text_splitter.split_documents(documents)

#     # Create embeddings 
#     embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectordb = FAISS.from_documents(splits, embeddings)

#     # Initialize model
#     TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT)
#     BASE_MODEL = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT, device_map=torch.device('cpu'), torch_dtype=torch.float32)
#     pipe = pipeline(
#         'text2text-generation',
#         model=BASE_MODEL,
#         tokenizer=TOKENIZER,
#         max_length=256,
#         do_sample=True,
#         temperature=0.3,
#         top_p=0.95,
#     )

#     llm = HuggingFacePipeline(pipeline=pipe)

#     # Build a QA chain
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectordb.as_retriever(),
#     )
#     return qa_chain

# def process_answer(instruction, qa_chain):
#     generated_text = qa_chain.run(instruction)
#     return generated_text

# if uploaded_file is not None:
#     os.makedirs("docs", exist_ok=True)
#     filepath = os.path.join("docs", uploaded_file.name)
#     with open(filepath, "wb") as temp_file:
#         temp_file.write(uploaded_file.read())
#         temp_filepath = temp_file.name

#     with st.spinner('Embeddings are in process...'):
#         qa_chain = initialize_qa_chain(temp_filepath, selected_model)
# else:
#     qa_chain = None

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     if message["role"] == "user":
#         st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
#     else:
#         st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)

# # React to user input
# if prompt := st.chat_input("Ask a related question?"):
#     # Display user message in chat message container
#     st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     if qa_chain:
#         # Generate response
#         response = process_answer({'query': prompt}, qa_chain)
#     else:
#         # Prompt to upload a file
#         response = "Please upload a PDF file to enable the chatbot."

#     # Display assistant response in chat message container
#     st.markdown(f"<div class='assistant-message'>{response}</div>", unsafe_allow_html=True)
    
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})

import streamlit as st
import os
from PyPDF2 import PdfMerger
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

st.set_page_config(page_title="Cyces.AI", page_icon="🤖")
st.title("Cyces PDF Chatbot")
# Custom CSS for chat messages
st.markdown("""
    <style>
    .user-message {
    text-align: right;
    background-color: #4a4a4a;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    display: flex;
    justify-content: flex-end; /* Aligns content to the right */
    width: fit-content;
    max-width: 70%;
    margin-left: auto;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
}

.assistant-message {
    text-align: left;
    background-color: #c2860e;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    display: flex;
    justify-content: flex-start; /* Aligns content to the left */
    width: fit-content;
    max-width: 70%;
    margin-right: auto;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
}
    </style>
""", unsafe_allow_html=True)

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

def merge_pdfs(filepaths, output_path):
    merger = PdfMerger()
    for filepath in filepaths:
        merger.append(filepath)
    merger.write(output_path)
    merger.close()

# Add a sidebar for model selection and user details



uploaded_files = st.sidebar.file_uploader("Upload files", type=["pdf"], accept_multiple_files=True)
st.sidebar.write("---")
model_options = ["MBZUAI/LaMini-T5-738M", "google/flan-t5-small"]
# "google/flan-t5-base",
selected_model = st.sidebar.selectbox("Choose Model", model_options)

@st.cache_resource
def initialize_qa_chain(filepath, CHECKPOINT):
    loader = PDFMinerLoader(filepath)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Create embeddings 
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(splits, embeddings)

    # Initialize model
    TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT)
    BASE_MODEL = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT, device_map=torch.device('cpu'), torch_dtype=torch.float32)
    pipe = pipeline(
        'text2text-generation',
        model=BASE_MODEL,
        tokenizer=TOKENIZER,
        max_length=256,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Build a QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
    )
    return qa_chain

def process_answer(instruction, qa_chain):
    generated_text = qa_chain.run(instruction)
    return generated_text

if uploaded_files:
    os.makedirs("docs", exist_ok=True)
    filepaths = []

    for uploaded_file in uploaded_files:
        filepath = os.path.join("docs", uploaded_file.name)
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
            filepaths.append(temp_file.name)

    combined_pdf_path = "docs/combined.pdf"
    merge_pdfs(filepaths, combined_pdf_path)

    with st.spinner('Embeddings are in process...'):
        qa_chain = initialize_qa_chain(combined_pdf_path, selected_model)
else:
    qa_chain = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("Ask a related question?"):
    # Display user message in chat message container
    st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if qa_chain:
        # Generate response
        response = process_answer({'query': prompt}, qa_chain)
    else:
        # Prompt to upload a file
        response = "Please upload a PDF file to enable the chatbot."

    # Display assistant response in chat message container
    st.markdown(f"<div class='assistant-message'>{response}</div>", unsafe_allow_html=True)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
