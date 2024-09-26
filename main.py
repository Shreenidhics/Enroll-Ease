import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq 

# Load environment variables
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

# Load the PDF document
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return docs
# This function uses PyPDFLoader to load the contents of a PDF document. It reads the PDF and returns its content as a list of documents.

# Split text into chunks
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    return documents
# The function splits the document content into chunks of 1000 characters with 200 characters overlapping, 
# using a RecursiveCharacterTextSplitter to ensure that text chunks are manageable for embeddings and retrieval.

# Initialize embeddings and vector store
def initialize_embeddings(documents):
    embeddings = OllamaEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    return db
# This initializes the OllamaEmbeddings model and then uses FAISS to index the embedded document chunks for fast similarity search.

# Initialize language model (LLM) with Groq
def initialize_llm():
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
    return llm
# This function initializes the Groq LLM with a specific model (mixtral-8x7b-32768) and an API key loaded from the environment.

# Create prompt template
def create_prompt():
    prompt = ChatPromptTemplate.from_template("""
    You are an assistant who provides answers to the questions of the user. 
    Get the information from the pdf and provide correct and precise answers. 
    If you don't know the answer then say "I am sorry you might have to contact the college for more info on this XXXXX"
    <context>
    {context}
    </context>
    Question: {input}
    """)
    return prompt
# A template is created for the chatbot's responses. It instructs the model on how to respond to user queries, using context from the retrieved documents.

# Create document chain and retrieval chain
def create_chains(llm, prompt, db):
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)
    return retrieval_chain
# This sets up two chains: a document chain to process the context and a retrieval chain to search the FAISS vector store and find relevant chunks from the documents based on the query.

# Function to process question and return answer
def process_question(retrieval_chain, question):
    response = retrieval_chain.invoke({"input": question})
    return response['answer']
# This function takes a question from the user, passes it to the retrieval chain, and returns the generated answer.

# Preload PDF documents and embeddings
def preload_data(pdf_path):
    docs = load_pdf(pdf_path)
    documents = split_documents(docs)
    db = initialize_embeddings(documents)
    return db
# This function preloads the PDF by loading it, splitting it into chunks, and then creating a FAISS-based database of embeddings.

# Streamlit chatbot UI
def run_streamlit():
    st.title("ENROLL EASE")

    # Add custom styling
    st.markdown("""
        <style>
            .sidebar .sidebar-content {
                padding: 20px;
                background-color: #f8f9fa;
            }
            .sidebar .sidebar-content h3 {
                margin-top: 0;
            }
            .chat-box {
                margin-bottom: 10px;
                padding: 10px;
                border-radius: 5px;
                background-color: #f1f1f1;
            }
            .user-message {
                font-weight: bold;
                color: #000000;
                margin-bottom: 5px
            }
            .bot-message {
                margin-top: 5px;
                color: #0a0a0a;
            }
            .button-container button {
                display: block;
                width: 100%;
                margin-bottom: 10px;
                padding: 10px;
                text-align: center;
                border: none;
                background-color: #007bff;
                color: white;
                font-size: 16px;
                cursor: pointer;
                border-radius: 5px;
            }
            .button-container button:hover {
                background-color: #0056b3;
            }
        </style>
    """, unsafe_allow_html=True)
    # Custom styling to create a clean interface, defining the layout for the chat, buttons, and side panel.

    st.sidebar.title("Ask Queries Regarding:")

    # Pre-provided PDF paths
    pdf_paths = {
        "college_info":"about_college.pdf",
        "seatsmatrix":"seatsmatrix.pdf",
        "admission_process":"admissionprocess.pdf",
        "Cutoff": "cuttoffs.pdf",
        "fees": "feesstructure.pdf",
    }
    # Predefined paths for different PDFs, categorized for different topics.

    # Initialize selection state
    if 'pdf_path' not in st.session_state:
        st.session_state.pdf_path = None
    # Initialize the session state to keep track of the selected PDF.

    # Add buttons with equal size and centered text
    st.sidebar.markdown("<div class='button-container'>", unsafe_allow_html=True)
    if st.sidebar.button("About College"):
        st.session_state.pdf_path = pdf_paths["college_info"]
        st.session_state.pop('db', None)  # Clear previous embeddings
        st.session_state.db = preload_data(st.session_state.pdf_path)
        st.experimental_rerun()

    if st.sidebar.button("Admission Process"):
        st.session_state.pdf_path = pdf_paths["admission_process"]
        st.session_state.pop('db', None)  # Clear previous embeddings
        st.session_state.db = preload_data(st.session_state.pdf_path)
        st.experimental_rerun()

    if st.sidebar.button("Cut-off"):
        st.session_state.pdf_path = pdf_paths["Cutoff"]
        st.session_state.pop('db', None)  # Clear previous embeddings
        st.session_state.db = preload_data(st.session_state.pdf_path)
        st.experimental_rerun()

    if st.sidebar.button("Seat Allotment"):
        st.session_state.pdf_path = pdf_paths["seatsmatrix"]
        st.session_state.pop('db', None)  # Clear previous embeddings
        st.session_state.db = preload_data(st.session_state.pdf_path)
        st.experimental_rerun()

    if st.sidebar.button("Fees Structure"):
        st.session_state.pdf_path = pdf_paths["fees"]
        st.session_state.pop('db', None)  # Clear previous embeddings
        st.session_state.db = preload_data(st.session_state.pdf_path)
        st.experimental_rerun()
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    # Buttons to select the specific PDFs to load. Each button resets the session state, clears previous data, and loads the selected PDF.

    # Display chat history and input if PDF is selected
    if st.session_state.pdf_path is not None:
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        # Displaying the chat history from the session state.

        for chat in st.session_state.chat_history:
            st.markdown(f"<div class='chat-box'><span class='user-message'>{chat['question']}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-box'><span class='bot-message'><strong>Bot:</strong> {chat['answer']}</span></div>", unsafe_allow_html=True)
        # Loop through and render each question-answer pair from the chat history.

        question = st.text_input("Enter your question:", key="input", label_visibility="collapsed", placeholder="Type your question here...")

        if st.button("Send") and question:
            if 'db' in st.session_state:
                llm = initialize_llm()
                prompt = create_prompt()
                retrieval_chain = create_chains(llm, prompt, st.session_state.db)
                answer = process_question(retrieval_chain, question)
                st.session_state.chat_history.append({"question": question, "answer": answer})
                st.experimental_rerun()
            else:
                st.warning("Please select a category first.")
       # When a question is submitted, the selected PDF's embeddings are used for retrieval and the language model generates an answer, which is displayed.
if __name__ == "__main__":
    run_streamlit()
