import os
import logging
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader  # Use PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Set up logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for more detailed logs
logger = logging.getLogger(__name__)

st.set_page_config(page_title="KumR's PDF RAGGER using DeepSeek-R1", layout="wide")

# Load the embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Smaller model

# Streamlit interface
st.title("KumR's PDF RAGGER using DeepSeek-R1 ")

# Text input widget to get Groq API key from the user
groq_api_key = st.text_input("Enter your Groq API Key", type="password")

if not groq_api_key:
    st.warning("Please enter your Groq API Key to proceed.")
    st.stop()

# Set the Groq API key in the environment
os.environ["GROQ_API_KEY"] = groq_api_key

# Load the LLM from Groq
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0, request_timeout=60)  # Add timeout

def process_document_to_chroma_db(file_name):
    try:
        logger.info("Loading document...")
        loader = PyPDFLoader(os.path.join(working_dir, file_name))  # Use PyPDFLoader
        documents = loader.load()
        
        logger.info("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Smaller chunks
        texts = text_splitter.split_documents(documents)
        
        logger.info("Creating vector database...")
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embedding,
            persist_directory=os.path.join(working_dir, "doc_vectorstore")
        )
        logger.info("Document processed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return -1

def answer_question(user_question):
    try:
        logger.info("Loading vector database...")
        vectordb = Chroma(
            persist_directory=os.path.join(working_dir, "doc_vectorstore"),
            embedding_function=embedding
        )
        
        logger.info("Creating retriever...")
        retriever = vectordb.as_retriever()
        
        logger.info("Generating answer...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
        )
        response = qa_chain.invoke({"query": user_question})
        answer = response["result"]
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "An error occurred while generating the answer."

# File uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    try:
        # Define save path
        working_dir = os.getcwd()
        save_path = os.path.join(working_dir, uploaded_file.name)
        
        # Save the uploaded file
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the document
        st.info("Processing document...")
        result = process_document_to_chroma_db(uploaded_file.name)
        if result == 0:
            st.info("Document Processed Successfully")
        else:
            st.error("Failed to process document")
    except Exception as e:
        st.error(f"Error saving or processing file: {e}")

# Text widget to get user input
user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        try:
            st.info("Generating answer...")
            answer = answer_question(user_question)
            st.markdown("### DeepSeek-R1 Response")
            st.markdown(answer)
        except Exception as e:
            st.error(f"Error generating answer: {e}")