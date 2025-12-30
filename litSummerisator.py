import streamlit as st
import pdfplumber
import boto3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Translation function )
def translate_text(text, target_lang_code):
    try:
        translate = boto3.client('translate', region_name='ap-south-1')
        result = translate.translate_text(
            Text=text,
            SourceLanguageCode='en',
            TargetLanguageCode=target_lang_code
        )
        return result['TranslatedText']
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return text

# Function to extract text from a single page )
def extract_page_text(page):
    return page.extract_text() or ""

# Streamlit UI
st.set_page_config(page_title="Literature Summarizer", page_icon="📚")
st.title("Literature Summarizer")
st.subheader("Upload a PDF file to summarize and query its content.")

# Initialize session state
if 'pdf_hash' not in st.session_state:
    st.session_state.pdf_hash = None
if 'text' not in st.session_state:
    st.session_state.text = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'summary' not in st.session_state:
    st.session_state.summary = None

# UI elements
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
language = st.selectbox("Select Language", ("English", "Telugu", "Hindi", "Spanish", "French", "German", "Italian", "Portuguese"))
query = st.text_input("Enter your query or follow-up question (if any):")
lang_map = {"English": "en", "Telugu": "te", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de", "Italian": "it", "Portuguese": "pt"}

# Display chat history
if st.session_state.chat_history:
    st.subheader("Conversation History")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.write(f"**Q{i+1}:** {q}")
        st.write(f"**A{i+1}:** {a}")
        st.markdown("---")

# Summarization logic
if uploaded_file is not None and st.button("Summarize") and language:
    progressBar = st.progress(0)
    status_text = st.empty()

    # Compute PDF hash
    pdf_bytes = uploaded_file.read()
    pdf_hash = hashlib.md5(pdf_bytes).hexdigest()
    uploaded_file.seek(0)# Reset file pointer after reading

    # Extract text only if PDF is new
    if st.session_state.pdf_hash != pdf_hash or st.session_state.text is None:
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                num_pages = len(pdf.pages)
                with ThreadPoolExecutor() as executor:
                    text_chunks = list(executor.map(extract_page_text, pdf.pages))
                text = "".join(text_chunks)
                progressBar.progress(50)
                status_text.text("Text extraction complete!")
        except Exception as e:
            st.error(f"Failed to extract text from PDF: {e}")
            st.stop()

        if not text.strip():
            st.error("No text could be extracted from the PDF.")
            st.stop()

        st.session_state.text = text
        st.session_state.pdf_hash = pdf_hash

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=50)
        st.session_state.chunks = text_splitter.split_text(text)
        progressBar.progress(75)
        status_text.text("Text splitting complete!")
    else:
        text = st.session_state.text
        progressBar.progress(75)
        status_text.text("Using cached text and chunks.")

    # Initialize vector store and conversation chain
    if st.session_state.vectorstore is None:
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        st.session_state.vectorstore = FAISS.from_texts(st.session_state.chunks, embedding=embeddings)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
            llm=Ollama(model="llama3.2:1b"),
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=False
        )

    # Summarize text
    chain = load_summarize_chain(
        llm=Ollama(model="llama3.2:1b"),
        chain_type="map_reduce",
        map_prompt=PromptTemplate(
            input_variables=["text"],
            template="Summarize the following text {text} into a concise summary in about 150 words."
        ),
        verbose=False
    )
    documents = [Document(page_content=chunk) for chunk in st.session_state.chunks]
    with st.spinner("Summarizing..."):
        summary = chain.run({"input_documents": documents})
    progressBar.progress(100)
    status_text.text("Summarization complete!")

    # Translate summary if needed
    if language != "English":
        summary = translate_text(summary, lang_map[language])

    st.session_state.summary = summary
    st.write(f"Summary in {language}:")
    st.write(summary)

    # Download summary
    st.download_button(
        label="Download Summary as Text File",
        data=summary,
        file_name="summary.txt",
        mime="text/plain"
    )

# Query logic with separate button
if st.button("Ask Question", key="ask_question_button"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    elif st.session_state.vectorstore is None or st.session_state.conversation is None:
        st.warning("Please summarize a PDF first to enable querying.")
    else:
        with st.spinner("🔍 Searching for the answer..."):
            result = st.session_state.conversation({"question": query})
            answer = result["answer"]
            # Store query and answer in chat history
            st.session_state.chat_history.append((query, answer))
            st.success("💡 Answer:")
            st.write(answer)

# Display cached summary if available
if st.session_state.summary:
    st.subheader(f"Summary in {language}:")
    st.write(st.session_state.summary)
    st.download_button(
        label="Download Summary as Text File",
        data=st.session_state.summary,
        file_name="summary.txt",
        mime="text/plain",
        key="download_summary_button"  # Added unique key for download button
    )