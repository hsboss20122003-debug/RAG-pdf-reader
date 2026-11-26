import os
import tempfile
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

st.set_page_config(
    page_title="Statistics RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Statistics RAG Assistant")
st.caption("Ask questions based on your uploaded Statistics PDF.")
st.divider()

uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])

default_pdf_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "STATS.pdf"
)


@st.cache_resource
def create_rag_components(pdf_path: str):
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    docs = PyPDFLoader(pdf_path).load()
    if not docs:
        raise ValueError("The PDF contains no readable text.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80
    )
    chunks = text_splitter.split_documents(docs)

    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    client = QdrantClient(location=":memory:")
    client.create_collection(
        collection_name="rag_collection",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="rag_collection",
        embedding=embeddings
    )
    vectorstore.add_documents(chunks)

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY not found. Add it to the .env file."
        )

    llm = ChatNVIDIA(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_api_key=api_key,
        temperature=0.2,
        max_completion_tokens=2500
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert tutor in Statistical Techniques. "
            "Answer accurately based ONLY on the provided context. "
            "If the answer cannot be found in the context, say that you don't know.\n\n"
            "Context:\n{context}"
        ),
        ("human", "{question}")
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever, prompt | llm | StrOutputParser()


def query_rag(question: str, retriever, rag_chain):
    retrieved_docs = retriever.invoke(question)
    if not retrieved_docs:
        return None

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )
    return rag_chain.invoke({
        "context": context,
        "question": question
    })


question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is the principle of least squares?"
)

if st.button("Get Answer", type="primary"):
    if question.strip():
        with st.spinner("Searching PDF and generating answer..."):
            try:
                if uploaded_pdf is not None:
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf",
                        delete=False
                    ) as temp_pdf:
                        temp_pdf.write(uploaded_pdf.getvalue())
                        pdf_path = temp_pdf.name
                else:
                    pdf_path = default_pdf_path

                retriever, rag_chain = create_rag_components(pdf_path)
                answer = query_rag(question, retriever, rag_chain)
                if answer is None:
                    st.warning("No relevant documents found.")
                    st.stop()
                st.subheader("Your Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")