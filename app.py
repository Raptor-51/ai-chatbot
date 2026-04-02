import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from huggingface_hub import InferenceClient
import traceback

# ================== LOAD API ==================
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")

if not hf_token:
    st.error("❌ HuggingFace API key not found.")
    st.stop()

client = InferenceClient(token=hf_token)

# ================== UI ==================
st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🤖")
st.title("🤖 AI Knowledge Assistant")

# ================== LOAD PDF DATA ==================
@st.cache_resource
def load_data():
    documents = []

    if not os.path.exists("data"):
        os.makedirs("data")

    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"data/{file}")
            documents.extend(loader.load())

    if not documents:
        return None

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(docs, embeddings)
    return db

db = load_data()

if db is None:
    st.warning("⚠️ No PDFs found in 'data' folder.")
    st.stop()

retriever = db.as_retriever(search_kwargs={"k": 2})

# ================== CHAT MEMORY ==================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ================== INPUT ==================
query = st.chat_input("Ask something from your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.write("⏳ Thinking...")

        try:
            # 🔥 Retrieve docs
            docs = retriever.invoke(query)

            if not docs:
                placeholder.write("❌ No relevant data found in documents.")
                st.stop()

            context = "\n".join([doc.page_content for doc in docs])

            # 🔥 Prompt
            prompt = f"""
Answer ONLY from the context below.
If not found, say "I don't know."

Context:
{context}

Question:
{query}
"""

            # 🔥 HF working call
            response = client.text_generation(
                model="google/flan-t5-base",
                prompt=prompt,
                max_new_tokens=256
            )

            answer = response.strip()

            if not answer:
                answer = "⚠️ No response generated."

            placeholder.write(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception:
            error_msg = traceback.format_exc()
            placeholder.write(f"❌ Error:\n{error_msg}")