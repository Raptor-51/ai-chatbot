import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from huggingface_hub import InferenceClient

# ================== LOAD API ==================
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")

if not hf_token:
    st.error("❌ HuggingFace API key not found. Set HUGGINGFACE_API_TOKEN in .env")
    st.stop()

# ✅ Client
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
    st.warning("⚠️ No PDFs found in 'data' folder. Please add some PDFs.")
    st.stop()

retriever = db.as_retriever(search_kwargs={"k": 2})

# ================== CHAT MEMORY ==================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
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
            # 🔥 Retrieve relevant docs
            docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in docs])

            # 🔥 Prompt
            prompt = f"""
You are an AI assistant. Answer ONLY from the given context.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{query}
"""

            # ✅ FINAL WORKING CHAT API
            response = client.chat.completions.create(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512
            )

            answer = response.choices[0].message.content

            placeholder.write(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as e:
            placeholder.write(f"❌ Error: {e}")