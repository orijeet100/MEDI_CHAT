import os
import pickle
from datetime import datetime

# ─── make a writable cache directory inside the container ───
os.environ["HF_HOME"]              = "/tmp/hf_home"               # master cache
os.environ["TRANSFORMERS_CACHE"]   = "/tmp/hf_home/transformers"  # AutoModel/Tokenizer
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/hf_home/st"      # Sentence-Transformers

import faiss
import numpy as np
import streamlit as st
import torch
from dotenv import load_dotenv
from huggingface_hub import HfFolder
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, StoppingCriteriaList, StoppingCriteria

# torch.classes.__path__ = []

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        super().__init__()
        self.stop_ids = stop_ids
    def __call__(self, input_ids, scores, **kwargs):
        return any(input_ids[0, -1] == i for i in self.stop_ids)



# ─── Caching Helpers ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=False)
def load_llm_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    # Ensure we have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id

    model.to("cpu")
        
    return tokenizer, model

# ─── Medical RAG Chatbot Class ────────────────────────────────────────────────
class MedicalRAGChatbot:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    ):
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.embedding_model = None
        self.tokenizer = None
        self.llm_model = None
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
    
    def load_models(self):
        """Load embedding and LLM/tokenizer (cached)."""
        with st.spinner("🔄 Loading embedding model..."):
            self.embedding_model = load_embedding_model(self.embedding_model_name)
        st.success("✅ Embedding model loaded!")

        with st.spinner("🔄 Loading LLM and tokenizer..."):
            self.tokenizer, self.llm_model = load_llm_and_tokenizer(self.llm_model_name)
        st.success("✅ LLM and tokenizer loaded!")

    def load_database(self, load_path: str = "medical_vector_db"):
        """Load FAISS index and metadata from working directory."""
        faiss_path = f"/app/src/{load_path}_faiss.index"
        data_path = f"/app/src/{load_path}_data.pkl"
        if not os.path.exists(faiss_path) or not os.path.exists(data_path):
            st.error(f"Database files not found:\n• {faiss_path}\n• {data_path}")
            return False

        # Load FAISS index
        self.index = faiss.read_index(faiss_path)
        # Load chunks & metadata
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.chunk_metadata = data["metadata"]
        # ensure embedding model is loaded
        if self.embedding_model is None:
            self.embedding_model = load_embedding_model(self.embedding_model_name)

        # Display info
        idx_size = os.path.getsize(faiss_path) / (1024**2)
        data_size = os.path.getsize(data_path) / (1024**2)
        sources = len({m["source_file"] for m in self.chunk_metadata})
        st.success("✅ Database loaded!")
        st.info(f"• {len(self.chunks)} chunks from {sources} files\n"
                f"• FAISS index: {idx_size:.2f} MB\n"
                f"• Metadata: {data_size:.2f} MB")
        return True

    def query_database(self, query: str, top_k: int = 3):
        """Embed query, search FAISS, and return (chunk, score, metadata) list."""
        emb = self.embedding_model.encode(query)
        D, I = self.index.search(np.array([emb]), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append((self.chunks[idx], float(score), self.chunk_metadata[idx]))
        return results

    def create_medical_prompt(self, query: str, chunks: list[str]):
        """Combine retrieved chunks into a single prompt for the LLM."""
        context = "\n\n".join(chunks)
        return (
            "You are a helpful medical assistant. Use the following context to answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )


    def generate_response(self, prompt, max_length=200):
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, padding=True,
                                max_length=1024)   # hard cap prompt
        stop = StoppingCriteriaList([StopOnTokens([self.tokenizer.eos_token_id])])
    
        with torch.no_grad():
            out = self.llm_model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.5,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,   # 🛑 add
                stopping_criteria=stop,                     # 🛑 add
            )
    
        answer = self.tokenizer.decode(
            out[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        return answer or "I’m sorry, I couldn’t generate a response."


    def chat(self, query: str, top_k: int = 3, show_sources: bool = True):
        """Retrieve + generate pipeline."""
        results = self.query_database(query, top_k)
        if not results:
            return "No info found in database.", []
        chunks = [r[0] for r in results]

        print(chunks)
        
        prompt = self.create_medical_prompt(query, chunks)

        print(prompt)
        answer = self.generate_response(prompt)
        print(answer)
        return answer, results

# ─── Streamlit App ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Medical RAG Chatbot 🏥", layout="wide")
    st.title("🏥 Medical RAG Chatbot")
    st.markdown("Ask medical questions based on your document database.")

    if 'bot' not in st.session_state:
        st.session_state.bot = MedicalRAGChatbot()
        st.session_state.loaded_db = False
        st.session_state.loaded_models = False
        st.session_state.history = []

    # Sidebar controls
    with st.sidebar:
        st.header("🔧 Setup")
        if st.button("Load Database"):
            st.session_state.loaded_db = st.session_state.bot.load_database()
        if st.button("Load Models"):
            if st.session_state.loaded_db:
                st.session_state.bot.load_models()
                st.session_state.loaded_models = True
            else:
                st.error("Load database first!")
        st.markdown("---")
        st.subheader("⚙️ Settings")
        top_k = st.slider("Sources to retrieve", 1, 2, 1)
        show_sources = st.checkbox("Show sources", True)
        max_len = st.slider("Max response length", 200, 201, 200)
        st.markdown("---")
        st.subheader("📊 Status")
        st.success("DB Loaded" if st.session_state.loaded_db else "DB Not Loaded")
        st.success("Models Loaded" if st.session_state.loaded_models else "Models Not Loaded")

    # Main chat
    if st.session_state.loaded_db and st.session_state.loaded_models:
        query = st.text_input("💬 Your question:", key="input")
        if st.button("Ask"):
            with st.spinner("🔍 Retrieving and generating..."):
                answer, sources = st.session_state.bot.chat(query, top_k, show_sources)
                st.session_state.history.append((query, answer, sources))
                st.rerun()

        # Display history
        for q, a, srcs in reversed(st.session_state.history):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
            if show_sources:
                with st.expander(f"Sources ({len(srcs)} chunks)"):
                    for i, (chunk, score, md) in enumerate(srcs, 1):
                        st.markdown(
                            f"> **{i}.** {md['source_file']} (sim={score:.4f}):\n> {chunk[:200]}..."
                        )
            st.markdown("---")
    else:
        st.info("Please load the database and models via the sidebar.")

if __name__ == "__main__":
    main()
