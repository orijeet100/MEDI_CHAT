

# 🏥 MED RAG: Medical Retrieval-Augmented Generation Chatbot

A powerful, production-ready Streamlit app for medical question answering using Retrieval-Augmented Generation (RAG). MED RAG combines a local FAISS vector database, state-of-the-art embedding models, and a large language model (LLM) to provide accurate, context-aware answers to medical queries based on your own document collection.

---


**👉 Try it on Hugging Face Spaces, and use high CPU or  GPU for faster results: [https://huggingface.co/spaces/orijeetmukherjee/MED_RAG](https://huggingface.co/spaces/orijeetmukherjee/MED_RAG)**


## 🚀 Features

- **Retrieval-Augmented Generation (RAG):** Combines semantic search with LLMs for grounded, up-to-date answers.
- **Medical Document Database:** Uses FAISS for fast vector search over your own medical documents.
- **Custom Embeddings:** Utilizes `sentence-transformers/all-MiniLM-L6-v2` for efficient, high-quality embeddings.
- **LLM Integration:** Out-of-the-box support for models like `meta-llama/Llama-3.2-1B-Instruct` (customizable).
- **Source Attribution:** Optionally shows which document chunks were used to answer each question.
- **Streamlit UI:** Clean, interactive web interface with chat history, settings, and status indicators.
- **Dockerized:** Easy to deploy anywhere with Docker.
- **Extensible:** Easily swap in your own models or document sources.

---

## 🖥️ Quickstart

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
```

### 2. Prepare the Database
Place your FAISS index and metadata files as `medical_vector_db_faiss.index` and `medical_vector_db_data.pkl` in the `src/` directory. (See code/docs for how to build these from your own documents.)

### 3. Run Locally
```bash
streamlit run src/streamlit_app.py
```

### 4. Or Run with Docker
```bash
docker build -t medrag .
docker run -p 8501:8501 medrag
```

---

## 🧑‍⚕️ Usage
- Open your browser to [http://localhost:8501](http://localhost:8501)
- Use the sidebar to load the database and models
- Ask medical questions in the chat box
- View sources for each answer (if enabled)

---

## ⚙️ Settings & Customization
- **Models:** Change embedding or LLM models by editing `src/streamlit_app.py`
- **Database:** Replace the FAISS index and metadata with your own
- **UI:** Customize Streamlit components as desired

---

## 📝 Requirements
- Python 3.9+
- See `requirements.txt` for all dependencies:
  - streamlit, sentence-transformers, faiss-cpu, transformers, torch, huggingface-hub, numpy, pandas, PyPDF2, python-dotenv, scikit-learn, accelerate, tqdm, altair

---

## 🤗 Hugging Face Space
Deploy this app on Hugging Face Spaces for free cloud hosting:


---

## 📄 License
Specify your license here (e.g., MIT, Apache-2.0, etc.)

---

## 🙏 Citation
If you use this project in your research or products, please consider citing or linking back to this repository.

