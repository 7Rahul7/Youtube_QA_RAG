
# 🎥 YouTube Transcript Question Answering Web App

This is a FastAPI-based web application that allows you to:
- Search YouTube videos based on a topic
- Automatically fetch and process the video transcripts
- Ask intelligent questions about the videos
- Get accurate answers using powerful Hugging Face LLMs and FAISS-based semantic search

---

## 🚀 Features

- 🔍 YouTube video search integration
- 🎬 Transcript extraction using YouTube Transcript API
- 🧠 Smart question answering using RAG (Retrieval-Augmented Generation)
- ⚡ Fast and interactive web UI using FastAPI + Jinja2 templates
- 🔎 FAISS for fast, chunk-level semantic search on transcripts

---

## 🧪 Demo Use Case

1. Search term: `Transformers in NLP`
2. Question: `What is the role of attention mechanism?`
3. Output: AI model returns an answer from video transcripts relevant to your query.

---

## 📁 Folder Structure

```bash
YOUTUBE_QA/
├── .env                       # Hugging Face API token
├── .gitignore
├── requirements.txt
├── pyproject.toml             # Optional (for build tools or formatting)
├── README.md
├── main.py                    # (Optional) root-level script
├── app/
│   ├── main.py                # FastAPI app with routes and logic
│   ├── qa_engine.py           # Search, transcript, FAISS, and LLM
│   └── templates/
│       └── index.html         # Jinja2-based HTML UI
````

---

## 🔐 .env Configuration

Create a `.env` file in your project root (`YOUTUBE_QA/.env`) with the following:

```env
HUGGINGFACE_API_TOKEN=your_huggingface_access_token
```

> 🔑 You can get your token from: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
> ⚠️ Never share or push your `.env` to GitHub

---

## 📦 Installation Instructions

### 1. Clone the repository

```bash
git clone https://github.com/7Rahul7/Youtube_QA_RAG.git

```

### 2. Create and activate virtual environment (using `uv`)

```bash
uv venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Run the App

```bash
cd app/
then run command below
    python main.py
```

Then open your browser at: [http://localhost:8000](http://localhost:8000)

---

## 🌍 How It Works

1. You enter a **search query** (e.g., "Machine learning basics")
2. The app finds relevant **YouTube videos**
3. It fetches and splits the **transcripts**
4. FAISS finds the best matching **chunks**
5. An **LLM answers your question** using only those chunks (RAG)

---

## ✅ requirements.txt

```txt
fastapi
uvicorn
jinja2
youtube-search
youtube-transcript-api
langchain==0.1.17
langchain-community==0.0.26
faiss-cpu
transformers
sentence-transformers
torch
python-dotenv
accelerate
huggingface_hub
```

---

## 🧱 Example Models You Can Use

* `HuggingFaceH4/zephyr-7b-beta` ✅ (Open and fast)
* `mistralai/Mistral-7B-Instruct-v0.1` 🔒 (Requires token and access)
* `tiiuae/falcon-7b-instruct` ✅
* or any Hugging Face `text-generation` LLM

---

## 📌 Tips

* Use GPU if possible for faster LLM inference.
* For large-scale use, cache the FAISS index.
* If using private models, make sure you're authenticated via `.env`.

---

## 📌 Roadmap (Optional Features)

* [ ] Upload your own transcripts (PDF, text)
* [ ] Add Gradio or Streamlit UI
* [ ] Use OpenAI/Gemini LLMs as fallback
* [ ] Whisper fallback for videos without transcripts

---

## 🧑‍💻 Author

**Your Name**

🔗 GitHub: [@7Rahul7](https://github.com/7Rahul7)

---
