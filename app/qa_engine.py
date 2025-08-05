import torch
from youtube_search import YoutubeSearch
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load .env file
load_dotenv()

# Read token and login
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
if hf_token:
    login(hf_token)

def search_youtube(query, max_results=3):
    results = YoutubeSearch(query, max_results=max_results).to_dict()
    videos = [
        {
            "title": video["title"],
            "url": f"https://www.youtube.com{video['url_suffix']}",
            "id": video["url_suffix"].split("v=")[-1]
        }
        for video in results
    ]
    return videos

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([item['text'] for item in transcript])
    except:
        return None

def build_faiss_index(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.create_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(split_docs, embeddings)

def load_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)

def run_qa_pipeline(search_term, question, llm=None):
    videos = search_youtube(search_term)
    transcripts = []
    for video in videos:
        transcript = get_transcript(video["id"])
        if transcript:
            transcripts.append(transcript)

    if not transcripts:
        return {"answer": "No transcript found", "videos": videos}

    vector_db = build_faiss_index(transcripts)
    llm = llm or load_llm()
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = qa_chain.run(question)

    return {
        "answer": answer,
        "videos": videos
    }
