from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from qa_engine import run_qa_pipeline, load_llm 

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
llm_model = load_llm()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request, search_term: str = Form(...), question: str = Form(...)):
    result = run_qa_pipeline(search_term, question, llm_model)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "answer": result["answer"],
        "videos": result["videos"],
        "search_term": search_term,
        "question": question
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
