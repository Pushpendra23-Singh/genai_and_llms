"""
GenAI Summarizer Microservice — FastAPI + Ollama
pip install fastapi uvicorn ollama
Run: uvicorn main:app --reload
Requires: ollama pull mistral
"""
from fastapi import FastAPI
from pydantic import BaseModel
import ollama

app = FastAPI()

class Req(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/summarize")
def summarize(req: Req):
    res = ollama.chat(model="mistral", messages=[
        {"role": "user", "content": f"Summarize in 2 sentences:\n{req.text}"}
    ])
    return {"summary": res["message"]["content"]}

@app.post("/chat")
def chat(req: Req):
    res = ollama.chat(model="mistral", messages=[
        {"role": "user", "content": req.text}
    ])
    return {"reply": res["message"]["content"]}

# Run: uvicorn main:app --reload
# Test: curl -X POST http://localhost:8000/summarize -H "Content-Type: application/json" -d '{"text":"your text here"}'
