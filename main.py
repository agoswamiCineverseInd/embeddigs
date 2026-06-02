from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = SentenceTransformer(
    "paraphrase-MiniLM-L3-v2",
    backend="onnx"
)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/embed")
def embed(text: str):
    return {"embedding": model.encode(text).tolist()}