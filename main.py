from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


class EmbeddingRequest(BaseModel):
    text: str


@app.post("/embedding")
async def embedding(req: EmbeddingRequest):

    vector = model.encode(
        req.text,
        normalize_embeddings=True
    )

    return {
        "embedding": vector.tolist(),
        "dimensions": len(vector)
    }