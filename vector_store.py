import time
import uuid

from pinecone import Pinecone, ServerlessSpec

from config import settings

_pc: Pinecone = None
_index = None

EMBED_BATCH_SIZE = 96  # Pinecone Inference max batch size for multilingual-e5-large


def _get_index():
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
        existing = [i.name for i in _pc.list_indexes().indexes]
        if settings.pinecone_index_name not in existing:
            _pc.create_index(
                name=settings.pinecone_index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=settings.pinecone_region),
            )
            # Wait until index is ready before using it
            while not _pc.describe_index(settings.pinecone_index_name).status.ready:
                time.sleep(2)
        _index = _pc.Index(settings.pinecone_index_name)
    return _pc, _index


def _embed(pc: Pinecone, texts: list[str], input_type: str) -> list[list[float]]:
    all_vectors = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        result = pc.inference.embed(
            model=settings.pinecone_embed_model,
            inputs=batch,
            parameters={"input_type": input_type},
        )
        all_vectors.extend([r["values"] for r in result])
    return all_vectors


def upsert_chunks(chunks: list[str], source: str) -> None:
    pc, index = _get_index()
    embeddings = _embed(pc, chunks, input_type="passage")
    vectors = [
        {
            "id": str(uuid.uuid4()),
            "values": emb,
            "metadata": {"text": chunk, "source": source},
        }
        for chunk, emb in zip(chunks, embeddings)
    ]
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i + 100])


def query_similar(question: str) -> list[dict]:
    pc, index = _get_index()
    embedding = _embed(pc, [question], input_type="query")[0]
    result = index.query(
        vector=embedding,
        top_k=settings.top_k,
        include_metadata=True,
    )
    return [
        {
            "text": m.metadata["text"],
            "source": m.metadata["source"],
            "score": m.score,
        }
        for m in result.matches
    ]
