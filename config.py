from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    pinecone_api_key: str
    pinecone_index_name: str = "rag-index"
    pinecone_region: str = "us-east-1"
    pinecone_embed_model: str = "multilingual-e5-large"
    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
