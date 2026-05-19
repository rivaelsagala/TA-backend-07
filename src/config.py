"""
src/config.py
--------------
Konfigurasi terpusat menggunakan Pydantic Settings.
Semua nilai dibaca dari file .env secara otomatis.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Seluruh konfigurasi aplikasi RAG Peraturan Desa.
    Nilai dibaca dari .env di root folder.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- HuggingFace (Untuk LLM Chat via Router API) ----
    hf_token: str = Field(default="", alias="HF_TOKEN")
    hf_api_url: str = Field(
        default="https://router.huggingface.co/v1/chat/completions",
        # default="https://router.huggingface.co/v1",
        alias="HF_API_URL"
    )
    hf_llm_model: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        alias="HF_LLM_MODEL"
    )
    
    # ---- Fine-tuned Model (Local API on B200 Server) ----
    finetuned_api_url: str = Field(
        default="http://localhost:6000/api",
        alias="FINETUNED_API_URL"
    )
    finetuned_model_name: str = Field(
        default="model_merged_legal",
        alias="FINETUNED_MODEL_NAME"
    )

    # ---- LLM Model ----
    llm_model: str = Field(default="meta-llama/Llama-3.1-8B-Instruct", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")


    # ---- OpenAI ----
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.maiarouter.ai/v1", alias="OPENAI_BASE_URL")
    
    # # ---- LLM Model (OpenAI GPT-4o) ----
    # llm_model: str = Field(default="openai/gpt-4.1-mini", alias="LLM_MODEL")
    # llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    # llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")

    # ---- RAGAS Model (gpt-3.5-turbo-16k) ----
    ragas_model: str = Field(default="openai/gpt-3.5-turbo-16k", alias="RAGAS_MODEL")


    # ---- Embedding (OpenAI text-embedding-3-small) ----
    embedding_model: str = Field(
        default="openai/text-embedding-3-small",
        alias="EMBEDDING_MODEL",
    )
    embedding_dim: int = Field(default=1536, alias="EMBEDDING_DIM")

    # ---- PostgreSQL Database ----
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_name: str = Field(default="postgres", alias="DB_NAME")
    db_user: str = Field(default="postgres", alias="DB_USER")
    db_password: str = Field(default="", alias="DB_PASSWORD")

    # ---- Supabase ----
    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    supabase_key: str = Field(default="", alias="SUPABASE_KEY")
    supabase_table_name: str = Field(default="documents", alias="SUPABASE_TABLE_NAME")

    # ---- Chunking ----
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    # ---- Retriever ----
    top_k_results: int = Field(default=2, alias="TOP_K_RESULTS")


    # ---- API / Flask ----
    # api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    # api_port: int = Field(default=8000, alias="API_PORT")
    # api_reload: bool = Field(default=True, alias="API_RELOAD")


# Singleton instance — digunakan di seluruh aplikasi
settings = Settings()
