"""
Service untuk evaluasi RAG menggunakan framework RAGAS
"""
import os
import warnings
from typing import Dict, Any, List, Optional
from loguru import logger
from datasets import Dataset
import numpy as np
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from dotenv import load_dotenv

from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

warnings.filterwarnings("ignore", category=DeprecationWarning)


class SemanticAnswerSimilarity:

    def __init__(self, model_name: Optional[str] = None):
        self.api_key  = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "")
        self.model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-large")
        )

        self._embedder = OpenAIEmbeddings(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model_name,
            default_headers={"User-Agent": "curl/7.68.0"}
        )
        logger.info(
            f"SemanticAnswerSimilarity initialized — model: {self.model_name}"
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Hitung cosine similarity antara dua vektor, kembalikan float 0–1."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        # cosine similarity bisa -1..1; clip ke 0..1 agar konsisten
        raw = float(np.dot(a, b) / (norm_a * norm_b))
        return float(np.clip(raw, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_sas(
        self,
        answer: str,
        ground_truth: str,
    ) -> float:
        """
        Hitung SAS untuk satu pasang (answer, ground_truth).

        Returns:
            float: Similarity score 0.0 – 1.0
        """
        try:
            vecs = self._embedder.embed_documents([answer, ground_truth])
            score = self._cosine_similarity(vecs[0], vecs[1])
            logger.debug(f"SAS score: {score:.4f}")
            return round(score, 4)
        except Exception as e:
            logger.error(f"Error computing SAS: {e}")
            return 0.0

    def compute_sas_batch(
        self,
        pairs: List[Dict[str, str]],
    ) -> List[float]:
        """
        Hitung SAS untuk banyak pasang sekaligus (batch) secara efisien.
        Menggunakan satu API call untuk semua teks.

        Args:
            pairs: List of dict dengan key 'answer' dan 'ground_truth'.
                   Contoh: [{"answer": "...", "ground_truth": "..."}, ...]

        Returns:
            List[float]: SAS score 0.0 – 1.0 untuk setiap pasang,
                         dengan urutan yang sama seperti input.
        """
        if not pairs:
            return []
        try:
            texts = []
            for p in pairs:
                texts.append(p["answer"])
                texts.append(p["ground_truth"])

            all_vecs = self._embedder.embed_documents(texts)

            scores = []
            for i in range(0, len(all_vecs), 2):
                score = self._cosine_similarity(all_vecs[i], all_vecs[i + 1])
                scores.append(round(score, 4))

            logger.debug(f"SAS batch ({len(pairs)} pairs): {scores}")
            return scores
        except Exception as e:
            logger.error(f"Error computing SAS batch: {e}")
            return [0.0] * len(pairs)

class RagasEvaluationService:
    """
    Service untuk mengevaluasi respons RAG menggunakan RAGAS metrics
    
    Metrik yang digunakan:
    1. Faithfulness (0-1): Mengukur apakah jawaban didukung oleh konteks yang diambil
       - Input: answer + contexts
       - Tinggi = jawaban tidak mengarang, semua klaim ada di konteks
       
    2. Answer Relevancy (0-1): Mengukur apakah jawaban relevan dengan pertanyaan
       - Input: question + answer  
       - Tinggi = jawaban menjawab pertanyaan dengan tepat
       
    3. Context Precision (0-1): Mengukur apakah konteks yang diambil relevan dan presisi
       - Input: question + contexts + ground_truth
       - Tinggi = konteks yang diambil tepat sasaran, tidak banyak noise
       - Konteks relevan muncul di ranking atas
       
    4. Context Recall (0-1): Mengukur seberapa banyak informasi dari ground truth tercakup dalam konteks
       - Input: contexts + ground_truth (WAJIB diisi dengan jawaban pakar, bukan jawaban LLM)
       - Tinggi = konteks berhasil mencakup semua informasi penting dari referensi
       
    5. Context Entity Recall (0-1): Mengukur seberapa banyak entitas penting dari ground truth ada di konteks
       - Input: contexts + ground_truth (WAJIB diisi dengan jawaban pakar)
       - Tinggi = entitas kunci (nama, pasal, dll) dari referensi ditemukan di konteks
       
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "")
        
        langchain_llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model="openai/gpt-4o-mini",  
            temperature=0.0,
            max_tokens=2000,
            default_headers={"User-Agent": "curl/7.68.0"}
        )
        
        langchain_embeddings = OpenAIEmbeddings(
            api_key=self.api_key,
            base_url=self.base_url,
            model="openai/text-embedding-3-large",
            default_headers={"User-Agent": "curl/7.68.0"}
        )
        
        self.custom_llm = LangchainLLMWrapper(langchain_llm)
        self.custom_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
        
        # Metrik yang akan digunakan
        self.metrics = [
            faithfulness,        
            answer_relevancy,   
            context_precision,   
            context_recall,     
        ]
        
        logger.info("RAGAS Evaluation Service initialized with wrappers")
    
    def evaluate_single_response(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None
    ) -> Dict[str, Any]:

        try:
            if ground_truth is None:
                logger.warning(
                    "'ground_truth' tidak diberikan. "
                    "Metrik context_precision, context_recall, dan noise_sensitivity "
                    "akan dilewati karena membutuhkan referensi jawaban pakar."
                )
                metrics_to_run = [faithfulness, answer_relevancy]
                data_sample = {
                    "question": [question],
                    "contexts": [contexts],
                    "answer": [answer]
                }
            else:
                metrics_to_run = self.metrics
                data_sample = {
                    "question": [question],
                    "contexts": [contexts],
                    "answer": [answer],
                    "ground_truth": [ground_truth]
                }
            
            eval_dataset = Dataset.from_dict(data_sample)
            
            logger.info(f"🔍 Evaluating response for question: {question[:50]}...")
            
            evaluation_result = evaluate(
                dataset=eval_dataset,
                metrics=metrics_to_run,
                llm=self.custom_llm,
                embeddings=self.custom_embeddings
            )
            
            df_results = evaluation_result.to_pandas()
            result_dict = df_results.iloc[0].to_dict()
            
            def get_metric_val(key):
                val = result_dict.get(key)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return None
                return float(val)

            formatted_result = {
                "faithfulness":      get_metric_val("faithfulness"),
                "answer_relevancy":  get_metric_val("answer_relevancy"),
                "context_precision": get_metric_val("context_precision"),
                "context_recall":    get_metric_val("context_recall"),
            }

            if ground_truth is not None:
                sas_score = sas_service.compute_sas(answer, ground_truth)
                formatted_result["semantic_similarity"] = sas_score
                logger.debug(f"SAS (answer vs ground_truth): {sas_score}")
            else:
                formatted_result["semantic_similarity"] = None
                logger.debug("SAS skipped because ground_truth is not provided.")
            

            
            logger.info(f"Evaluation completed: {formatted_result}")
            return formatted_result
            
        except Exception as e:
            logger.error(f"Error during RAGAS evaluation: {str(e)}")
            return {
                "error": str(e),
                "faithfulness":      0,
                "answer_relevancy":  0,
                "context_precision": 0,
                "context_recall":    0,
                "semantic_similarity": 0,
            }
    
    def format_contexts_from_sources(self, sources: List[Dict[str, Any]]) -> List[str]:
        """
        Format sources dari RAG menjadi list of strings untuk RAGAS
        
        Args:
            sources: List of dict dengan key 'content' dan 'metadata'
        
        Returns:
            List of context strings
        """
        contexts = []
        for source in sources:
            if isinstance(source, dict) and "content" in source:
                contexts.append(source["content"])
            elif isinstance(source, str):
                contexts.append(source)
        
        return contexts

sas_service = SemanticAnswerSimilarity()
ragas_service = RagasEvaluationService()
