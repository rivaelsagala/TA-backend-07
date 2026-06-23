"""
Service untuk evaluasi RAG menggunakan framework RAGAS
"""
import os
import warnings
from typing import Dict, Any, List
from loguru import logger
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    # context_entity_recall,
    NoiseSensitivity
)
from dotenv import load_dotenv

# Import library Langchain untuk custom endpoint (Maiarouter)
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# Import wrapper Ragas untuk standarisasi format
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

# Supaya warning tidak mengganggu
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
       
    6. Noise Sensitivity (0-1): Mengukur apakah model terpengaruh oleh konteks yang tidak relevan (noise)
       - Input: question + answer + contexts + ground_truth
       - Rendah = model tidak mudah dipengaruhi noise, lebih baik
       
    ⚠️ PENTING: Metrik context_recall, context_entity_recall, dan noise_sensitivity memerlukan
    ground_truth berupa jawaban pakar yang valid. Jika reference tidak dikirim, nilai metrik
    tersebut tidak akan akurat.
    """
    
    def __init__(self):
        # Konfigurasi Maiarouter API dari env
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "")
        
        # 1. Inisialisasi LLM Langchain
        # Turunkan temperature ke 0.0 agar juri absolut dan tidak berubah-ubah
        langchain_llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model="openai/gpt-3.5-turbo-16k",
            temperature=0.0,
            max_tokens = 2000
        )
        
        # 2. Inisialisasi Embeddings Langchain
        langchain_embeddings = OpenAIEmbeddings(
            api_key=self.api_key,
            base_url=self.base_url,
            model="openai/text-embedding-3-large"
        )
        
        # 3. WAJIB: Bungkus LLM dan Embeddings dengan Wrapper bawaan RAGAS agar format JSON tidak rusak
        self.custom_llm = LangchainLLMWrapper(langchain_llm)
        self.custom_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
        
        # Inisialisasi NoiseSensitivity
        self.noise_sensitivity = NoiseSensitivity()
        
        # Metrik yang akan digunakan
        self.metrics = [
            faithfulness,           # Apakah jawaban didukung oleh konteks?
            answer_relevancy,       # Apakah jawaban relevan dengan pertanyaan?
            context_precision,      # Apakah konteks relevan dan presisi? (butuh ground_truth)
            context_recall,         # Apakah konteks mencakup info di ground truth? (butuh ground_truth)
            # context_entity_recall,  # Apakah entitas penting dari ground truth ada di konteks? (butuh ground_truth)
            self.noise_sensitivity  # Seberapa sensitif model terhadap noise? (butuh ground_truth)
        ]
        
        logger.info("RAGAS Evaluation Service initialized with wrappers")
    
    def evaluate_single_response(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None
    ) -> Dict[str, Any]:
        """
        Evaluasi satu respons RAG
        
        Args:
            question: Pertanyaan user
            answer: Jawaban dari sistem RAG
            contexts: List konteks yang diambil dari vector database
            ground_truth: Ground truth answer (jawaban pakar/referensi yang valid).
                          WAJIB diisi untuk hasil context_recall, context_entity_recall,
                          dan noise_sensitivity yang akurat. Jika None, metrik-metrik
                          tersebut tidak akan valid karena menggunakan jawaban LLM sebagai acuan.
        
        Returns:
            Dictionary berisi hasil evaluasi semua metrik
        """
        try:
            if ground_truth is None:
                logger.warning(
                    "'ground_truth' tidak diberikan. "
                    "Metrik context_recall, context_entity_recall, dan noise_sensitivity "
                    "tidak akan akurat karena menggunakan jawaban LLM sebagai ground truth. "
                    "Untuk pengujian valid, berikan jawaban pakar sebagai ground_truth."
                )
                ground_truth = answer
            
            # Siapkan dataset untuk evaluasi
            data_sample = {
                "question": [question],
                "contexts": [contexts],
                "answer": [answer],
                "ground_truth": [ground_truth]
            }
            
            eval_dataset = Dataset.from_dict(data_sample)
            
            logger.info(f"🔍 Evaluating response for question: {question[:50]}...")
            
            # Jalankan evaluasi
            evaluation_result = evaluate(
                dataset=eval_dataset,
                metrics=self.metrics,
                llm=self.custom_llm,
                embeddings=self.custom_embeddings
            )
            
            # Konversi hasil ke dictionary
            df_results = evaluation_result.to_pandas()
            result_dict = df_results.iloc[0].to_dict()
            
            # Format hasil
            formatted_result = {
                "faithfulness": float(result_dict.get("faithfulness", 0)),
                "answer_relevancy": float(result_dict.get("answer_relevancy", 0)),
                "context_precision": float(result_dict.get("context_precision", 0)),
                "context_recall": float(result_dict.get("context_recall", 0)),
                # "context_entity_recall": float(result_dict.get("context_entity_recall", 0)),
                "noise_sensitivity": float(result_dict.get("noise_sensitivity", 0)),
            }
            
            # Rata-rata dari seluruh metrik
            # formatted_result["average_score"] = round(
            #     sum(formatted_result.values()) / len(formatted_result), 4
            # )
            
            logger.info(f"Evaluation completed: {formatted_result}")
            return formatted_result
            
        except Exception as e:
            logger.error(f"Error during RAGAS evaluation: {str(e)}")
            return {
                "error": str(e),
                "faithfulness": 0,
                "answer_relevancy": 0,
                "context_precision": 0,
                "context_recall": 0,
                # "context_entity_recall": 0,
                "noise_sensitivity": 0,
                # "average_score": 0
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

# Singleton instance
ragas_service = RagasEvaluationService()
