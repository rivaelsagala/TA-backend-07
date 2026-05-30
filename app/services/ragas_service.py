"""
Service untuk evaluasi RAG menggunakan framework RAGAS
"""
import warnings
from typing import Dict, Any, List
from loguru import logger
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Import library Langchain untuk custom endpoint (Maiarouter)
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# Import konfigurasi
from src.config import settings

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
    """
    
    def __init__(self):
        # Konfigurasi Maiarouter API dari settings
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url
        
        # Inisialisasi LLM untuk RAGAS
        self.custom_llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=settings.ragas_model,
            temperature=0.1
        )
        
        # Inisialisasi Embeddings untuk RAGAS
        self.custom_embeddings = OpenAIEmbeddings(
            api_key=self.api_key,
            base_url=self.base_url,
            model=settings.embedding_model
        )
        
        # Metrik yang akan digunakan
        self.metrics = [
            faithfulness,       # Apakah jawaban didukung oleh konteks
            answer_relevancy,   # Apakah jawaban relevan dengan pertanyaan
            context_precision   # Apakah konteks yang diambil relevan dan presisi
        ]
        
        logger.info("RAGAS Evaluation Service initialized")
    
    def evaluate_single_response(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        reference: str = None
    ) -> Dict[str, Any]:
        """
        Evaluasi satu respons RAG
        
        Args:
            question: Pertanyaan user
            answer: Jawaban dari sistem RAG
            contexts: List konteks yang diambil dari vector database
            reference: Ground truth answer (opsional, untuk context_precision tidak wajib)
        
        Returns:
            Dictionary berisi hasil evaluasi (faithfulness, answer_relevancy, context_precision)
        """
        try:
            # Untuk metrik yang digunakan sekarang, reference tidak diperlukan
            # Tapi tetap disimpan untuk kompatibilitas
            if reference is None:
                reference = answer
            
            # Siapkan dataset untuk evaluasi
            data_sample = {
                "question": [question],
                "contexts": [contexts],
                "answer": [answer],
                "ground_truth": [reference]  # RAGAS menggunakan 'ground_truth' bukan 'reference'
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
                "average_score": float(
                    (result_dict.get("faithfulness", 0) + 
                    result_dict.get("answer_relevancy", 0) + 
                    result_dict.get("context_precision", 0)) / 3
                )
            }
            
            logger.info(f"Evaluation completed: {formatted_result}")
            return formatted_result
            
        except Exception as e:
            logger.error(f"Error during RAGAS evaluation: {str(e)}")
            return {
                "error": str(e),
                "faithfulness": 0,
                "answer_relevancy": 0,
                "context_precision": 0,
                "average_score": 0
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
