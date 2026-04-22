# from app.services.rag_service import get_answer_from_rag
# from app.services.rag_service import retrieve_documents_only
# from app.services.evaluation_service import ragas_evaluator
# from src.config import settings
# from typing import Tuple, Dict, Any
# from loguru import logger
# from app.services.rag_service import retrieve_documents_only
# from app.services.rag_service import hf_service


# def ask_perdesai(question: str) -> Tuple[Dict[str, Any], int]:
#     """
#     Jawab pertanyaan user menggunakan RAG + HuggingFace Router API
    
#     Args:
#         question: Pertanyaan dari user
        
#     Returns:
#         Tuple (response_dict, status_code)
#     """
#     try:
#         logger.info(f"Processing question with HuggingFace: {question}")
        
#         # 1. Retrieve dokumen relevan dari Supabase
#         retrieved_docs = retrieve_documents_only(question)
        
#         if not retrieved_docs:
#             logger.warning("No documents retrieved from vector store")
#             return {
#                 "status": "error",
#                 "message": "Tidak ada dokumen yang relevan ditemukan",
#                 "answer": None,
#                 "sources": []
#             }, 404
        
#         # 2. Format context dari dokumen yang diambil
#         context = "\n\n".join([
#             f"[Dokumen {i+1}]\n{doc['content']}\nMetadata: {doc['metadata']}"
#             for i, doc in enumerate(retrieved_docs)
#         ])
        
#         logger.debug(f"Context length: {len(context)} characters")
        
#         # 3. Panggil HuggingFace API dengan context
#         answer = hf_service.chat_with_context(
#             user_question=question,
#             context=context
#         )
        
#         if not answer:
#             logger.error("Failed to get answer from HuggingFace API")
#             return {
#                 "status": "error",
#                 "message": "Gagal mendapatkan jawaban dari LLM",
#                 "answer": None,
#                 "sources": [],
#                 "evaluation": None
#             }, 500
        
#         # 4. Jalankan evaluasi RAGAS (jika diaktifkan)
#         evaluation_results = None
#         if settings.evaluation_enabled:
#             try:
#                 logger.info("🔍 Running RAGAS evaluation...")
#                 contexts = [doc['content'] for doc in retrieved_docs]
                
#                 evaluation_scores = ragas_evaluator.evaluate_single_response_sync(
#                     question=question,
#                     answer=answer,
#                     contexts=contexts
#                 )
                
#                 # Format hasil evaluasi
#                 evaluation_results = ragas_evaluator.format_evaluation_results(evaluation_scores)
#                 logger.success("✅ RAGAS evaluation completed")
                
#             except Exception as eval_error:
#                 logger.warning(f"⚠️ RAGAS evaluation failed: {str(eval_error)}")
#                 evaluation_results = {
#                     "error": f"Evaluation failed: {str(eval_error)}",
#                     "overall_score": 0.0,
#                     "detailed_metrics": {}
#                 }
        
#         # 5. Format response dengan evaluasi
#         return {
#             "status": "success",
#             "message": "Jawaban berhasil diproses" + (" dengan evaluasi RAGAS" if settings.evaluation_enabled else ""),
#             "answer": answer,
#             "sources": [
#                 {
#                     "content": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
#                     "metadata": doc['metadata']
#                 }
#                 for doc in retrieved_docs
#             ],
#             "evaluation": evaluation_results
#         }, 200
        
#     except Exception as e:
#         logger.error(f"Error in ask_with_huggingface: {str(e)}", exc_info=True)
#         return {
#             "status": "error",
#             "message": f"Server error: {str(e)}",
#             "answer": None,
#             "sources": [],
#             "evaluation": None
#         }, 500

# def test_rag_retrieval(query: str):
#     try:
#         if not query or query.strip() == "":
#             return {"status": "error", "message": "Query pencarian tidak boleh kosong."}, 400
            
#         # Panggil service pencarian
#         retrieved_chunks = retrieve_documents_only(query)
        
#         return {
#             "status": "success",
#             "query": query,
#             "total_results": len(retrieved_chunks),
#             "results": retrieved_chunks
#         }, 200
        
#     except Exception as e:
#         return {"status": "error", "message": f"Terjadi kesalahan saat mencari dokumen: {str(e)}"}, 500