#!/usr/bin/env python3
"""
Script untuk membandingkan hasil dari semua model yang tersedia.
Sangat berguna untuk testing dan evaluasi kualitas respons dari berbagai model.

Usage:
    python compare_models.py
"""

import requests
import json
import time
from typing import Dict, Any, List
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:5000/api/chat"
TIMEOUT = 60

# Define available models
MODELS = {
    1: {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "type": "original",
        "provider": "HuggingFace"
    },
    2: {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "type": "original",
        "provider": "HuggingFace"
    },
    3: {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "type": "original",
        "provider": "HuggingFace"
    },
    4: {
        "name": "model_merged_legal",
        "type": "fine-tuned",
        "provider": "B200 Server"
    },
    5: {
        "name": "openai/gpt-4.1-mini",
        "type": "openai",
        "provider": "Maia Router"
    }
}

# Test queries
TEST_QUERIES = [
    "Apa itu AI?",
    "Jelaskan tentang machine learning",
    "Siapa yang bertanggung jawab melakukan pengawasan terhadap pengelolaan kegiatan dan keuangan KIBBLA?"
]


class ModelComparator:
    """Class untuk membandingkan hasil dari berbagai model"""
    
    def __init__(self, base_url: str = BASE_URL, timeout: int = TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout
        self.results: List[Dict[str, Any]] = []
    
    def make_request(
        self,
        message: str,
        model_id: int,
        session_id: int = 1,
        user_id: int = 1
    ) -> Dict[str, Any]:
        """
        Make API request ke endpoint chat
        
        Args:
            message: Query/pertanyaan
            model_id: ID model yang akan digunakan
            session_id: Session ID
            user_id: User ID
            
        Returns:
            Dictionary berisi hasil request
        """
        payload = {
            "session_id": session_id,
            "user_id": user_id,
            "message": message,
            "model_id": model_id
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                self.base_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            elapsed_time = time.time() - start_time
            
            result = {
                "status": "success" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "elapsed_time": elapsed_time,
                "timestamp": datetime.now().isoformat()
            }
            
            if response.status_code == 200:
                data = response.json()
                result.update({
                    "model_used": data.get("model_used", "N/A"),
                    "answer": data.get("answer", "N/A"),
                    "sources_count": len(data.get("sources", [])),
                    "response_data": data
                })
            else:
                result.update({
                    "error": response.text[:200],
                    "response_data": response.json() if response.headers.get('content-type') == 'application/json' else None
                })
            
            return result
            
        except requests.exceptions.Timeout:
            return {
                "status": "error",
                "status_code": "TIMEOUT",
                "error": f"Request timeout after {self.timeout}s",
                "elapsed_time": self.timeout,
                "timestamp": datetime.now().isoformat()
            }
        except requests.exceptions.ConnectionError as e:
            return {
                "status": "error",
                "status_code": "CONNECTION_ERROR",
                "error": f"Connection error: {str(e)}",
                "elapsed_time": 0,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "status_code": "EXCEPTION",
                "error": str(e),
                "elapsed_time": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def compare_query(self, query: str, models_to_test: List[int] = None) -> None:
        """
        Jalankan query terhadap semua model dan bandingkan hasil
        
        Args:
            query: Pertanyaan yang akan ditest
            models_to_test: List model ID yang akan ditest (default: semua)
        """
        if models_to_test is None:
            models_to_test = list(MODELS.keys())
        
        print("\n" + "=" * 100)
        print(f"QUERY: {query}")
        print("=" * 100)
        
        results_for_query = {}
        
        for model_id in models_to_test:
            if model_id not in MODELS:
                print(f"❌ Model ID {model_id} tidak ditemukan")
                continue
            
            model_info = MODELS[model_id]
            model_name = model_info["name"]
            
            print(f"\n📝 Testing Model {model_id}: {model_name} ({model_info['provider']})")
            print("-" * 100)
            
            result = self.make_request(query, model_id)
            results_for_query[model_id] = result
            self.results.append({
                "query": query,
                "model_id": model_id,
                **result
            })
            
            if result["status"] == "success":
                print(f"✅ Status: SUCCESS")
                print(f"⏱️  Response Time: {result['elapsed_time']:.2f}s")
                print(f"📚 Sources: {result['sources_count']} document(s)")
                print(f"\n💬 Answer:\n{result['answer'][:500]}...")
            else:
                print(f"❌ Status: ERROR")
                print(f"Error: {result.get('error', 'Unknown error')}")
            
            # Add delay between requests to avoid rate limiting
            time.sleep(1)
        
        # Print comparison summary
        self._print_comparison_summary(query, results_for_query)
    
    def _print_comparison_summary(self, query: str, results: Dict[int, Dict]) -> None:
        """Print summary comparison dari hasil query"""
        print("\n" + "=" * 100)
        print("COMPARISON SUMMARY")
        print("=" * 100)
        
        print(f"\n{'Model ID':<12} {'Model Name':<40} {'Status':<15} {'Time (s)':<12} {'Sources':<10}")
        print("-" * 100)
        
        for model_id, result in results.items():
            model_name = MODELS[model_id]["name"][:38]
            status = "✓ SUCCESS" if result["status"] == "success" else f"✗ {result['status_code']}"
            time_str = f"{result['elapsed_time']:.2f}" if result["status"] == "success" else "-"
            sources = str(result.get('sources_count', 0)) if result["status"] == "success" else "-"
            
            print(f"{model_id:<12} {model_name:<40} {status:<15} {time_str:<12} {sources:<10}")
        
        # Calculate statistics
        successful = sum(1 for r in results.values() if r["status"] == "success")
        times = [r["elapsed_time"] for r in results.values() if r["status"] == "success"]
        
        if times:
            print("\n" + "=" * 100)
            print("STATISTICS")
            print("=" * 100)
            print(f"Total Models Tested: {len(results)}")
            print(f"Successful: {successful}/{len(results)}")
            print(f"Success Rate: {(successful/len(results)*100):.1f}%")
            print(f"\nResponse Time Statistics:")
            print(f"  Average: {sum(times)/len(times):.2f}s")
            print(f"  Fastest: {min(times):.2f}s (Model {list(results.keys())[list(times).index(min(times))]})")
            print(f"  Slowest: {max(times):.2f}s (Model {list(results.keys())[list(times).index(max(times))]})")
    
    def compare_all_queries(self, queries: List[str] = None, models_to_test: List[int] = None) -> None:
        """
        Jalankan semua queries terhadap semua model
        
        Args:
            queries: List queries yang akan ditest
            models_to_test: List model ID yang akan ditest
        """
        if queries is None:
            queries = TEST_QUERIES
        
        print("\n" + "🚀" * 50)
        print("MODEL COMPARISON TOOL")
        print("🚀" * 50)
        print(f"\nTest Configuration:")
        print(f"  API Endpoint: {self.base_url}")
        print(f"  Queries to Test: {len(queries)}")
        print(f"  Models to Test: {len(models_to_test or MODELS)} models")
        print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for query in queries:
            self.compare_query(query, models_to_test)
        
        self._print_final_summary()
    
    def _print_final_summary(self) -> None:
        """Print final summary dari semua tests"""
        print("\n\n" + "=" * 100)
        print("FINAL SUMMARY - ALL TESTS")
        print("=" * 100)
        
        total_requests = len(self.results)
        successful = sum(1 for r in self.results if r["status"] == "success")
        failed = total_requests - successful
        
        print(f"\nTotal Requests: {total_requests}")
        print(f"Successful: {successful} ✓")
        print(f"Failed: {failed} ✗")
        print(f"Success Rate: {(successful/total_requests*100):.1f}%")
        
        # Model performance
        print(f"\n{'Model ID':<12} {'Model Name':<40} {'Success':<12} {'Avg Time':<12}")
        print("-" * 100)
        
        for model_id in MODELS.keys():
            model_results = [r for r in self.results if r["model_id"] == model_id]
            if not model_results:
                continue
            
            model_name = MODELS[model_id]["name"][:38]
            success_count = sum(1 for r in model_results if r["status"] == "success")
            times = [r["elapsed_time"] for r in model_results if r["status"] == "success"]
            avg_time = sum(times) / len(times) if times else 0
            
            success_rate = f"{(success_count/len(model_results)*100):.0f}%"
            avg_time_str = f"{avg_time:.2f}s" if avg_time > 0 else "-"
            
            print(f"{model_id:<12} {model_name:<40} {success_rate:<12} {avg_time_str:<12}")
        
        print(f"\n✅ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)
    
    def export_results(self, filename: str = "comparison_results.json") -> None:
        """Export hasil comparison ke file JSON"""
        output_data = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "total_requests": len(self.results),
                "successful": sum(1 for r in self.results if r["status"] == "success"),
                "failed": sum(1 for r in self.results if r["status"] != "success")
            },
            "results": self.results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results exported to: {filename}")


def main():
    """Main function"""
    comparator = ModelComparator()
    
    # Test dengan semua queries dan semua models
    try:
        comparator.compare_all_queries()
        
        # Export results
        comparator.export_results("comparison_results.json")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
