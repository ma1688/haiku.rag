#!/usr/bin/env python3
"""
Comprehensive retrieval optimization tool for haiku.rag

This script provides:
1. Retrieval performance diagnosis
2. Configuration optimization recommendations
3. Database analysis and optimization
4. Performance testing and benchmarking
"""

import asyncio
import json
import statistics
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console
from rich.progress import Progress
from rich.table import Table

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from haiku.rag.client import HaikuRAG
from haiku.rag.query_processor import query_processor

console = Console()


class RetrievalOptimizer:
    """Comprehensive retrieval optimization tool."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.results = {}
    
    async def diagnose_database(self) -> Dict:
        """Diagnose database and retrieval issues."""
        console.print("🔍 Diagnosing database and retrieval performance...", style="bold blue")
        
        if not self.db_path.exists():
            console.print(f"❌ Database file not found: {self.db_path}", style="red")
            return {}
        
        try:
            async with HaikuRAG(self.db_path) as rag:
                # Get basic database info
                documents = await rag.get_all_documents()
                console.print(f"  📊 Total documents: {len(documents)}")
                
                if len(documents) == 0:
                    console.print("❌ No documents found in database!", style="red")
                    return {"error": "No documents found"}
                
                # Analyze document characteristics
                doc_lengths = []
                chinese_chars = 0
                english_chars = 0
                total_chars = 0
                
                for doc in documents:
                    content = doc.content
                    doc_lengths.append(len(content))
                    total_chars += len(content)
                    
                    for char in content:
                        if '\u4e00' <= char <= '\u9fff':
                            chinese_chars += 1
                        elif char.isalpha() and ord(char) < 128:
                            english_chars += 1
                
                # Get chunk statistics
                cursor = rag.store._connection.cursor()
                cursor.execute("SELECT COUNT(*) FROM chunks")
                chunk_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM chunk_embeddings")
                embedding_count = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT 
                        AVG(LENGTH(content)) as avg_length,
                        MIN(LENGTH(content)) as min_length,
                        MAX(LENGTH(content)) as max_length
                    FROM chunks
                """)
                chunk_stats = cursor.fetchone()
                
                analysis = {
                    "total_documents": len(documents),
                    "total_chunks": chunk_count,
                    "total_embeddings": embedding_count,
                    "avg_doc_length": statistics.mean(doc_lengths) if doc_lengths else 0,
                    "avg_chunk_length": chunk_stats[0] if chunk_stats[0] else 0,
                    "min_chunk_length": chunk_stats[1] if chunk_stats[1] else 0,
                    "max_chunk_length": chunk_stats[2] if chunk_stats[2] else 0,
                    "chinese_ratio": chinese_chars / total_chars if total_chars > 0 else 0,
                    "english_ratio": english_chars / total_chars if total_chars > 0 else 0,
                    "embeddings_complete": chunk_count == embedding_count
                }
                
                self.results["diagnosis"] = analysis
                return analysis
                
        except Exception as e:
            console.print(f"❌ Diagnosis failed: {e}", style="red")
            return {"error": str(e)}
    
    async def test_retrieval_performance(self) -> Dict:
        """Test retrieval performance with various queries."""
        console.print("⚡ Testing retrieval performance...", style="bold blue")
        
        test_queries = [
            "08096的年度股东大会",
            "股东周年大会投票结果",
            "賞之味控股",
            "TASTY CONCEPTS",
            "财务报表",
            "董事会决议",
            "年度报告",
            "AGM",
            "annual meeting",
            "financial report"
        ]
        
        try:
            async with HaikuRAG(self.db_path) as rag:
                results = {
                    "vector_search": [],
                    "fts_search": [],
                    "hybrid_search": []
                }
                
                successful_queries = 0
                total_queries = len(test_queries)
                
                with Progress() as progress:
                    task = progress.add_task("Testing queries...", total=total_queries)
                    
                    for query in test_queries:
                        query_results = {"query": query}
                        
                        try:
                            # Test vector search
                            vector_results = await rag.chunk_repository.search_chunks(query, limit=5)
                            query_results["vector_results"] = len(vector_results)
                            query_results["vector_max_score"] = max([score for _, score in vector_results]) if vector_results else 0
                            
                            # Test FTS search
                            fts_results = await rag.chunk_repository.search_chunks_fts(query, limit=5)
                            query_results["fts_results"] = len(fts_results)
                            query_results["fts_max_score"] = max([score for _, score in fts_results]) if fts_results else 0
                            
                            # Test hybrid search
                            hybrid_results = await rag.search(query, limit=5)
                            query_results["hybrid_results"] = len(hybrid_results)
                            query_results["hybrid_max_score"] = max([score for _, score in hybrid_results]) if hybrid_results else 0
                            
                            if hybrid_results:
                                successful_queries += 1
                            
                            results["hybrid_search"].append(query_results)
                            
                        except Exception as e:
                            console.print(f"  ❌ Query '{query}' failed: {e}")
                            query_results["error"] = str(e)
                        
                        progress.advance(task)
                
                success_rate = successful_queries / total_queries * 100
                results["success_rate"] = success_rate
                results["successful_queries"] = successful_queries
                results["total_queries"] = total_queries
                
                self.results["performance"] = results
                return results
                
        except Exception as e:
            console.print(f"❌ Performance testing failed: {e}", style="red")
            return {"error": str(e)}
    
    def generate_optimization_config(self, analysis: Dict) -> str:
        """Generate optimized configuration based on analysis."""
        console.print("⚙️ Generating optimization configuration...", style="bold blue")
        
        config_lines = [
            "# Optimized configuration for haiku.rag",
            "# Generated based on your data characteristics",
            "",
        ]
        
        # Determine optimal chunk size based on content
        if analysis.get("chinese_ratio", 0) > 0.5:
            # Chinese content optimization
            config_lines.extend([
                "# Chinese content optimization",
                "CHUNK_SIZE=2048",
                "CHUNK_OVERLAP=512",
                "",
                "# Chinese-optimized embedding model",
                "EMBEDDINGS_PROVIDER=siliconflow",
                "EMBEDDINGS_MODEL=Qwen/Qwen3-Embedding-8B",
                "EMBEDDINGS_VECTOR_DIM=4096",
                "",
            ])
        else:
            # English content optimization
            config_lines.extend([
                "# English content optimization",
                "CHUNK_SIZE=1024",
                "CHUNK_OVERLAP=256",
                "",
                "# High-quality embedding model",
                "EMBEDDINGS_PROVIDER=openai",
                "EMBEDDINGS_MODEL=text-embedding-3-large",
                "EMBEDDINGS_VECTOR_DIM=3072",
                "",
            ])
        
        # Add general optimizations
        config_lines.extend([
            "# QA settings",
            "QA_PROVIDER=ollama",
            "QA_MODEL=qwen3",
            "",
            "# API keys (fill in as needed)",
            "SILICONFLOW_API_KEY=your-siliconflow-api-key",
            "OPENAI_API_KEY=your-openai-api-key",
            "",
        ])
        
        return "\n".join(config_lines)
    
    def print_diagnosis_results(self, analysis: Dict):
        """Print diagnosis results in a formatted table."""
        if "error" in analysis:
            console.print(f"❌ Diagnosis error: {analysis['error']}", style="red")
            return
        
        table = Table(title="Database Diagnosis Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Status", style="green")
        
        # Add rows with status indicators
        table.add_row("Total Documents", str(analysis["total_documents"]), 
                     "✅ Good" if analysis["total_documents"] > 0 else "❌ Empty")
        
        table.add_row("Total Chunks", str(analysis["total_chunks"]),
                     "✅ Good" if analysis["total_chunks"] > 0 else "❌ No chunks")
        
        table.add_row("Embeddings Complete", str(analysis["embeddings_complete"]),
                     "✅ Complete" if analysis["embeddings_complete"] else "❌ Incomplete")
        
        table.add_row("Avg Chunk Length", f"{analysis['avg_chunk_length']:.0f} chars",
                     "✅ Good" if analysis['avg_chunk_length'] > 500 else "⚠️ Small chunks")
        
        table.add_row("Chinese Content", f"{analysis['chinese_ratio']*100:.1f}%",
                     "🇨🇳 Chinese-optimized" if analysis['chinese_ratio'] > 0.5 else "🇺🇸 English-optimized")
        
        console.print(table)
    
    def print_performance_results(self, results: Dict):
        """Print performance test results."""
        if "error" in results:
            console.print(f"❌ Performance test error: {results['error']}", style="red")
            return
        
        console.print(f"\n📊 Performance Test Results:")
        console.print(f"  Success Rate: {results['success_rate']:.1f}%")
        console.print(f"  Successful Queries: {results['successful_queries']}/{results['total_queries']}")
        
        if results['success_rate'] >= 80:
            console.print("  🎉 Excellent retrieval performance!", style="green")
        elif results['success_rate'] >= 60:
            console.print("  ✅ Good retrieval performance", style="green")
        elif results['success_rate'] >= 40:
            console.print("  ⚠️ Moderate performance, room for improvement", style="yellow")
        else:
            console.print("  ❌ Poor performance, optimization needed", style="red")
    
    async def run_full_optimization(self):
        """Run complete optimization analysis."""
        console.print("🚀 Starting comprehensive retrieval optimization...", style="bold green")
        console.print("=" * 60)
        
        # 1. Diagnose database
        analysis = await self.diagnose_database()
        if analysis:
            self.print_diagnosis_results(analysis)
        
        # 2. Test performance
        performance = await self.test_retrieval_performance()
        if performance:
            self.print_performance_results(performance)
        
        # 3. Generate optimized config
        if analysis and "error" not in analysis:
            config = self.generate_optimization_config(analysis)
            
            # Save optimized config
            config_path = self.db_path.parent / ".env.optimized"
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config)
            
            console.print(f"\n✅ Optimized configuration saved to: {config_path}", style="green")
        
        # 4. Provide recommendations
        console.print("\n🎯 Optimization Recommendations:", style="bold yellow")
        
        if analysis and not analysis.get("embeddings_complete", True):
            console.print("1. 🔄 Rebuild database: haiku-rag rebuild")
        
        if analysis and analysis.get("avg_chunk_length", 0) < 500:
            console.print("2. 📏 Increase chunk size for better context")
        
        if performance and performance.get("success_rate", 0) < 60:
            console.print("3. 🧠 Consider using a better embedding model")
            console.print("4. 📝 Check if documents contain the information you're searching for")
        
        console.print("\n📋 Next Steps:")
        console.print("1. Apply optimized configuration: cp .env.optimized .env")
        console.print("2. Rebuild database if needed: haiku-rag rebuild")
        console.print("3. Test your specific queries")
        
        # Save detailed results
        results_file = self.db_path.parent / "optimization_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        console.print(f"💾 Detailed results saved to: {results_file}", style="green")


async def main():
    """Main function."""
    if len(sys.argv) < 2:
        console.print("Usage: python retrieval_optimizer.py <database_path>", style="red")
        console.print("Example: python retrieval_optimizer.py ./data/database.db")
        return
    
    db_path = sys.argv[1]
    
    optimizer = RetrievalOptimizer(db_path)
    await optimizer.run_full_optimization()


if __name__ == "__main__":
    asyncio.run(main())
