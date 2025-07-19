#!/usr/bin/env python3
"""
Interactive QA Demo for Haiku RAG

This script demonstrates how to use the interactive QA system with sample documents.
It creates a sample knowledge base and starts an interactive chat session.
"""
import asyncio
import tempfile
from pathlib import Path

from haiku.rag.client import HaikuRAG
from haiku.rag.qa.interactive import start_interactive_qa


async def setup_sample_knowledge_base(db_path: str) -> None:
    """Set up a sample knowledge base with various documents."""
    
    sample_documents = [
        {
            "content": """
# Python Programming Language

Python is a high-level, interpreted programming language with dynamic semantics. 
Its high-level built-in data structures, combined with dynamic typing and dynamic binding, 
make it very attractive for Rapid Application Development, as well as for use as a 
scripting or glue language to connect existing components together.

## Key Features
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Large community support
- Versatile applications (web development, data science, AI, automation)

## Popular Libraries
- NumPy: Numerical computing
- Pandas: Data manipulation and analysis
- Django/Flask: Web frameworks
- TensorFlow/PyTorch: Machine learning
- Requests: HTTP library
            """,
            "uri": "python_guide.md",
            "metadata": {"topic": "programming", "language": "python"}
        },
        {
            "content": """
# Machine Learning Fundamentals

Machine Learning (ML) is a subset of artificial intelligence (AI) that provides 
systems the ability to automatically learn and improve from experience without 
being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
- Uses labeled training data
- Examples: Classification, Regression
- Algorithms: Linear Regression, Decision Trees, Random Forest, SVM

### Unsupervised Learning
- Finds patterns in data without labels
- Examples: Clustering, Dimensionality Reduction
- Algorithms: K-Means, PCA, DBSCAN

### Reinforcement Learning
- Learns through interaction with environment
- Uses rewards and penalties
- Applications: Game playing, Robotics, Autonomous vehicles

## Common Applications
- Image recognition
- Natural language processing
- Recommendation systems
- Fraud detection
- Predictive analytics
            """,
            "uri": "ml_fundamentals.md",
            "metadata": {"topic": "machine-learning", "category": "ai"}
        },
        {
            "content": """
# Data Science Workflow

Data Science is an interdisciplinary field that uses scientific methods, processes, 
algorithms and systems to extract knowledge and insights from structured and 
unstructured data.

## The Data Science Process

1. **Problem Definition**
   - Understand business objectives
   - Define success metrics
   - Identify data requirements

2. **Data Collection**
   - Gather relevant data sources
   - APIs, databases, web scraping
   - Ensure data quality and completeness

3. **Data Exploration & Cleaning**
   - Exploratory Data Analysis (EDA)
   - Handle missing values
   - Remove outliers and inconsistencies
   - Feature engineering

4. **Modeling**
   - Select appropriate algorithms
   - Train and validate models
   - Hyperparameter tuning
   - Cross-validation

5. **Evaluation**
   - Assess model performance
   - Use appropriate metrics
   - Compare different approaches

6. **Deployment**
   - Implement in production
   - Monitor performance
   - Maintain and update models

## Essential Tools
- Python/R for programming
- Jupyter Notebooks for experimentation
- SQL for data querying
- Git for version control
- Cloud platforms (AWS, GCP, Azure)
            """,
            "uri": "data_science_workflow.md",
            "metadata": {"topic": "data-science", "category": "process"}
        }
    ]
    
    print("🔧 Setting up sample knowledge base...")
    
    async with HaikuRAG(db_path) as client:
        for doc_data in sample_documents:
            doc = await client.create_document(
                content=doc_data["content"],
                uri=doc_data["uri"],
                metadata=doc_data["metadata"]
            )
            print(f"   ✅ Added document: {doc_data['uri']} (ID: {doc.id})")
    
    print(f"📚 Knowledge base ready with {len(sample_documents)} documents!")
    print()


async def main():
    """Main demo function."""
    print("🤖 Haiku RAG Interactive QA Demo")
    print("=" * 40)
    print()
    
    # Create a temporary database for the demo
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Setup sample knowledge base
        await setup_sample_knowledge_base(db_path)
        
        # Display instructions
        print("📖 Sample Knowledge Base Contents:")
        print("   • Python Programming Language")
        print("   • Machine Learning Fundamentals") 
        print("   • Data Science Workflow")
        print()
        print("💡 Try asking questions like:")
        print("   • 'What is Python?'")
        print("   • 'Explain machine learning types'")
        print("   • 'What are the steps in data science?'")
        print("   • 'What libraries are popular for ML?'")
        print()
        print("🚀 Starting interactive QA session...")
        print("   Type /help for available commands")
        print("   Type /quit to exit")
        print()
        
        # Start interactive session
        await start_interactive_qa(db_path)
        
    finally:
        # Clean up temporary database
        try:
            Path(db_path).unlink()
            print("🧹 Cleaned up temporary database")
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Error running demo: {e}")
