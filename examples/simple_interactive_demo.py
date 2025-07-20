#!/usr/bin/env python3
"""
Simple Interactive QA Demo (without heavy dependencies)

This demonstrates the core conversation functionality of the interactive QA system
without requiring all the heavy dependencies like markitdown, sqlite-vec, etc.
"""
import asyncio
from datetime import datetime
from typing import List, Tuple


class MockChunk:
    """Mock chunk for demonstration."""
    def __init__(self, content: str, document_uri: str = None):
        self.content = content
        self.document_uri = document_uri


class MockClient:
    """Mock RAG client for demonstration."""
    
    def __init__(self):
        # Sample knowledge base
        self.knowledge = [
            MockChunk("Python is a high-level, interpreted programming language known for its simplicity and readability.", "python_intro.md"),
            MockChunk("Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.", "python_features.md"),
            MockChunk("Popular Python libraries include NumPy for numerical computing, Pandas for data analysis, and Django for web development.", "python_libraries.md"),
            MockChunk("Machine learning is a subset of AI that enables computers to learn without explicit programming.", "ml_basics.md"),
            MockChunk("Supervised learning uses labeled data, unsupervised learning finds patterns in unlabeled data.", "ml_types.md"),
        ]
    
    async def search(self, query: str, limit: int = 3) -> List[Tuple[MockChunk, float]]:
        """Mock search function."""
        results = []
        query_lower = query.lower()
        
        for chunk in self.knowledge:
            # Simple keyword matching
            score = 0.0
            for word in query_lower.split():
                if word in chunk.content.lower():
                    score += 0.3
            
            if score > 0:
                results.append((chunk, score))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]


class ConversationHistory:
    """Manages conversation history and context."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[dict] = []
        self.session_start = datetime.now()
    
    def add_exchange(self, question: str, answer: str, search_results: List = None):
        """Add a question-answer exchange to history."""
        exchange = {
            "timestamp": datetime.now(),
            "question": question,
            "answer": answer,
            "search_results": search_results or []
        }
        self.history.append(exchange)
        
        # Keep only the most recent exchanges
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context_summary(self) -> str:
        """Generate a context summary from recent conversation history."""
        if not self.history:
            return ""
        
        context_parts = []
        for exchange in self.history[-3:]:  # Last 3 exchanges for context
            context_parts.append(f"Q: {exchange['question']}")
            context_parts.append(f"A: {exchange['answer'][:200]}...")  # Truncate long answers
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear conversation history."""
        self.history.clear()
        self.session_start = datetime.now()


class SimpleQAAgent:
    """Simple QA agent for demonstration."""
    
    def __init__(self, client: MockClient):
        self.client = client
        self.conversation_history = ConversationHistory()
    
    async def answer_with_context(self, question: str) -> Tuple[str, List]:
        """Answer a question with conversation context."""
        # Get conversation context
        context = self.conversation_history.get_context_summary()
        
        # Search for relevant information
        search_results = await self.client.search(question, limit=3)
        
        # Generate answer based on search results
        if search_results:
            answer_parts = ["Based on the available information:\n"]
            for i, (chunk, score) in enumerate(search_results, 1):
                answer_parts.append(f"{i}. {chunk.content}")
            
            answer = "\n".join(answer_parts)
            
            # Add context if available
            if context:
                answer += f"\n\nConsidering our previous conversation, this information should help answer your question about: {question}"
        else:
            answer = "I couldn't find relevant information in the knowledge base to answer your question."
        
        # Add to conversation history
        self.conversation_history.add_exchange(question, answer, search_results)
        
        return answer, search_results


async def interactive_demo():
    """Run an interactive demo session."""
    print("🤖 Simple Interactive QA Demo")
    print("=" * 40)
    print()
    print("This demo shows the core conversation functionality.")
    print("Type 'quit' to exit, 'history' to see conversation history, 'clear' to reset.")
    print()
    
    # Setup
    client = MockClient()
    qa_agent = SimpleQAAgent(client)
    
    print("💡 Try asking questions like:")
    print("   • 'What is Python?'")
    print("   • 'What are its features?'")
    print("   • 'Tell me about machine learning'")
    print("   • 'What are the types of ML?'")
    print()
    
    while True:
        try:
            # Get user input
            question = input("❓ Ask a question: ").strip()
            
            if not question:
                continue
            
            # Handle special commands
            if question.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif question.lower() == 'history':
                print("\n📜 Conversation History:")
                if not qa_agent.conversation_history.history:
                    print("   No conversation history yet.")
                else:
                    for i, exchange in enumerate(qa_agent.conversation_history.history, 1):
                        timestamp = exchange["timestamp"].strftime("%H:%M:%S")
                        print(f"\n{i}. [{timestamp}]")
                        print(f"   Q: {exchange['question']}")
                        print(f"   A: {exchange['answer'][:100]}...")
                print()
                continue
            elif question.lower() == 'clear':
                qa_agent.conversation_history.clear()
                print("✅ Conversation history cleared.\n")
                continue
            
            # Process question
            print(f"\n🤔 Thinking about: {question}")
            answer, search_results = await qa_agent.answer_with_context(question)
            
            # Display answer
            print(f"\n🤖 Answer:")
            print(answer)
            
            # Display sources
            if search_results:
                print(f"\n📚 Sources (found {len(search_results)} results):")
                for i, (chunk, score) in enumerate(search_results, 1):
                    source = chunk.document_uri or "Unknown"
                    print(f"   {i}. {source} (score: {score:.2f})")
            
            print("\n" + "-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue


if __name__ == "__main__":
    print("Starting Simple Interactive QA Demo...")
    asyncio.run(interactive_demo())
