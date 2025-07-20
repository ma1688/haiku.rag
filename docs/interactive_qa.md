# Interactive QA System

The Interactive QA system provides a conversational interface for querying your knowledge base. It maintains conversation context, allowing for natural follow-up questions and multi-turn conversations.

## Features

- **Conversational Context**: Remembers previous questions and answers for natural follow-ups
- **Rich Terminal Interface**: Beautiful formatting with colors, emojis, and markdown rendering
- **Search Integration**: Direct access to document search functionality
- **History Management**: View and clear conversation history
- **Multiple Commands**: Built-in commands for enhanced interaction
- **Real-time Feedback**: Shows thinking indicators and search sources

## Quick Start

### Using the CLI

Start an interactive chat session:

```bash
# Use default database
haiku-rag chat

# Specify custom database
haiku-rag chat --db /path/to/your/database.db

# Use specific model
haiku-rag chat --model gpt-4o-mini
```

### Using Python API

```python
import asyncio
from haiku.rag.qa.interactive import start_interactive_qa

# Start interactive session
asyncio.run(start_interactive_qa("database.db"))
```

### Running the Demo

Try the included demo with sample documents:

```bash
python examples/interactive_qa_demo.py
```

## Available Commands

During an interactive session, you can use these special commands:

| Command | Description |
|---------|-------------|
| `/help` | Show help information and available commands |
| `/history` | Display conversation history with timestamps |
| `/clear` | Clear conversation history and start fresh |
| `/search <query>` | Search documents directly without QA processing |
| `/quit` or `/exit` | Exit the interactive session |

## Example Conversation

```
🤖 Welcome to Haiku RAG Interactive QA

Ask a question: What is Python?

❓ You: What is Python?

🤖 Assistant:
Python is a high-level, interpreted programming language with dynamic semantics. It's designed to be easy to learn and use, with extensive standard library support and cross-platform compatibility.

📚 Sources used:
1. Score: 0.892 | Source: python_guide.md
   Python is a high-level, interpreted programming language with dynamic semantics...

Ask a question: What are its main applications?

❓ You: What are its main applications?

🤖 Assistant:
Based on our previous discussion about Python, its main applications include:

- Web development (using frameworks like Django and Flask)
- Data science and analytics
- Artificial intelligence and machine learning
- Automation and scripting
- Scientific computing

The versatility of Python makes it suitable for rapid application development across many domains.
```

## Context Management

The system automatically maintains conversation context by:

1. **Storing Recent Exchanges**: Keeps the last 10 question-answer pairs
2. **Context Enhancement**: Adds relevant conversation history to new questions
3. **Smart Truncation**: Limits context length to prevent token overflow
4. **Temporal Tracking**: Records timestamps for each exchange

### Context Example

```python
# First question
"What is machine learning?"
# Answer: "Machine learning is a subset of AI..."

# Follow-up question (automatically enhanced with context)
"What are its types?"
# Enhanced internally to:
# "Previous conversation context:
#  Q: What is machine learning?
#  A: Machine learning is a subset of AI...
#  
#  Current question: What are its types?"
```

## Advanced Usage

### Custom QA Agent

```python
from haiku.rag.client import HaikuRAG
from haiku.rag.qa.interactive import ContextAwareQAAgent

async def custom_qa_session():
    async with HaikuRAG("database.db") as client:
        # Create custom agent with specific settings
        agent = ContextAwareQAAgent(
            client=client,
            model="gpt-4",
            max_context_length=3000
        )
        
        # Ask questions with context
        answer, sources = await agent.answer_with_context("Your question")
        print(f"Answer: {answer}")
        print(f"Sources: {len(sources)}")
```

### Programmatic Session

```python
from haiku.rag.qa.interactive import InteractiveQASession

async def automated_qa():
    async with InteractiveQASession("database.db") as session:
        # Simulate user interactions
        questions = [
            "What is Python?",
            "What are its benefits?",
            "How is it used in data science?"
        ]
        
        for question in questions:
            answer, sources = await session.qa_agent.answer_with_context(question)
            print(f"Q: {question}")
            print(f"A: {answer}\n")
```

## Configuration

### Environment Variables

The interactive QA system respects all standard haiku.rag configuration:

```bash
# QA Provider and Model
QA_PROVIDER="ollama"  # or "openai", "anthropic"
QA_MODEL="qwen3"      # or "gpt-4o-mini", "claude-3-5-haiku-20241022"

# Embedding Settings
EMBEDDINGS_PROVIDER="ollama"
EMBEDDINGS_MODEL="mxbai-embed-large"

# Ollama Configuration
OLLAMA_BASE_URL="http://localhost:11434"

# API Keys (if using commercial providers)
OPENAI_API_KEY="your-key"
ANTHROPIC_API_KEY="your-key"
```

### Conversation Settings

```python
from haiku.rag.qa.interactive import ConversationHistory, ContextAwareQAAgent

# Custom conversation history
history = ConversationHistory(max_history=20)  # Keep more history

# Custom context length
agent = ContextAwareQAAgent(
    client=client,
    max_context_length=5000  # Longer context
)
```

## Tips for Better Conversations

1. **Be Specific**: Ask clear, specific questions for better results
2. **Use Follow-ups**: Take advantage of context with follow-up questions
3. **Explore with /search**: Use direct search to understand available content
4. **Check Sources**: Review the sources shown with each answer
5. **Clear When Needed**: Use `/clear` to start fresh conversations on new topics

## Troubleshooting

### Common Issues

**No results found**
- Check if documents are properly indexed
- Try different search terms
- Use `/search` to explore available content

**Context too long**
- The system automatically truncates long contexts
- Use `/clear` to reset if needed
- Adjust `max_context_length` if using programmatically

**Model errors**
- Verify your QA provider is properly configured
- Check API keys for commercial providers
- Ensure Ollama is running for local models

### Debug Mode

For development and debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run your interactive session
await start_interactive_qa("database.db")
```

## Integration Examples

### Jupyter Notebook

```python
# In a Jupyter cell
import asyncio
from haiku.rag.qa.interactive import ContextAwareQAAgent
from haiku.rag.client import HaikuRAG

async def notebook_qa():
    async with HaikuRAG("database.db") as client:
        agent = ContextAwareQAAgent(client)
        
        # Interactive Q&A in notebook
        answer, sources = await agent.answer_with_context("Your question")
        return answer, sources

# Run in notebook
answer, sources = await notebook_qa()
print(answer)
```

### Web Application

```python
from fastapi import FastAPI
from haiku.rag.qa.interactive import ContextAwareQAAgent
from haiku.rag.client import HaikuRAG

app = FastAPI()
qa_sessions = {}  # Store sessions by user ID

@app.post("/chat/{user_id}")
async def chat(user_id: str, question: str):
    if user_id not in qa_sessions:
        client = HaikuRAG("database.db")
        await client.__aenter__()
        qa_sessions[user_id] = ContextAwareQAAgent(client)
    
    agent = qa_sessions[user_id]
    answer, sources = await agent.answer_with_context(question)
    
    return {
        "answer": answer,
        "sources": [{"content": s[0].content, "score": s[1]} for s in sources]
    }
```
