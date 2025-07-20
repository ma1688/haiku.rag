"""
Tests for the interactive QA system.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datasets import Dataset

from haiku.rag.client import HaikuRAG
from haiku.rag.qa.interactive import (
    ConversationHistory,
    ContextAwareQAAgent,
    InteractiveQASession,
    start_interactive_qa
)


class TestConversationHistory:
    """Test conversation history management."""
    
    def test_init(self):
        """Test conversation history initialization."""
        history = ConversationHistory(max_history=5)
        assert history.max_history == 5
        assert len(history.history) == 0
        assert history.session_start is not None
    
    def test_add_exchange(self):
        """Test adding question-answer exchanges."""
        history = ConversationHistory(max_history=3)
        
        # Add first exchange
        history.add_exchange("What is Python?", "Python is a programming language.")
        assert len(history.history) == 1
        assert history.history[0]["question"] == "What is Python?"
        assert history.history[0]["answer"] == "Python is a programming language."
        
        # Add more exchanges
        history.add_exchange("What is JavaScript?", "JavaScript is a scripting language.")
        history.add_exchange("What is Java?", "Java is a programming language.")
        assert len(history.history) == 3
        
        # Test max history limit
        history.add_exchange("What is C++?", "C++ is a programming language.")
        assert len(history.history) == 3  # Should still be 3 due to max_history
        assert history.history[0]["question"] == "What is JavaScript?"  # First one removed
    
    def test_get_context_summary(self):
        """Test context summary generation."""
        history = ConversationHistory()
        
        # Empty history
        assert history.get_context_summary() == ""
        
        # Add exchanges
        history.add_exchange("What is Python?", "Python is a programming language used for web development, data science, and automation.")
        history.add_exchange("What are its benefits?", "Python is easy to learn, has great libraries, and is versatile.")
        
        context = history.get_context_summary()
        assert "What is Python?" in context
        assert "What are its benefits?" in context
        assert "Python is a programming language" in context
    
    def test_clear(self):
        """Test clearing conversation history."""
        history = ConversationHistory()
        history.add_exchange("Test question", "Test answer")
        assert len(history.history) == 1
        
        history.clear()
        assert len(history.history) == 0


@pytest.mark.asyncio
class TestContextAwareQAAgent:
    """Test context-aware QA agent."""
    
    async def test_init(self):
        """Test agent initialization."""
        client = HaikuRAG(":memory:")
        agent = ContextAwareQAAgent(client)
        
        assert agent._client == client
        assert agent.conversation_history is not None
        assert agent.max_context_length == 2000
    
    @patch('haiku.rag.qa.interactive.get_qa_agent')
    async def test_answer_with_context(self, mock_get_qa_agent):
        """Test answering with conversation context."""
        # Setup mocks
        mock_base_agent = AsyncMock()
        mock_base_agent.answer.return_value = "Python is a programming language."
        mock_get_qa_agent.return_value = mock_base_agent
        
        mock_client = AsyncMock()
        mock_client.search.return_value = [
            (MagicMock(content="Python documentation", document_uri="python.org"), 0.9)
        ]
        
        # Create agent
        agent = ContextAwareQAAgent(mock_client)
        agent.base_agent = mock_base_agent
        
        # Test first question (no context)
        answer, search_results = await agent.answer_with_context("What is Python?")
        
        assert answer == "Python is a programming language."
        assert len(search_results) == 1
        assert len(agent.conversation_history.history) == 1
        
        # Test second question (with context)
        mock_base_agent.answer.return_value = "Python is great for beginners."
        answer, search_results = await agent.answer_with_context("Is it good for beginners?")
        
        assert answer == "Python is great for beginners."
        assert len(agent.conversation_history.history) == 2
        
        # Verify that context was used in the enhanced question
        call_args = mock_base_agent.answer.call_args[0][0]
        assert "Previous conversation context:" in call_args
        assert "What is Python?" in call_args
    
    @patch('haiku.rag.qa.interactive.get_qa_agent')
    async def test_answer_compatibility(self, mock_get_qa_agent):
        """Test standard answer method for compatibility."""
        mock_base_agent = AsyncMock()
        mock_base_agent.answer.return_value = "Test answer"
        mock_get_qa_agent.return_value = mock_base_agent
        
        mock_client = AsyncMock()
        mock_client.search.return_value = []
        
        agent = ContextAwareQAAgent(mock_client)
        agent.base_agent = mock_base_agent
        
        answer = await agent.answer("Test question")
        assert answer == "Test answer"


@pytest.mark.asyncio
class TestInteractiveQASession:
    """Test interactive QA session."""
    
    async def test_context_manager(self):
        """Test async context manager functionality."""
        session = InteractiveQASession(":memory:")
        
        async with session:
            assert session.client is not None
            assert session.qa_agent is not None
            assert isinstance(session.qa_agent, ContextAwareQAAgent)
    
    @patch('haiku.rag.qa.interactive.Prompt.ask')
    async def test_search_command(self, mock_prompt):
        """Test search command handling."""
        session = InteractiveQASession(":memory:")
        
        async with session:
            # Mock search results
            session.client.search = AsyncMock(return_value=[
                (MagicMock(content="Test content", document_uri="test.txt"), 0.8)
            ])
            
            # Test search command
            await session._handle_search_command("test query")
            
            # Verify search was called
            session.client.search.assert_called_once_with("test query", limit=5)
    
    async def test_display_methods(self):
        """Test display methods don't raise errors."""
        session = InteractiveQASession(":memory:")
        
        async with session:
            # These should not raise exceptions
            session._display_welcome()
            session._display_question("Test question?")
            session._display_help()
            session._display_history()
            
            # Test with mock search results
            mock_chunk = MagicMock()
            mock_chunk.content = "Test content"
            mock_chunk.document_uri = "test.txt"
            search_results = [(mock_chunk, 0.9)]
            
            session._display_answer("Test answer", search_results)


@pytest.mark.asyncio
async def test_start_interactive_qa():
    """Test starting interactive QA session."""
    with patch('haiku.rag.qa.interactive.InteractiveQASession') as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        await start_interactive_qa(":memory:", "test-model")
        
        mock_session_class.assert_called_once_with(":memory:", "test-model")
        mock_session.run.assert_called_once()


def test_conversation_history_integration():
    """Test conversation history integration with real data."""
    history = ConversationHistory(max_history=2)
    
    # Simulate a conversation
    exchanges = [
        ("What is machine learning?", "Machine learning is a subset of AI that enables computers to learn without explicit programming."),
        ("What are its applications?", "ML is used in recommendation systems, image recognition, natural language processing, and more."),
        ("How does it work?", "ML algorithms find patterns in data and use them to make predictions or decisions.")
    ]
    
    for question, answer in exchanges:
        history.add_exchange(question, answer)
    
    # Should only keep the last 2 exchanges
    assert len(history.history) == 2
    assert history.history[0]["question"] == "What are its applications?"
    assert history.history[1]["question"] == "How does it work?"
    
    # Test context generation
    context = history.get_context_summary()
    assert "What are its applications?" in context
    assert "How does it work?" in context
    assert "What is machine learning?" not in context  # Should be removed due to limit


@pytest.mark.asyncio
async def test_qa_corpus_integration(qa_corpus: Dataset):
    """Test interactive QA with real corpus data."""
    async with HaikuRAG(":memory:") as client:
        # Add a document from the corpus
        doc = qa_corpus[0]
        await client.create_document(
            content=doc["document_extracted"], 
            uri=doc["document_id"]
        )
        
        # Create context-aware agent
        agent = ContextAwareQAAgent(client)
        
        # Test question answering
        question = "What is this document about?"
        answer, search_results = await agent.answer_with_context(question)
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(search_results, list)
        assert len(agent.conversation_history.history) == 1
        
        # Test follow-up question
        follow_up = "Can you provide more details?"
        answer2, search_results2 = await agent.answer_with_context(follow_up)
        
        assert isinstance(answer2, str)
        assert len(answer2) > 0
        assert len(agent.conversation_history.history) == 2
