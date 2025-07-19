"""
Tests for SiliconFlow embedding provider.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from haiku.rag.config import Config


@pytest.mark.asyncio
async def test_siliconflow_embedder_success():
    """Test successful SiliconFlow embedding generation."""
    # Mock configuration
    with patch.object(Config, 'SILICONFLOW_API_KEY', 'test-api-key'), \
         patch.object(Config, 'SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1'), \
         patch.object(Config, 'EMBEDDINGS_MODEL', 'Qwen/Qwen3-Embedding-8B'), \
         patch.object(Config, 'EMBEDDINGS_VECTOR_DIM', 4096):
        
        try:
            from haiku.rag.embeddings.siliconflow import Embedder as SiliconFlowEmbedder
        except ImportError:
            pytest.skip("httpx package not installed")
        
        embedder = SiliconFlowEmbedder("Qwen/Qwen3-Embedding-8B", 4096)
        
        # Mock successful API response
        mock_response_data = {
            "data": [
                {
                    "embedding": [0.1] * 4096,
                    "index": 0,
                    "object": "embedding"
                }
            ],
            "model": "Qwen/Qwen3-Embedding-8B",
            "object": "list",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            
            # Test embedding generation
            embedding = await embedder.embed("test text")
            
            # Verify results
            assert len(embedding) == 4096
            assert all(isinstance(x, float) for x in embedding)
            assert embedding == [0.1] * 4096
            
            # Verify API call
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[1]['json']['model'] == 'Qwen/Qwen3-Embedding-8B'
            assert call_args[1]['json']['input'] == 'test text'
            assert call_args[1]['headers']['Authorization'] == 'Bearer test-api-key'


@pytest.mark.asyncio
async def test_siliconflow_embedder_missing_api_key():
    """Test SiliconFlow embedder with missing API key."""
    with patch.object(Config, 'SILICONFLOW_API_KEY', ''), \
         patch.object(Config, 'SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1'):
        
        try:
            from haiku.rag.embeddings.siliconflow import Embedder as SiliconFlowEmbedder
        except ImportError:
            pytest.skip("httpx package not installed")
        
        with pytest.raises(ValueError, match="SILICONFLOW_API_KEY environment variable is required"):
            SiliconFlowEmbedder("Qwen/Qwen3-Embedding-8B", 4096)


@pytest.mark.asyncio
async def test_siliconflow_embedder_api_error():
    """Test SiliconFlow embedder with API error."""
    with patch.object(Config, 'SILICONFLOW_API_KEY', 'test-api-key'), \
         patch.object(Config, 'SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1'):
        
        try:
            from haiku.rag.embeddings.siliconflow import Embedder as SiliconFlowEmbedder
        except ImportError:
            pytest.skip("httpx package not installed")
        
        embedder = SiliconFlowEmbedder("Qwen/Qwen3-Embedding-8B", 4096)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Mock HTTP error
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.json.return_value = {
                "error": {"message": "Invalid API key"}
            }
            
            http_error = httpx.HTTPStatusError(
                "401 Unauthorized", 
                request=MagicMock(), 
                response=mock_response
            )
            mock_client.post.side_effect = http_error
            
            with pytest.raises(RuntimeError, match="SiliconFlow API error.*401.*Invalid API key"):
                await embedder.embed("test text")


@pytest.mark.asyncio
async def test_siliconflow_embedder_wrong_dimension():
    """Test SiliconFlow embedder with wrong embedding dimension."""
    with patch.object(Config, 'SILICONFLOW_API_KEY', 'test-api-key'), \
         patch.object(Config, 'SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1'):
        
        try:
            from haiku.rag.embeddings.siliconflow import Embedder as SiliconFlowEmbedder
        except ImportError:
            pytest.skip("httpx package not installed")
        
        embedder = SiliconFlowEmbedder("Qwen/Qwen3-Embedding-8B", 4096)
        
        # Mock response with wrong dimension
        mock_response_data = {
            "data": [
                {
                    "embedding": [0.1] * 1024,  # Wrong dimension
                    "index": 0,
                    "object": "embedding"
                }
            ]
        }
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            
            with pytest.raises(ValueError, match="Expected embedding dimension 4096, got 1024"):
                await embedder.embed("test text")


@pytest.mark.asyncio
async def test_siliconflow_embedder_network_error():
    """Test SiliconFlow embedder with network error."""
    with patch.object(Config, 'SILICONFLOW_API_KEY', 'test-api-key'), \
         patch.object(Config, 'SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1'):
        
        try:
            from haiku.rag.embeddings.siliconflow import Embedder as SiliconFlowEmbedder
        except ImportError:
            pytest.skip("httpx package not installed")
        
        embedder = SiliconFlowEmbedder("Qwen/Qwen3-Embedding-8B", 4096)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Mock network error
            mock_client.post.side_effect = httpx.RequestError("Connection failed")
            
            with pytest.raises(RuntimeError, match="SiliconFlow API request failed.*Connection failed"):
                await embedder.embed("test text")


def test_siliconflow_embedder_factory():
    """Test SiliconFlow embedder through factory function."""
    with patch.object(Config, 'EMBEDDINGS_PROVIDER', 'siliconflow'), \
         patch.object(Config, 'EMBEDDINGS_MODEL', 'Qwen/Qwen3-Embedding-8B'), \
         patch.object(Config, 'EMBEDDINGS_VECTOR_DIM', 4096), \
         patch.object(Config, 'SILICONFLOW_API_KEY', 'test-api-key'):
        
        try:
            from haiku.rag.embeddings import get_embedder
            embedder = get_embedder()
            assert embedder._model == 'Qwen/Qwen3-Embedding-8B'
            assert embedder._vector_dim == 4096
        except ImportError:
            pytest.skip("httpx package not installed")


def test_siliconflow_embedder_factory_missing_httpx():
    """Test SiliconFlow embedder factory with missing httpx."""
    with patch.object(Config, 'EMBEDDINGS_PROVIDER', 'siliconflow'):
        
        # Mock missing httpx import
        with patch('haiku.rag.embeddings.siliconflow.httpx', None):
            with patch.dict('sys.modules', {'haiku.rag.embeddings.siliconflow': None}):
                from haiku.rag.embeddings import get_embedder
                
                with pytest.raises(ImportError, match="SiliconFlow embedder requires the 'httpx' package"):
                    get_embedder()


@pytest.mark.asyncio
async def test_siliconflow_embedder_integration():
    """Integration test with real-like configuration."""
    # This test uses the actual configuration structure
    test_config = {
        'SILICONFLOW_API_KEY': 'sk-test123',
        'SILICONFLOW_BASE_URL': 'https://api.siliconflow.cn/v1',
        'EMBEDDINGS_MODEL': 'Qwen/Qwen3-Embedding-8B',
        'EMBEDDINGS_VECTOR_DIM': 4096
    }
    
    with patch.multiple(Config, **test_config):
        try:
            from haiku.rag.embeddings.siliconflow import Embedder as SiliconFlowEmbedder
        except ImportError:
            pytest.skip("httpx package not installed")
        
        embedder = SiliconFlowEmbedder(
            test_config['EMBEDDINGS_MODEL'], 
            test_config['EMBEDDINGS_VECTOR_DIM']
        )
        
        # Mock successful response
        mock_embedding = [0.1] * 4096
        mock_response_data = {
            "data": [{"embedding": mock_embedding}]
        }
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            
            # Test embedding
            result = await embedder.embed("Hello, world!")
            
            assert result == mock_embedding
            assert len(result) == 4096
            
            # Verify correct API call
            call_args = mock_client.post.call_args
            assert "embeddings" in call_args[0][0]  # URL contains embeddings endpoint
            assert call_args[1]['json']['model'] == 'Qwen/Qwen3-Embedding-8B'
            assert call_args[1]['json']['input'] == 'Hello, world!'
