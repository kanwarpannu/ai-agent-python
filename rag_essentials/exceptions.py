class RAGError(Exception):
    """Base exception for the rag_essentials package."""


class ConfigurationError(RAGError):
    """Raised when required configuration is invalid or missing."""


class IngestionError(RAGError):
    """Raised when PDF loading/chunking/indexing fails."""


class RetrievalError(RAGError):
    """Raised when vector search fails."""


class LLMInvocationError(RAGError):
    """Raised when LLM invocation fails."""
