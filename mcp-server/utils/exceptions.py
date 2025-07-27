class MCPError(Exception):
    """Base exception for MCP-related errors."""
    
    def __init__(self, message: str, code: int = -32603):
        self.message = message
        self.code = code
        super().__init__(message)


class MCPMethodNotFoundError(MCPError):
    """Exception raised when an MCP method is not found."""
    
    def __init__(self, method: str):
        super().__init__(f"Method '{method}' not found", code=-32601)


class MCPInvalidParamsError(MCPError):
    """Exception raised when MCP method parameters are invalid."""
    
    def __init__(self, message: str = "Invalid parameters"):
        super().__init__(message, code=-32602)


class LLMProviderError(Exception):
    """Exception raised when LLM provider encounters an error."""
    pass


class SessionStorageError(Exception):
    """Exception raised when session storage encounters an error."""
    pass


class VectorStoreError(Exception):
    """Exception raised when vector store encounters an error."""
    pass


class PromptBuilderError(Exception):
    """Exception raised when prompt builder encounters an error."""
    pass