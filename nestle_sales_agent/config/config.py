# config/config.py
import os

class Config:
    """Central configuration management"""
    # OpenAI Settings
    OPENAI_API_KEY = "OPEN_API_KEY"
    GPT_MODEL = "gpt-4o"
    MAX_TOKENS = 3000
    
    # ChromaDB Settings
    CHROMA_BASE_PATH = "Your_PATH"
    COLLECTION_SETTINGS = {"hnsw:space": "cosine"}
    
    # Collection Names
    COLLECTIONS = {
        "product": "product_collection",
        "recipe": "recipe_collection",
        "health": "health_concern_collection",
        "parenting": "parenting_advice_collection",
        "general": "general_collection"
    }
    
    # Session Settings
    SESSION_TIMEOUT = 1800  # 30 minutes
    MAX_SESSION_DURATION = 7200  # 2 hours
    
    # Conversation Settings
    MAX_CONVERSATION_BUFFER = 10
    REACT_CONFIDENCE_THRESHOLD = 0.7
    MAX_ITEMS_PER_RESPONSE = 3
    
    @classmethod
    def get_chroma_config(cls, category: str):
        """Get ChromaDB configuration for a specific category"""
        return {
            "path": cls.CHROMA_BASE_PATH,
            "collection_name": cls.COLLECTIONS.get(category, "general_collection"),
            "settings": cls.COLLECTION_SETTINGS
        }
