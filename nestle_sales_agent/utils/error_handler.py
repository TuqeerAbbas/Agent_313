# utils/error_handler.py

import time
from typing import Dict, Optional

class ConversationError(Exception):
    """
    Enhanced custom exception class for conversation handling.
    Provides detailed error information and context for better debugging and handling.
    """
    def __init__(self, message: str, error_type: str, context: Optional[Dict] = None):
        self.message = message
        self.error_type = error_type
        self.context = context or {}
        self.timestamp = time.time()
        
        # Add error categorization
        self.severity = self._determine_severity()
        
        super().__init__(self.message)
        
    def _determine_severity(self) -> str:
        """Determine error severity based on type"""
        high_severity_types = {
            "safety_violation", 
            "medical_emergency",
            "security_breach"
        }
        
        medium_severity_types = {
            "processing_error",
            "validation_error",
            "state_error"
        }
        
        if self.error_type in high_severity_types:
            return "high"
        elif self.error_type in medium_severity_types:
            return "medium"
        return "low"
        
    def get_error_details(self) -> Dict:
        """Get formatted error details for logging and analysis"""
        return {
            "message": self.message,
            "type": self.error_type,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "context": self.context
        }