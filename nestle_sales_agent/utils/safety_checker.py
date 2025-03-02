# utils/safety_checker.py

import re
import logging
from typing import Dict, List, Optional, Tuple

class SafetyChecker:
    """Simplified safety checking system focusing on inappropriate content and PII."""
    
    def __init__(self):
        # Only critical inappropriate patterns
        self.inappropriate_patterns = [
            r'\b(porn|xxx|adult content|explicit sex)\b',
            r'\b(illegal drugs|cocaine|heroin)\b',
            r'\b(suicide|kill myself|self harm)\b',
            r'\b(terrorist|bomb-making|extremist)\b'
        ]
        
        # PII patterns
        self.pii_patterns = {
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "address": r"\b\d+\s+[A-Za-z\s,]+\b"
        }

    def check_content_safety(self, content: str, age_group: Optional[str] = None) -> Dict:
        """Basic safety check focusing only on critical concerns"""
        logging.info(f"\n=== Safety Check ===\nChecking content: {content}")
        
        try:
            results = {
                "is_safe": True,
                "concerns": [],
                "risk_level": "low",
            }
            
            # Check for inappropriate content
            if self._check_inappropriate_content(content):
                logging.warning("Inappropriate content detected")
                results["is_safe"] = False
                results["concerns"].append("inappropriate_content")
            
            # Check for PII
            pii_check = self._check_pii(content)
            if pii_check["has_pii"]:
                logging.warning(f"PII detected: {pii_check['pii_types']}")
                results["concerns"].append("contains_pii")
            
            # If content contains food/recipe related terms, mark as safe
            food_terms = ["recipe", "food", "eat", "meal", "feeding", "nutrition", 
                         "cook", "baby food", "snack"]
            if any(term in content.lower() for term in food_terms):
                results["is_safe"] = True
                
            logging.info(f"Safety check results: {results}")
            return results
            
        except Exception as e:
            logging.error(f"Error in content safety check: {str(e)}")
            return {"is_safe": True, "concerns": [], "risk_level": "low"}

    def _check_inappropriate_content(self, content: str) -> bool:
        """Check only for seriously inappropriate content"""
        return any(re.search(pattern, content.lower()) for pattern in self.inappropriate_patterns)

    def _check_pii(self, content: str) -> Dict:
        """Check for personal identifiable information"""
        results = {"has_pii": False, "pii_types": []}
        
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, content):
                results["has_pii"] = True
                results["pii_types"].append(pii_type)
                
        return results

    def sanitize_content(self, content: str) -> str:
        """Remove or mask sensitive information"""
        sanitized = content
        
        # Mask PII
        for pii_type, pattern in self.pii_patterns.items():
            sanitized = re.sub(pattern, f"[MASKED_{pii_type.upper()}]", sanitized)
        
        # Remove inappropriate content
        for pattern in self.inappropriate_patterns:
            sanitized = re.sub(pattern, "[REMOVED]", sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def validate_input(self, user_message: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Validate and sanitize user input"""
        if not user_message or not user_message.strip():
            return False, None, "Please provide a message to continue our conversation."
            
        if len(user_message) > 1000:
            return False, None, "Message is too long. Please provide a shorter message."
            
        # Sanitize input
        sanitized_message = self.utils.sanitize_input(user_message)
        
        # Check content safety
        safety_result = self.safety_checker.check_content_safety(sanitized_message)
        if not safety_result["is_safe"]:
            return False, None, "I apologize, but I cannot process that type of content."
            
        return True, sanitized_message, None