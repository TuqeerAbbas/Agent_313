# utils/conversation_utils.py

import re
import logging
from typing import Dict, Tuple, Optional, Set
import time

class ConversationUtils:
    """Utility class providing comprehensive conversation management support."""
    
    def __init__(self, conversation_manager):
        self.manager = conversation_manager
        
    def validate_age_group(self, age_group: str) -> Tuple[bool, Optional[str]]:
        """Enhanced age group validation with better normalization"""
        valid_age_groups = {
            "0-6 months": {"min_months": 0, "max_months": 6},
            "6-12 months": {"min_months": 6, "max_months": 12},
            "12-18 months": {"min_months": 12, "max_months": 18},
            "18-24 months": {"min_months": 18, "max_months": 24},
            "2-3 years": {"min_months": 24, "max_months": 36},
            "3-4 years": {"min_months": 36, "max_months": 48}
        }
        
        try:
            if not age_group:
                return False, None
                
            normalized = self._normalize_age_group(age_group)
            
            # Direct match check
            if normalized in valid_age_groups:
                return True, normalized
                
            # Try to interpret and convert
            converted = self._convert_age_group(normalized, valid_age_groups)
            if converted:
                return True, converted
                
            return False, None
            
        except Exception as e:
            logging.error(f"Age group validation error: {str(e)}")
            return False, None
            
    def _normalize_age_group(self, age_group: str) -> str:
        """Normalize age group format"""
        age_group = age_group.lower().strip()
        
        # Remove extra spaces
        age_group = re.sub(r'\s+', ' ', age_group)
        
        # Standardize unit names
        age_group = age_group.replace('mos', 'months')
        age_group = age_group.replace('mo', 'months')
        age_group = age_group.replace('yrs', 'years')
        age_group = age_group.replace('yr', 'years')
        
        return age_group

    def sanitize_input(self, input_text: str) -> str:
        """Sanitizes user input to prevent injection and ensure safe processing"""
        # Remove potential HTML/XML
        sanitized = re.sub(r'<[^>]+>', '', input_text)
        
        # Remove potential SQL injection patterns
        sanitized = re.sub(r';\s*DROP|;\s*DELETE|;\s*UPDATE|;\s*INSERT', '', sanitized, 
                          flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        sanitized = ' '.join(sanitized.split())
        
        # Remove potential script tags and common XSS patterns
        sanitized = re.sub(r'<script|javascript:|data:', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()

    def validate_age_content(self, content: str, age_group: str) -> Tuple[bool, list]:
        """Validates content appropriateness for specific age group"""
        age_rules = {
            "0-6 months": {
                "restricted_topics": ["solid food", "walking", "talking"],
                "required_warnings": ["always consult pediatrician", "supervision required"]
            },
            "6-12 months": {
                "restricted_topics": ["cow's milk", "honey", "hard foods"],
                "required_warnings": ["choking hazard", "allergies"]
            }
            # Add more age groups as needed
        }
        
        if age_group not in age_rules:
            return True, []  # Default to safe if unknown age group
            
        rules = age_rules[age_group]
        warnings = []
        
        # Check for restricted topics
        for topic in rules["restricted_topics"]:
            if topic.lower() in content.lower():
                warnings.append(f"Content contains restricted topic for {age_group}: {topic}")
        
        # Check for required warnings
        for warning in rules["required_warnings"]:
            if warning.lower() not in content.lower():
                warnings.append(f"Missing required warning for {age_group}: {warning}")
                
        return len(warnings) == 0, warnings

    def calculate_similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)