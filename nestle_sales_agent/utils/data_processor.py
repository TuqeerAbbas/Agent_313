# utils/data_processor.py

import uuid
import time
import logging
from typing import Dict, List, Tuple, Any, Optional

class DataProcessor:
    """
    Enhanced data processor with improved validation and transformation capabilities.
    Handles standardization of various data types used in conversations.
    """
    
    def __init__(self):
        self.validators = {
            "product": self._validate_product_data,
            "recipe": self._validate_recipe_data,
            "health": self._validate_health_data,
            "parenting": self._validate_parenting_data
        }
        
        self.transformers = {
            "product": self._transform_product_data,
            "recipe": self._transform_recipe_data,
            "health": self._transform_health_data,
            "parenting": self._transform_parenting_data
        }
        
    def process_data(self, raw_data: Dict, data_type: str) -> Dict:
        """Process data with validation and transformation"""
        try:
            # Validate data
            validator = self.validators.get(data_type)
            if validator:
                is_valid, validation_notes = validator(raw_data)
                if not is_valid:
                    raise ValueError(f"Invalid {data_type} data: {validation_notes}")
            
            # Transform data
            transformer = self.transformers.get(data_type)
            if transformer:
                transformed_data = transformer(raw_data)
                return transformed_data
                
            raise ValueError(f"Unknown data type: {data_type}")
            
        except Exception as e:
            logging.error(f"Data processing error: {str(e)}")
            raise

    def _transform_product_data(self, raw_data: Dict) -> Dict:
        """Transform raw product data into standardized format"""
        return {
            "id": str(raw_data.get("id", str(uuid.uuid4()))),
            "name": raw_data.get("name", "Unnamed Product"),
            "category": raw_data.get("category", "Uncategorized"),
            "age_group": raw_data.get("age_group", "Not Specified"),
            "description": raw_data.get("description", ""),
            "features": raw_data.get("features", []),
            "safety_info": raw_data.get("safety_info", []),
            "usage_instructions": raw_data.get("usage_instructions", []),
            "metadata": {
                "last_updated": raw_data.get("last_updated", time.time()),
                "version": raw_data.get("version", "1.0")
            }
        }

    def _transform_recipe_data(self, raw_data: Dict) -> Dict:
        """Transform raw recipe data into standardized format"""
        return {
            "id": str(raw_data.get("id", str(uuid.uuid4()))),
            "name": raw_data.get("name", "Unnamed Recipe"),
            "category": "Recipe",
            "age_group": raw_data.get("age_group", "Not Specified"),
            "ingredients": raw_data.get("ingredients", []),
            "instructions": raw_data.get("instructions", []),
            "preparation_time": raw_data.get("preparation_time", "Not Specified"),
            "difficulty": raw_data.get("difficulty", "Not Specified"),
            "nutritional_info": raw_data.get("nutritional_info", {}),
            "dietary_flags": raw_data.get("dietary_flags", [])
        }

    def _validate_product_data(self, data: Dict) -> Tuple[bool, Optional[Dict]]:
        """Validate product data structure and content"""
        required_fields = {"name", "category", "age_group"}
        
        # Check required fields
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            return False, {"missing_fields": list(missing_fields)}
            
        return True, None

    # Add similar validation methods for other data types
    def _validate_recipe_data(self, data: Dict) -> Tuple[bool, Optional[Dict]]:
        """Validate recipe data structure and content"""
        required_fields = {"name", "ingredients", "instructions"}
        
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            return False, {"missing_fields": list(missing_fields)}
            
        return True, None
    
    def _validate_health_data(self, data: Dict) -> Tuple[bool, Optional[Dict]]:
        """Validate health data structure and content"""
        required_fields = {"topic", "content", "age_group"}
        
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            return False, {"missing_fields": list(missing_fields)}
            
        return True, None

    def _validate_parenting_data(self, data: Dict) -> Tuple[bool, Optional[Dict]]:
        """Validate parenting advice data structure and content"""
        required_fields = {"topic", "content", "age_group"}
        
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            return False, {"missing_fields": list(missing_fields)}
            
        return True, None

    def _transform_health_data(self, raw_data: Dict) -> Dict:
        """Transform raw health information into standardized format"""
        return {
            "id": str(raw_data.get("id", str(uuid.uuid4()))),
            "topic": raw_data.get("topic", "General Health"),
            "category": "Health",
            "age_group": raw_data.get("age_group", "Not Specified"),
            "content": raw_data.get("content", ""),
            "safety_notices": raw_data.get("safety_notices", []),
            "professional_consultation": raw_data.get("professional_consultation", True),
            "metadata": {
                "last_updated": raw_data.get("last_updated", time.time()),
                "version": raw_data.get("version", "1.0"),
                "verified_by": raw_data.get("verified_by", "System")
            }
        }

    def _transform_parenting_data(self, raw_data: Dict) -> Dict:
        """Transform raw parenting advice into standardized format"""
        return {
            "id": str(raw_data.get("id", str(uuid.uuid4()))),
            "topic": raw_data.get("topic", "General Parenting"),
            "category": "Parenting",
            "age_group": raw_data.get("age_group", "Not Specified"),
            "content": raw_data.get("content", ""),
            "developmental_stage": raw_data.get("developmental_stage", "Not Specified"),
            "considerations": raw_data.get("considerations", []),
            "metadata": {
                "last_updated": raw_data.get("last_updated", time.time()),
                "version": raw_data.get("version", "1.0")
            }
        }