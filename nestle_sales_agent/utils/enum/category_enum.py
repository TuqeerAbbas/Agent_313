# utils/enums/category_enum.py

from enum import Enum
from typing import Optional

class CategoryEnum(Enum):
    """
    Centralized enumeration for managing conversation categories.
    Provides consistent mapping between full names and short identifiers.
    """
    PRODUCT = "Product Recommendation"
    RECIPE = "Recipe Recommendation"
    HEALTH = "Health Concern" 
    PARENTING = "Parenting Advice"

    @staticmethod
    def get_short_name(category: str) -> str:
        category_map = {
            "Product Recommendation": "product",
            "Recipe Recommendation": "recipe",
            "Health Concern": "health",
            "Parenting Advice": "parenting",
            # Add mappings for variant forms
            "parenting_advice": "parenting",
            "health_concern": "health"
        }
        return category_map.get(category, category.lower())

    @classmethod
    def get_full_name(cls, short_name: str) -> Optional[str]:
        """
        Converts a short identifier to its full category name.
        Used for user-facing responses and logging.
        
        Args:
            short_name: Short identifier (e.g. "product")
            
        Returns:
            Full category name (e.g. "Product Recommendation") or None if not found
        """
        mapping = {
            "product": cls.PRODUCT.value,
            "recipe": cls.RECIPE.value,
            "health": cls.HEALTH.value,
            "parenting": cls.PARENTING.value
        }
        return mapping.get(short_name)

    @classmethod
    def is_valid_category(cls, category: str) -> bool:
        """
        Checks if a given string is a valid category name (either full or short).
        
        Args:
            category: Category name to validate
            
        Returns:
            True if category is valid, False otherwise
        """
        return (category in [e.value for e in cls] or 
                category in [cls.get_short_name(e.value) for e in cls])