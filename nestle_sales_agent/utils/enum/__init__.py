"""
This module provides easy access to enumeration classes used throughout the application
for maintaining consistent categorization and type safety.
"""

from .category_enum import CategoryEnum

# Make CategoryEnum available when importing from utils.enums
__all__ = ['CategoryEnum']