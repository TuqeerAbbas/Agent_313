# models/user_state.py
import time
from typing import Set, List, Dict, Any

class UserState:
    """Manages user state and conversation context"""
    def __init__(self):
        # Core state
        self.current_category: str = None
        self.age_group: str = None
        self.current_topic: str = None
        self.last_interaction = None
        self.conversation_history = []
        
        # Category-specific state tracking
        self.current_category = None  # Tracks current interaction category
        self.last_message = None      # Stores last user message
        self.last_react_decision = {} # Stores last ReACT analysis
        
        # Product-specific state
        self.current_products = []
        self.shown_products = set()
        self.all_available_products = []  # For pagination
        self.current_page = 0
        self.last_product_search = None
        
        # Recipe-specific state
        self.current_recipes = []
        self.shown_recipes = set()
        self.all_available_recipes = []   # For pagination
        self.recipe_current_page = 0
        self.last_recipe_search = None
        
        # Health concern state
        self.current_health_topics = []
        self.shown_health_info = set()
        self.all_available_health_info = []  # For pagination
        self.health_current_page = 0
        self.health_emergency_history = {}
        
        # Parenting advice state
        self.current_advice_topics = []
        self.shown_advice = set()
        self.all_available_advice = []    # For pagination
        self.advice_current_page = 0
        self.developmental_context = {}
        
        # Dietary preferences
        self.dietary_preferences = {}
        
        # General Q&A state
        self.shown_qa_info: Set[str] = set()
        self.current_qa_topics: List[str] = []
        self.last_qa_search: float = None
        self.topic_history: Dict[str, List] = {}
        self.redirection_history: Dict[str, List] = {}
        self.developmental_insights: Dict[str, Dict] = {}
        
        # Enhanced age tracking
        self.age_history = {
            "current": None,
            "previous": [],  # List of dicts with age and context
            "references": []  # Track how ages are referenced
        }
        
        # Enhanced category tracking
        self.category_history = {
            "current": None,
            "previous": [],  
            "context": {}  # Stores summaries for each category
        }
        
        # Conversation summary tracking
        self.topic_summaries = {}

         # Initialize search history
        self.search_history = {}
        self.current_search_parameters = None

        # Category states container
        self.category_states = {
            "Recipe Recommendation": {"shown_items": set(), "current_items": []},
            "Parenting Advice": {"shown_items": set(), "current_items": []},
            "Health Concern": {"shown_items": set(), "current_items": []},
            "General Q&A": {"shown_items": set(), "current_items": []},
            "Upsell/Cross-sell": {"shown_items": set(), "current_items": []}
        }

        # Initialize conversation context
        self.conversation_context: List[Dict] = []

    def reset_product_state(self) -> None:
        """Reset product-related state when changing context"""
        self.shown_products = set()
        self.current_products = []
        self.last_product_search = None
        self.shown_upsell_items = set()
        self.current_upsell_items = []
        self.original_product_context = None

    def reset_recipe_state(self) -> None:
        """Reset recipe-related state when changing context"""
        self.shown_recipes = set()
        self.current_recipes = []
        self.last_recipe_search = None
        # Don't reset dietary_preferences as they might be relevant across conversations

    def reset_parenting_state(self) -> None:
        """Reset parenting-related state when changing context"""
        self.shown_advice = set()
        self.current_advice_topics = []
        self.last_advice_search = None
        # Don't reset sensitivity_history and developmental_context

    def reset_health_state(self) -> None:
        """Reset health-related state when changing context"""
        self.shown_health_info = set()
        self.current_health_topics = []
        self.last_health_search = None
        # Don't reset emergency and referral history
    
    def reset_qa_state(self) -> None:
        """Reset Q&A-related state when changing context"""
        self.shown_qa_info = set()
        self.current_qa_topics = []
        self.last_qa_search = None
        # Don't reset topic_history and developmental_insights
              
    def update_state(self, category: str = None, age_group: str = None, topic: str = None) -> None:
        """Enhanced state update with optional parameters and tracking"""
        if category:
            # If category changes, reset shown items
            if category != self.current_category:
                self.reset_shown_items()
            self.current_category = category
            
        if age_group:
            # If age group changes, reset shown items
            if age_group != self.age_group:
                self.reset_shown_items()
            self.age_group = age_group
            
        if topic:
            self.current_topic = topic
            
    def reset_shown_items(self) -> None:
        """Reset tracking when changing context"""
        if self.current_category in self.category_states:
            self.category_states[self.current_category]["shown_items"] = set()
            self.category_states[self.current_category]["current_items"] = []
        
    def add_conversation_context(self, context_entry: Dict) -> None:
        """Track ReACT reasoning chain"""
        self.conversation_context.append({
            'timestamp': time.time(),
            'context': context_entry
        })
        # Keep only last 5 context entries
        if len(self.conversation_context) > 5:
            self.conversation_context = self.conversation_context[-5:]

    def get_category_state(self, category: str) -> Dict:
        """Get current state for a specific category"""
        return self.category_states.get(category, {
            "shown_items": set(),
            "current_items": []
        })

    def update_category_state(self, category: str) -> None:
        """Update current category and initialize pagination if needed"""
        self.current_category = category
        
        # Initialize or reset pagination state for new category
        if category == "product":
            if not hasattr(self, 'current_page'):
                self.current_page = 0
            if not hasattr(self, 'all_available_products'):
                self.all_available_products = []       
        elif category == "recipe":
            if not hasattr(self, 'recipe_current_page'):
                self.recipe_current_page = 0
            if not hasattr(self, 'all_available_recipes'):
                self.all_available_recipes = []
        elif category == "recipe":
            if not hasattr(self, 'health_current_page'):
                self.health_current_page = 0
            if not hasattr(self, 'all_available_health_info'):
                self.all_available_health_info = []
        elif category == "recipe":
            if not hasattr(self, 'advice_current_page'):
                self.advice_current_page = 0
            if not hasattr(self, 'all_available_advice'):
                self.all_available_advice = []

    def get_current_page(self, category: str) -> int:
        """Get current page for specified category"""
        category_map = {
            "product": self.current_page,
            "recipe": self.recipe_current_page,
            "health": self.health_current_page,
            "advice": self.advice_current_page
        }
        return category_map.get(category, 0)

    def reset_pagination_state(self):
        """Reset all pagination-related state"""
        self.current_page = 0
        self.recipe_current_page = 0
        self.health_current_page = 0
        self.advice_current_page = 0

    def get_all_available_items(self, category: str) -> List[Dict]:
        """Get all available items for specified category"""
        category_map = {
            "product": self.all_available_products,
            "recipe": self.all_available_recipes,
            "health": self.all_available_health_info,
            "advice": self.all_available_advice
        }
        return category_map.get(category, [])

    def get_shown_items(self, category: str) -> set:
        """Get shown items for specified category"""
        category_map = {
            "product": self.shown_products,
            "recipe": self.shown_recipes,
            "health": self.shown_health_info,
            "advice": self.shown_advice
        }
        return category_map.get(category, set())

    def update_pagination_state(self, category: str, page: int) -> None:
        """Update pagination state for specified category"""
        if category == "product":
            self.current_page = page
        elif category == "recipe":
            self.recipe_current_page = page
        elif category == "health":
            self.health_current_page = page
        elif category == "advice":
            self.advice_current_page = page