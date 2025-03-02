# managers/conversation_manager.py

from typing import Dict, List, Any, Optional, Tuple
import json
import logging
import time
import os
import chromadb
from langchain_openai import OpenAIEmbeddings
import openai
import uuid
from nltk.tokenize import sent_tokenize
import math
import re
from config.config import Config
from models.user_state import UserState
from utils.error_handler import ConversationError
from utils.data_processor import DataProcessor
from utils.safety_checker import SafetyChecker
from utils.conversation_utils import ConversationUtils
from utils.logging_utils import setup_logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.enum.category_enum import CategoryEnum
from .product_manager import ProductManager
from .recipe_manager import RecipeManager
from .parenting_advice_manager import ParentingAdviceManager
from .health_concern_manager import HealthConcernManager
from .general_qa_manager import GeneralQAManager

class ConversationManager:
    """
    Enhanced Conversation Manager that coordinates between specialized handlers
    while maintaining conversation flow and state management.
    """
    def __init__(self):
       # Initialize conversation tracking first
        self.conversation_buffer: List[Dict] = []
        setup_logging()
        
        # Rest of initialization
        self.config = Config()
        chroma_config = self.config.get_chroma_config('general')
        self.client = chromadb.PersistentClient(path=chroma_config["path"])
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=self.config.OPENAI_API_KEY
        )
        
        self.user_state = UserState()
        
        # Initialize specialized managers
        self.product_manager = ProductManager(self)
        self.recipe_manager = RecipeManager(self)
        self.parenting_manager = ParentingAdviceManager(self)
        self.health_manager = HealthConcernManager(self)
        self.qa_manager = GeneralQAManager(self)
        
        # Initialize category handlers mapping using CategoryEnum
        self.category_handlers = {
            CategoryEnum.get_short_name(CategoryEnum.PRODUCT.value): 
                self.product_manager.handle_product_recommendation,
            CategoryEnum.get_short_name(CategoryEnum.RECIPE.value): 
                self.recipe_manager.handle_recipe_recommendation,
            CategoryEnum.get_short_name(CategoryEnum.HEALTH.value): 
                self.health_manager.handle_health_concern,
            CategoryEnum.get_short_name(CategoryEnum.PARENTING.value): 
                self.parenting_manager.handle_parenting_advice
        }

        # Initialize category patterns using CategoryEnum
        self._initialize_category_patterns()

        # Initialize utilities
        self.utils = ConversationUtils(self)
        self.data_processor = DataProcessor()
        self.safety_checker = SafetyChecker()
        
        # Initialize conversation tracking
        self.conversation_buffer: List[Dict] = []
        setup_logging()

        # Add category detection patterns
        self._initialize_category_patterns()

    def _is_pagination_request(self, message: str) -> bool:
        """
        Enhanced pagination detection using ReACT framework instead of keyword matching.
        """
        transition_analysis = self._analyze_conversation_transition(message)
        
        # Use high confidence threshold for pagination decisions
        if transition_analysis["confidence"] >= 0.8:
            is_pagination = transition_analysis["transition_type"] == "pagination"
            logging.info(f"Pagination decision: {is_pagination}, Reasoning: {transition_analysis['reasoning']}")
            return is_pagination
            
        # Fall back to context analysis if confidence is low
        return self._analyze_pagination_context(message, transition_analysis)

    def _analyze_pagination_context(self, message: str, transition_analysis: Dict) -> bool:
        """
        Analyzes conversation context when ReACT confidence is low.
        Provides a safety net for pagination decisions.
        """
        # Check if we're in the middle of showing items
        if not self.user_state.current_category:
            return False
            
        # Check if we have more items to show
        if not self._has_more_items(self.user_state.current_category):
            return False
            
        # Use transition analysis signals even with low confidence
        context_signals = transition_analysis.get("context_signals", [])
        maintains_context = transition_analysis.get("maintains_context", False)
        
        return maintains_context and any(
            signal.startswith("continuation") for signal in context_signals
        )

    def _handle_pagination_request(self) -> str:
        """Handle request for more items with enhanced logging"""
        logging.info("Handling pagination request")
        
        current_category = self.user_state.current_category
        if not current_category:
            logging.info("No current category found for pagination")
            return "I'm not sure what type of items you'd like to see more of. Could you clarify?"
        
        # Get standardized short name for category
        category_short_name = CategoryEnum.get_short_name(current_category)
        
        # Get handler using standardized short name
        handler = self.category_handlers.get(category_short_name)
        if not handler:
            logging.error(f"No handler found for category: {category_short_name} (original: {current_category})")
            return "I'm having trouble finding more items. Could you rephrase your request?"
        
        try:
            logging.info(f"Requesting next batch from {category_short_name} handler")
            
            # Use appropriate pagination method based on category
            if category_short_name == "product":
                response = handler.get_next_items_batch(
                    self.user_state.last_message,
                    self.user_state.last_react_decision
                )
            elif category_short_name == "recipe":
                response = handler.get_next_recipe_batch(
                    self.user_state.last_message,
                    self.user_state.last_react_decision
                )
            elif category_short_name in ["health", "health_concern"]:
                response = handler.get_next_health_batch(
                    self.user_state.last_message,
                    self.user_state.last_react_decision
                )
            elif category_short_name in ["parenting", "parenting_advice"]:
                response = handler.get_next_advice_batch(
                    self.user_state.last_message,
                    self.user_state.last_react_decision
                )
            else:
                logging.error(f"Unsupported category for pagination: {category_short_name}")
                return "I'm not sure how to show more items for this topic. Could you try asking differently?"
            
            logging.info(f"Pagination response generated: {response[:100]}...")
            return response
                
        except Exception as e:
            logging.error(f"Error in pagination handling for {category_short_name}: {str(e)}")
            return "I encountered an error while getting more items. Could you try your request again?"

    def _analyze_conversation_transition(self, user_message: str) -> Dict:
        """
        Uses ReACT framework to analyze conversation transitions with complete category awareness.
        This helps the model understand available conversation paths and current context.
        """
        context = self.get_conversation_memory()
        recent_history = self.format_conversation_history()
        
        # Get all available categories in a structured format
        available_categories = {
            "parenting_advice": {
                "description": "Guidance and tips for parenting challenges and child development",
                "keywords": ["advice", "parenting", "development", "behavior", "growth"],
                "requires_age": True
            },
            "health_concern": {
                "description": "Health-related questions and medical guidance with appropriate disclaimers",
                "keywords": ["health", "medical", "symptoms", "doctor", "illness"],
                "requires_age": True
            },
            "product": {
                "description": "Baby and toddler product recommendations and information",
                "keywords": ["product", "buy", "recommendation", "item", "brand"],
                "requires_age": True
            },
            "recipe": {
                "description": "Age-appropriate food recipes and nutrition guidance",
                "keywords": ["recipe", "food", "meal", "cook", "nutrition"],
                "requires_age": True
            }
        }

        transition_prompt = f"""
        Task: Analyze conversation flow to determine user's next needs.

        Available Categories:
        {json.dumps(available_categories, indent=2)}

        Current Conversation State:
        - Active Category: {self.user_state.current_category or "None (New Conversation)"}
        - Previous Categories: {[msg.get('category') for msg in context['messages'][-3:] if msg.get('category')]}
        - Age Group: {self.user_state.age_group}

        Previous Context:
        {recent_history}

        Current Message: {user_message}

        Analyze the user's intent considering:
        1. Are they:
        - Requesting more information within {self.user_state.current_category} (pagination)
        - Transitioning to one of our other available categories
        - Providing requested information for the current category
        - Starting a new conversation thread
        2. What signals in their message and our category structure support this?
        3. How does this relate to the current category and available options?
        4. Should we maintain the current category or transition to a new one?

        Return structured JSON:
        {{
            "transition_type": "pagination/category_change/information_provision/new_thread",
            "confidence": 0.0-1.0,
            "current_category": "{self.user_state.current_category or 'none'}",
            "target_category": "current or new category name from available categories",
            "maintains_context": boolean,
            "category_signals": {{
                "matched_keywords": ["keyword1", "keyword2"],
                "category_confidence": 0.0-1.0,
                "requires_age_check": boolean
            }},
            "reasoning": "detailed explanation",
            "suggested_action": "specific next step",
            "context_signals": ["relevant signal 1", "relevant signal 2"]
        }}
        """

        try:
            response = self.get_gpt_response(transition_prompt)
            analysis = json.loads(self._clean_json_response(response))
            
            # Log detailed analysis for debugging
            logging.info(f"Transition analysis:\n{json.dumps(analysis, indent=2)}")
            
            # Verify category validity
            if analysis["target_category"] not in available_categories and \
            analysis["target_category"] != analysis["current_category"]:
                logging.warning(f"Invalid target category detected: {analysis['target_category']}")
                analysis["confidence"] *= 0.5  # Reduce confidence for invalid categories
                
            return analysis
        except Exception as e:
            logging.error(f"Error in transition analysis: {str(e)}")
            return self._get_default_transition_analysis()

    def _handle_category_transition(self, user_message: str, transition_analysis: Dict) -> str:
        """
        Handles transitions between categories with full category awareness.
        """
        new_category = transition_analysis["target_category"]
        category_signals = transition_analysis.get("category_signals", {})
        
        logging.info(f"Handling category transition to: {new_category}")
        logging.info(f"Category signals: {json.dumps(category_signals, indent=2)}")
        
        # Check if we need age information for the new category
        if category_signals.get("requires_age_check", False) and not self.user_state.age_group:
            return "Could you please tell me your child's age? This helps me provide age-appropriate recommendations."
        
        # Save current category context before switching
        if self.user_state.current_category:
            self.save_category_summary(self.user_state.current_category)
        
        # Reset pagination state for new category
        self.user_state.reset_pagination_state()
        
        # Update category and route message
        self.user_state.current_category = new_category
        return self.route_to_appropriate_manager(
            category=new_category,
            user_message=user_message,
            age_group=self.user_state.age_group,
            requery=False
        )

    def _handle_pagination_with_context(self, user_message: str, transition_analysis: Dict) -> str:
        """
        Enhanced pagination handling that considers conversation context and handles
        multiple category name variations.
        """
        logging.info("Handling pagination with context")
        
        current_category = self.user_state.current_category
        if not current_category:
            return "I'm not sure what type of information you'd like to see more of. Could you clarify?"
        
        # Get standardized short name for category
        category_short_name = CategoryEnum.get_short_name(current_category)
        
        try:
            # This is the problematic section - completely rework it
            if category_short_name == "recipe":
                # Use the instance method directly
                react_decision = {
                    "decision": "pagination", 
                    "confidence": 0.9,
                    "dietary_considerations": [],
                    "preparation_preferences": {
                        "time": "medium", 
                        "complexity": "easy"
                    }
                }
                # Store this for future use
                self.user_state.last_react_decision = react_decision
                
                # Directly call the method on the instance
                return self.recipe_manager.get_next_recipe_batch(user_message, react_decision)
                
            elif category_short_name == "product":
                react_decision = {"decision": "pagination", "confidence": 0.9}
                self.user_state.last_react_decision = react_decision
                return self.product_manager.get_next_items_batch(user_message, react_decision)
                
            elif category_short_name in ["health", "health_concern"]:
                react_decision = {"decision": "pagination", "confidence": 0.9}
                self.user_state.last_react_decision = react_decision
                return self.health_manager.get_next_health_batch(user_message, react_decision)
                
            elif category_short_name in ["parenting", "parenting_advice"]:
                react_decision = {"decision": "pagination", "confidence": 0.9}
                self.user_state.last_react_decision = react_decision
                return self.parenting_manager.get_next_advice_batch(user_message, react_decision)
                
            else:
                logging.error(f"Unsupported category for pagination: {current_category} (short: {category_short_name})")
                return "I'm not sure how to show more information for this topic. Could you try asking in a different way?"
                    
        except Exception as e:
            logging.error(
                f"Error in pagination handling for {category_short_name} "
                f"(original: {current_category}): {str(e)}",
                exc_info=True
            )
            return "I encountered an error while getting more information. Could you try your request again?"

    def determine_age_group(self, age_input: str) -> Optional[str]:
        """Convert various age formats to standardized age groups"""
        try:
            # Remove any whitespace and convert to lowercase
            age_input = age_input.lower().strip()
            
            # Extract numbers and units using regex
            numbers = re.findall(r'\d+(?:\.\d+)?', age_input)
            if not numbers:
                return None
                
            number = float(numbers[0])
            
            # Convert everything to months for comparison
            if 'year' in age_input:
                months = number * 12
            else:
                months = number
                
            # Determine age group
            if 0 <= months <= 6:
                return "0-6 months"
            elif 6 < months < 12:
                return "6-12 months"
            elif 12 <= months <= 18:
                return "12-18 months"
            elif 18 < months <= 24:
                return "18-24 months"
            else:
                return None
                
        except Exception as e:
            logging.error(f"Error determining age group: {e}")
            return None

    def get_conversation_context(self) -> Dict:
        """Get current conversation context including last assistant message"""
        # Initialize default context
        default_context = {
            "current_category": None,
            "last_assistant_msg": None,
            "current_thread": [],
            "age_group": self.user_state.age_group if hasattr(self.user_state, 'age_group') else None,
            "previous_categories": [],
            "last_query": None
        }
        
        if not self.conversation_buffer:
            return default_context
        
        recent_messages = self.conversation_buffer[-3:]  # Get last 3 messages
        
        return {
        "current_category": recent_messages[-1].get("category") if recent_messages else None,
        "last_assistant_msg": recent_messages[-1].get("response") if recent_messages else None,
        "current_thread": recent_messages,
        "age_group": self.user_state.age_group if hasattr(self.user_state, 'age_group') else None,
        "previous_categories": [msg.get("category") for msg in recent_messages],
        "last_query": recent_messages[-1].get("user_message") if recent_messages else None
    }

    def _initialize_category_patterns(self):
        """Initialize patterns for category detection"""
        self.category_patterns = {
            CategoryEnum.RECIPE.value: {
                "keywords": ["recipe", "cook", "food", "meal", "eat", "feed", "nutrition",
                           "breakfast", "lunch", "dinner", "snack"],
                "patterns": [
                    r"recipe[s]?",
                    r"(how|what) to (cook|make|prepare)",
                    r"food[s]? for",
                    r"meal[s]? for"
                ]
            },
            CategoryEnum.PRODUCT.value: {
                "keywords": ["product", "buy", "purchase", "recommend", "brand", "item"],
                "patterns": [
                    r"where (can|to) (buy|get|find)",
                    r"which product[s]?",
                    r"need (to buy|a product)",
                    r"looking for (a|some) product[s]?"
                ]
            },
            CategoryEnum.HEALTH.value: {
                "keywords": ["health", "sick", "doctor", "fever", "symptom", "medicine"],
                "patterns": [
                    r"(is|are) (it|they|he|she) sick",
                    r"health (issue|problem|concern)",
                    r"should I (see|consult) (a|the) doctor",
                    r"(have|has) (a|some) symptom[s]?"
                ]
            },
            CategoryEnum.PARENTING.value: {
                "keywords": ["advice", "help", "guide", "handle", "deal", "parent"],
                "patterns": [
                    r"how (to|do I|should I) (handle|deal with|manage)",
                    r"parenting (advice|help|guidance)",
                    r"what (to|should) do (about|when)",
                    r"advice (for|about|on)"
                ]
            }
        }

    def detect_category(self, user_message: str, context: Dict) -> Optional[str]:
        """
        Detect message category with support for both full and short names.
        Returns the full category name for consistency.
        """
        message_lower = user_message.lower()
        logging.info(f"Detecting category for message: {message_lower}")
        
        # First check explicit category in content
        detected_category = self._detect_category_from_content(message_lower)
        if detected_category:
            logging.info(f"Detected category: {detected_category}")
            return detected_category
            
        # Check if maintaining current category for follow-ups/age responses
        if context["current_category"]:
            current_category = context["current_category"]
            # Ensure we're working with full category names
            if CategoryEnum.get_short_name(current_category):
                # Already a full name
                if self._is_followup_message(message_lower):
                    logging.info(f"Maintaining current category: {current_category}")
                    return current_category
            else:
                # Convert short name to full name if needed
                full_name = CategoryEnum.get_full_name(current_category)
                if full_name and self._is_followup_message(message_lower):
                    return full_name
        
        return None

    def _is_followup_message(self, message: str) -> bool:
        """Check if message is a follow-up to previous query"""
        followup_indicators = [
            "yes", "yeah", "right", "correct", "no", "nope",
            "what about", "and", "also", "too", "ok", "okay"
        ]
        return any(indicator in message for indicator in followup_indicators)

    def _is_age_response(self, message: str) -> bool:
        """Check if message is providing age information"""
        age_patterns = [
            r"\d+\s*(month|year)",
            r"(one|two|three|four|five|six|seven|eight|nine|ten)\s*(month|year)",
            r"(is|are|about|around)\s*\d+",
            r"(almost|nearly|just)\s*\d+"
        ]
        return any(re.search(pattern, message) for pattern in age_patterns)

    def _detect_category_from_content(self, message: str) -> Optional[str]:
        """Detect category based on message content and patterns"""
        category_scores = {category: 0 for category in self.category_patterns.keys()}
        
        for category, patterns in self.category_patterns.items():
            # Check keywords
            keyword_matches = sum(keyword in message for keyword in patterns["keywords"])
            category_scores[category] += keyword_matches * 1.0
            
            # Check regex patterns
            pattern_matches = sum(bool(re.search(pattern, message)) for pattern in patterns["patterns"])
            category_scores[category] += pattern_matches * 2.0

        # Get category with highest score if above threshold
        max_score = max(category_scores.values())
        if max_score >= 1.0:
            return max(category_scores.items(), key=lambda x: x[1])[0]
            
        return None

    def _is_providing_requested_info(self, user_message: str, last_assistant_msg: str) -> bool:
        """Check if user is providing previously requested information"""
        if not last_assistant_msg:
            return False
            
        # Common request patterns
        info_requests = [
            # Age-related requests
            (r"how old|what age|(\d+)\s*(month|year)",
            ["age", "months", "years", "old"]),
            
            # Recipe/feeding related
            (r"what (do you|would you like) to (cook|feed|make)",
            ["recipe", "food", "cook", "meal"]),
            
            # General clarification
            (r"could you (clarify|explain|tell me more)",
            ["yes", "no", "okay", "sure", "well"])
        ]
        
        for pattern, response_indicators in info_requests:
            if re.search(pattern, last_assistant_msg.lower()):
                return any(indicator in user_message.lower() 
                        for indicator in response_indicators)
        
        return False

    def format_conversation_history(self) -> str:
        """Format recent conversation history for context"""
        if not self.conversation_buffer:
            return "No previous conversation"
            
        history = []
        for msg in self.conversation_buffer[-3:]:
            history.extend([
                f"User: {msg['user_message']}",
                f"Assistant: {msg['response']}",
                f"Context: {msg.get('category', 'Unknown category')}"
            ])
        
        return "\n".join(history)

    def _get_default_transition_analysis(self) -> Dict:
        """Provide safe defaults if transition analysis fails"""
        return {
            "decision_type": "unclear",
            "confidence": 0.0,
            "target_category": self.user_state.current_category,
            "reasoning": "Failed to analyze transition",
            "suggested_action": "process_normally",
            "context_signals": []
        }

    def _clean_json_response(self, response: str) -> str:
        """Cleans and validates JSON response from GPT"""
        try:
            # Remove any markdown formatting
            response = response.replace('```json\n', '').replace('```', '')
            # Remove any leading/trailing whitespace
            response = response.strip()
            
            # If response starts with a letter that's not {, try to find the first {
            if response and not response.startswith('{'):
                start_idx = response.find('{')
                if start_idx != -1:
                    response = response[start_idx:]
                    
            # If response ends with something after }, try to find the last }
            if response and not response.endswith('}'):
                end_idx = response.rfind('}')
                if end_idx != -1:
                    response = response[:end_idx+1]
                    
            return response
        except Exception as e:
            logging.error(f"Error cleaning JSON response: {str(e)}")
            raise

    def handle_conversation(self, user_message: str) -> str:
        try:
            # Initialize conversation turn with detailed logging
            logging.info("\n=== New Conversation Turn ===")
            logging.info(f"User message: {user_message}")
            
            # Store message in state for context
            self.user_state.last_message = user_message
            
            # Gather current context and conversation memory
            context = self.get_conversation_context() or {}  # Ensure we have a dict
            memory = self.get_conversation_memory()
            logging.info(f"Current context: {context}")

            # Validate input before processing
            logging.info("Validating input...")
            is_valid, processed_message, validation_notes = self.validate_input(user_message)
            if not is_valid:
                logging.info(f"Input validation failed: {validation_notes}")
                return validation_notes
            
            # THIS IS A KEY FIX - Check if there's a conversation buffer but context is empty
            if self.conversation_buffer and not context.get('current_category'):
                # Recover lost category from the most recent conversation entry
                last_entry = self.conversation_buffer[-1]
                if last_entry.get('category'):
                    logging.info(f"Recovering lost category from conversation buffer: {last_entry.get('category')}")
                    self.user_state.current_category = last_entry.get('category')
                    context['current_category'] = last_entry.get('category')
                    
            # Process age information if present (MOVE THIS EARLIER)
            age_group = self.determine_age_group(user_message)
            if age_group:
                self.user_state.age_group = age_group
                logging.info(f"Updated age group to: {age_group}")        

            # Perform transition analysis to understand user intent
            logging.info("\n=== Starting Transition Analysis ===")
            transition_analysis = self._analyze_conversation_transition(user_message)
            logging.info(f"Transition Analysis:\n{json.dumps(transition_analysis, indent=2)}")

            # Handle high-confidence transitions with null checks
            if transition_analysis and transition_analysis.get("confidence", 0) >= self.config.REACT_CONFIDENCE_THRESHOLD:
                if transition_analysis.get("transition_type") == "pagination":
                    # Handle pagination with context awareness
                    logging.info("Handling pagination based on transition analysis")
                    pagination_response = self._handle_pagination_with_context(
                        user_message, 
                        transition_analysis
                    )
                    
                    # Update conversation buffer for pagination
                    self.update_conversation_buffer(
                        user_message=user_message,
                        response=pagination_response,
                        category=context.get("current_category"),
                        context={
                            "is_pagination": True,
                            "transition_analysis": transition_analysis
                        }
                    )
                    
                    return pagination_response

                elif transition_analysis.get("transition_type") == "category_change":
                    # Handle category transition
                    logging.info(f"Handling category transition to: {transition_analysis.get('target_category')}")
                    return self._handle_category_transition(user_message, transition_analysis)

            # Process age information if present
            #age_group = self.determine_age_group(user_message)
            #if age_group:
                #self.user_state.age_group = age_group
                #logging.info(f"Updated age group to: {age_group}")
            
            # Use stored age group for handlers
            #current_age_group = self.user_state.age_group
            # Gather current context and conversation memory - AFTER age update
            context = self.get_conversation_context() or {}
            memory = self.get_conversation_memory()
            logging.info(f"Current context: {context}")

            # Perform main ReACT analysis for content understanding
            logging.info("\n=== Starting Content ReACT Analysis ===")
            react_analysis = self.get_gpt_response(self.get_react_prompt(processed_message, memory))
            logging.info(f"Content Analysis from GPT:\n{react_analysis}")
            
            # Store analysis for future context
            self.user_state.last_react_decision = react_analysis
            
            # Determine appropriate category
            detected_category = transition_analysis.get("target_category") or \
                            self.detect_category(processed_message, context)
            
            if not detected_category:
                # Provide helpful guidance on available categories
                return ("I'd be happy to help. What would you like assistance with? "
                    "I can help with recipes, products, parenting advice, or health concerns.")

            # Determine final category based on context and analysis
            if context["current_category"] and transition_analysis.get("maintains_context", False):
                category = context["current_category"]
            else:
                category = detected_category
            
            logging.info(f"Final category determination: {category}")
            
            # Update state with category information
            self.user_state.update_category_state(category)
            
            # Process age context for multi-child scenarios
            age_context = self.handle_age_references(user_message, context)
            current_age_group = age_context.get("current_age_group")

            # Route to appropriate specialized manager
            logging.info(f"\n=== Routing to {category} Manager ===")
            response = self.route_to_appropriate_manager(
                category=category,
                user_message=processed_message,
                age_group=self.user_state.age_group,
                requery=False
            )

            # Create comprehensive context for conversation buffer
            buffer_context = {
                "age_group": current_age_group,
                "previous_category": context.get("current_category"),
                "is_followup": transition_analysis.get("maintains_context", False),
                "age_context": age_context,
                "transition_analysis": transition_analysis
            }
            
            # Update conversation history
            self.update_conversation_buffer(
                user_message=processed_message,
                response=response,
                category=category,
                context=buffer_context
            )
            
            logging.info(f"\nFinal response:\n{response}")
            return response
            
        except Exception as e:
            logging.error(f"Error in conversation handling: {str(e)}", exc_info=True)
            return self.generate_safe_fallback_response()

    def _clean_json_response(self, response: str) -> str:
        """Cleans and validates JSON response from GPT"""
        try:
            # Remove any markdown formatting
            response = response.replace('```json\n', '').replace('```', '')
            # Remove any leading/trailing whitespace
            response = response.strip()
            
            # If response starts with a letter that's not {, try to find the first {
            if response and not response.startswith('{'):
                start_idx = response.find('{')
                if start_idx != -1:
                    response = response[start_idx:]
                    
            # If response ends with something after }, try to find the last }
            if response and not response.endswith('}'):
                end_idx = response.rfind('}')
                if end_idx != -1:
                    response = response[:end_idx+1]
                    
            return response
        except Exception as e:
            logging.error(f"Error cleaning JSON response: {str(e)}")
            raise

    def handle_age_references(self, user_message: str, context: Dict) -> Dict:
        """
        Analyzes the message for age references and maintains age history.
        Determines if user is:
        1. Mentioning a new age
        2. Referring to a previously mentioned age
        3. Implicitly using the last mentioned age
        
        Returns a dictionary with complete age context information.
        """
        age_context = {
            "current_age_group": None,
            "is_new_age": False,
            "previous_ages": self.user_state.age_history.get("previous", []),
            "reference_type": None
        }
        
        # First check for explicit new age mention
        new_age = self.determine_age_group(user_message)
        if new_age:
            # Store current age in history before updating
            if self.user_state.age_group:
                self.user_state.age_history["previous"].append({
                    "age": self.user_state.age_group,
                    "timestamp": time.time(),
                    "context": self.get_age_mention_context()
                })
            
            # Update current age
            self.user_state.age_group = new_age
            age_context.update({
                "current_age_group": new_age,
                "is_new_age": True,
                "reference_type": "new_mention"
            })
            
            logging.info(f"New age mentioned: {new_age}")
            return age_context
        
        # Check for references to previous ages
        previous_age_reference = self.detect_previous_age_reference(user_message, context)
        if previous_age_reference:
            referenced_age = previous_age_reference["age"]
            age_context.update({
                "current_age_group": referenced_age,
                "is_new_age": False,
                "reference_type": "previous_reference",
                "referenced_context": previous_age_reference["context"]
            })
            
            logging.info(f"Referenced previous age: {referenced_age}")
            return age_context
        
        # If no new or referenced age, use current age
        if self.user_state.age_group:
            age_context.update({
                "current_age_group": self.user_state.age_group,
                "is_new_age": False,
                "reference_type": "current_continuation"
            })
            
            logging.info(f"Continuing with current age: {self.user_state.age_group}")
        
        return age_context

    def detect_previous_age_reference(self, user_message: str, context: Dict) -> Optional[Dict]:
        """
        Detects if the user is referring to a previously mentioned age.
        Uses context clues and reference patterns to identify which age is being referenced.
        """
        # Common patterns for referring to previous ages
        reference_patterns = [
            r"first child",
            r"older child",
            r"younger child",
            r"other child",
            r"previous age",
            r"earlier age"
        ]
        
        message_lower = user_message.lower()
        
        # Check for reference patterns
        for pattern in reference_patterns:
            if re.search(pattern, message_lower):
                # Found a reference, now determine which age
                previous_ages = self.user_state.age_history.get("previous", [])
                if previous_ages:
                    # Here you can implement more sophisticated logic to determine
                    # which previous age is being referenced based on the pattern
                    # and conversation context
                    referenced_age = previous_ages[-1]  # Most recent by default
                    return referenced_age
        
        return None

    def get_age_mention_context(self) -> Dict:
        """
        Captures the context in which an age was mentioned.
        This helps with later references to this age.
        """
        return {
            "category": self.user_state.current_category,
            "last_message": self.user_state.last_message,
            "conversation_turn": len(self.conversation_buffer)
        }

    def _count_category_changes(self) -> int:
        """Count number of category changes in conversation"""
        if len(self.conversation_buffer) < 2:
            return 0
            
        changes = 0
        previous_category = None
        
        for msg in self.conversation_buffer:
            current_category = msg.get("category")
            if previous_category and current_category != previous_category:
                changes += 1
            previous_category = current_category
        
        return changes

    def _calculate_user_engagement(self) -> float:
        """Calculate user engagement score"""
        if not self.conversation_buffer:
            return 0.0
            
        # Simple engagement score based on conversation length
        return min(1.0, len(self.conversation_buffer) / 10)

    def format_conversation_history(self) -> str:
        """Format recent conversation history for context"""
        if not self.conversation_buffer:
            return "No previous conversation"
            
        recent_messages = self.conversation_buffer[-3:]  # Get last 3 messages
        formatted_history = []
        
        for msg in recent_messages:
            formatted_history.extend([
                f"User: {msg['user_message']}",
                f"Assistant: {msg['response']}"
            ])
        
        return "\n".join(formatted_history)

    def handle_age_and_category_changes(self, user_message: str) -> Dict:
        """
        Handles age and category changes while maintaining history and context.
        Returns information about what changed.
        """
        changes = {
            "age_changed": False,
            "category_changed": False,
            "previous_age": self.user_state.age_history["current"],
            "previous_category": self.user_state.category_history["current"]
        }
        
        # Check for new age mention
        new_age = self.determine_age_group(user_message)
        if new_age and new_age != self.user_state.age_history["current"]:
            # Store current age in history before updating
            if self.user_state.age_history["current"]:
                self.user_state.age_history["previous"].append({
                    "age": self.user_state.age_history["current"],
                    "timestamp": time.time()
                })
            
            self.user_state.age_history["current"] = new_age
            changes["age_changed"] = True
            logging.info(f"Age changed to: {new_age}")

        # Check for category change
        new_category = self.detect_category(user_message, self.get_conversation_context())
        if new_category and new_category != self.user_state.category_history["current"]:
            # Before changing category, save summary of current discussion
            if self.user_state.category_history["current"]:
                self.save_category_summary(self.user_state.category_history["current"])
                
            # Store current category in history
            self.user_state.category_history["previous"].append({
                "category": self.user_state.category_history["current"],
                "timestamp": time.time()
            })
            
            self.user_state.category_history["current"] = new_category
            changes["category_changed"] = True
            logging.info(f"Category changed to: {new_category}")
        
        return changes

    def save_category_summary(self, category: str) -> None:
        """
        Creates and stores a summary of the discussion for a given category
        before switching to a new topic.
        """
        # Get relevant messages for this category
        category_messages = [
            msg for msg in self.conversation_buffer 
            if msg.get("category") == category
        ]
        
        if not category_messages:
            return
            
        # Generate summary using GPT
        summary_prompt = f"""
        Summarize the discussion about {category}:
        
        Messages:
        {json.dumps([{
            'user': msg['user_message'],
            'response': msg['response']
        } for msg in category_messages], indent=2)}
        
        Provide a concise summary including:
        1. Main topics discussed
        2. Key recommendations or information provided
        3. Any pending questions or unresolved points
        """
        
        try:
            summary = self.get_gpt_response(summary_prompt)
            self.user_state.topic_summaries[category] = {
                "summary": summary,
                "timestamp": time.time(),
                "message_count": len(category_messages)
            }
            logging.info(f"Saved summary for category: {category}")
        except Exception as e:
            logging.error(f"Error generating category summary: {str(e)}")

    def handle_previous_discussion_reference(self, user_message: str) -> str:
        """
        Handles references to previous discussions and provides relevant context.
        """
        reference_indicators = [
            "earlier", "before", "previous", "last time", "we talked about",
            "what about", "going back to", "remember when"
        ]
        
        if any(indicator in user_message.lower() for indicator in reference_indicators):
            previous_categories = self.user_state.category_history["previous"]
            
            if not previous_categories:
                return "I don't have any previous discussions to refer to. Would you like to start a new conversation?"
            
            # Generate response about previous discussions
            response_parts = ["Let me summarize what we discussed earlier:\n"]
            
            for prev_cat in previous_categories[-3:]:  # Show last 3 categories
                category = prev_cat["category"]
                if category in self.user_state.topic_summaries:
                    summary = self.user_state.topic_summaries[category]
                    response_parts.append(f"\nRegarding {category}:\n{summary['summary']}")
            
            response_parts.append("\nWhich of these topics would you like to discuss further?")
            return "\n".join(response_parts)
        
        return None

    def restore_previous_context(self, category: str) -> None:
        """
        Restores context when returning to a previously discussed category.
        """
        if category in self.user_state.topic_summaries:
            previous_summary = self.user_state.topic_summaries[category]
            self.user_state.category_history["current"] = category
            
            # Update conversation context
            self.update_conversation_buffer(
                user_message="Returning to previous discussion",
                response=f"Continuing our discussion about {category}. Previously we discussed: {previous_summary['summary']}",
                category=category,
                context={
                    "is_continuation": True,
                    "previous_summary": previous_summary
                }
            )

    def _has_more_items(self, category: str) -> bool:
        """Check if more items are available for the given category"""
        try:
            all_items = self.user_state.get_all_available_items(category)
            shown_items = self.user_state.get_shown_items(category)
            current_page = self.user_state.get_current_page(category)
            
            available_items = [item for item in all_items if str(item['id']) not in shown_items]
            start_idx = current_page * 3
            
            return len(available_items) > start_idx + 3
            
        except Exception as e:
            logging.error(f"Error checking for more items: {str(e)}")
            return False

    def generate_safe_fallback_response(self) -> str:
        """Generate a safe fallback response when errors occur"""
        return ("I apologize, but I'm having trouble processing your request at the moment. "
                "Could you please rephrase your question? I'm here to help with baby products, "
                "recipes, parenting advice, and related topics.")

    def handle_multi_intent_message(self, intents: List[Dict], user_message: str) -> str:
        """Handle messages containing multiple intents"""
        responses = []
        memory = self.get_conversation_memory()
        for intent in intents:
            try:
                # Route to appropriate handler based on intent
                handler = self.get_handler_for_intent(intent)
                if handler:
                    response = handler(
                        user_message=user_message,
                        age_group=self.user_state.age_group,
                        memory=memory,
                        requery=False
                    )
                    responses.append(response)
                    
            except Exception as e:
                logging.error(f"Error handling intent {intent}: {str(e)}")
        
        # Combine responses naturally
        return self.combine_multi_intent_responses(responses)

    def get_handler_for_intent(self, intent: Dict) -> Optional[callable]:
        """Get appropriate handler based on intent type"""
        intent_handlers = {
            "product": self.product_manager.handle_product_recommendation,
            "recipe": self.recipe_manager.handle_recipe_recommendation,
            "parenting_advice": self.parenting_manager.handle_parenting_advice,
            "health_concern": self.health_manager.handle_health_concern,
            "general": self.qa_manager.handle_general_qa
        }
        
        return intent_handlers.get(intent.get("category"))

    def route_to_appropriate_manager(self, category: str, user_message: str, age_group: str, requery: bool = False) -> str:
        """
        Enhanced routing that properly handles the two-step search process across all specialized managers.
        This function serves as the central routing mechanism for all user queries, ensuring they are
        directed to the appropriate specialized handler while maintaining proper state management.

        Args:
            category: The detected category of the user's request (full category name)
            user_message: The user's original message
            age_group: The age group context for the request (e.g., "0-6 months")
            requery: Boolean flag indicating if this is a repeated search attempt

        Returns:
            str: Formatted response from the appropriate handler with any necessary disclaimers
        """
        try:
            # First, verify we have required age information for age-sensitive categories
            if not age_group and CategoryEnum.is_valid_category(category):
                # We only require age information for certain categories
                return ("Could you please tell me your child's age? "
                    "This helps me provide age-appropriate recommendations.")

            # Convert category to short name for handler lookup if it's a full name
            short_category = CategoryEnum.get_short_name(category) or category
            
            # Initialize our handler mapping with built-in type checking
            handlers = {
                CategoryEnum.PRODUCT.value: self.product_manager.handle_product_recommendation,
                CategoryEnum.RECIPE.value: self.recipe_manager.handle_recipe_recommendation,
                CategoryEnum.PARENTING.value: self.parenting_manager.handle_parenting_advice,
                CategoryEnum.HEALTH.value: self.health_manager.handle_health_concern
            }

            # Get the appropriate handler for this category
            logging.info(f"Routing to {category} manager...")
            handler = handlers.get(category)
            if not handler:
                # If we can't find a handler with the full name, try the short name
                handler = self.category_handlers.get(short_category)
                
            if not handler:
                # If we still can't find a handler, log the error and return a friendly message
                logging.error(f"No handler found for category: {category} (short: {short_category})")
                return ("I apologize, but I'm not sure how to handle that request. "
                    "Could you please rephrase it?")

            # Check if this is a continuation of a previous search session
            is_continuation = (
                category == self.user_state.current_category and
                not requery and
                self.user_state.get_all_available_items(category)
            )

            # If this is a continuation, we might want to handle pagination differently
            if is_continuation:
                logging.info(f"Continuing previous {category} conversation")
            else:
                logging.info(f"Starting new {category} conversation")

            # Get the response from the appropriate handler
            response = handler(user_message, age_group, requery)

            # Update the conversation state with the new interaction
            self.update_conversation_buffer(
                user_message=user_message,
                response=response,
                category=category,  # Store the full category name
                context={
                    "age_group": age_group,
                    "previous_category": category,
                    "is_followup": True,
                    "is_continuation": is_continuation,
                    "handler_type": short_category  # Store the handler type for future reference
                }
            )

            # Update the user state with the current category
            self.user_state.current_category = category
            
            # If this was a new search (not a continuation), store the search parameters
            if not is_continuation:
                self.store_search_parameters(
                    category=category,
                    age_group=age_group,
                    query=user_message
                )

            return response

        except Exception as e:
            # Log the full error details for debugging
            logging.error(f"Error in routing to {category} handler: {str(e)}", exc_info=True)
            
            # Handle the error appropriately based on the category
            return self.handle_manager_error(category, e)

    def store_search_parameters(self, category: str, age_group: str, query: str) -> None:
        """
        Stores the search parameters for the current interaction, allowing for
        better context management and search continuation.
        
        Args:
            category: The category of the search (e.g., "Recipe Recommendation")
            age_group: The age group for the search (e.g., "12-18 months")
            query: The original user query
            
        This method maintains a history of search parameters per category and
        helps with pagination and context restoration.
        """
        # Create a search context entry
        search_context = {
            "timestamp": time.time(),
            "category": category,
            "age_group": age_group,
            "query": query,
            "search_parameters": {
                "page": self.user_state.get_current_page(category),
                "shown_items": list(self.user_state.get_shown_items(category)),
                "available_items": self.user_state.get_all_available_items(category)
            }
        }
        
        # Initialize search history if it doesn't exist
        if not hasattr(self, 'search_history'):
            self.search_history = {}
        
        # Store the search parameters in category-specific history
        if category not in self.search_history:
            self.search_history[category] = []
        
        # Add the new search context
        self.search_history[category].append(search_context)
        
        # Keep only the last 5 searches per category
        if len(self.search_history[category]) > 5:
            self.search_history[category] = self.search_history[category][-5:]
        
        # Update current search parameters
        self.current_search_parameters = search_context

    def get_last_search_parameters(self, category: str) -> dict:
        """
        Retrieves the most recent search parameters for a given category.
        
        Args:
            category: The category to get parameters for
            
        Returns:
            dict: The most recent search parameters or None if no history exists
        """
        if not hasattr(self, 'search_history'):
            return None
            
        category_history = self.search_history.get(category, [])
        return category_history[-1] if category_history else None

    def clear_search_parameters(self, category: str = None) -> None:
        """
        Clears search parameters for a specific category or all categories.
        
        Args:
            category: Optional category to clear. If None, clears all search history.
        """
        if not hasattr(self, 'search_history'):
            return
            
        if category:
            self.search_history[category] = []
        else:
            self.search_history = {}

    def assess_context_relevance(self) -> float:
        """Simple context relevance assessment"""
        return 1.0  # Default value, can be made more sophisticated later

    def get_category_confidence(self) -> float:
        """Get confidence score for current category"""
        return 0.8  # Default confidence for now, can be made more sophisticated later

    def handle_manager_error(self, category: str, error: Exception) -> str:
        """Handle errors from specialized managers"""
        logging.error(f"Error in {category} handler: {str(error)}")
        return f"I apologize, but I need some more information. Could you please tell me your child's age? This will help me provide age-appropriate {category.lower()}."

    def get_gpt_response(self, prompt: str, retries: int = 3) -> str:
        """Simplified GPT response handling"""
        from openai import OpenAI  # Import the client

        client = OpenAI(api_key=self.config.OPENAI_API_KEY)
        
        for attempt in range(retries):
            try:
                logging.info(f"Sending prompt to GPT:\n{prompt}\n")
                
                response = client.chat.completions.create(
                    model=self.config.GPT_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a Nestlé baby care expert trained to help with baby products, recipes, parenting advice, health concerns, and related topics."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                gpt_response = response.choices[0].message.content
                logging.info(f"Received response:\n{gpt_response}\n")
                
                return gpt_response
                
            except Exception as e:
                logging.error(f"GPT response error (attempt {attempt + 1}): {str(e)}")
                if attempt == retries - 1:
                    raise ConversationError(
                        "Failed to get valid GPT response",
                        "gpt_error",
                        {"prompt": prompt, "attempts": attempt + 1}
                    )
                time.sleep(2 ** attempt)

        raise ConversationError(
            "Max retries exceeded for GPT response",
            "max_retries_exceeded",
            {"prompt": prompt}
        )

    def validate_gpt_response(self, response: Dict) -> bool:
        """Validates GPT response for format, content, and safety"""
        try:
            if not response or 'choices' not in response:
                return False
                
            content = response['choices'][0]['message']['content']
            
            # Check for empty or too short responses
            if not content or len(content.strip()) < 10:
                return False
                
            # Check for JSON format if expected
            if self._expects_json_response(content):
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    return False
                    
            # Perform safety check
            safety_result = self.safety_checker.check_content_safety(content)
            if not safety_result['is_safe']:
                logging.warning(f"Safety check failed: {safety_result['concerns']}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error in GPT response validation: {str(e)}")
            return False

    def update_conversation_buffer(self, user_message: str, response: str, category: str, context: Dict) -> None:
        """Update buffer with full context and handle missing fields gracefully"""
        entry = {
            "user_message": user_message,
            "response": response,
            "timestamp": time.time(),
            "category": category,
            "age_group": context.get("age_group", self.user_state.age_group),  # Fallback to user_state
            "metadata": {
                "previous_category": context.get("previous_category"),
                "is_followup": context.get("is_followup", False),
                "category_confidence": self.get_category_confidence()
            }
        }
        
        self.conversation_buffer.append(entry)
        self.user_state.add_conversation_context(entry)

    def update_state(self, age_group: str) -> None:
        """
        Enhanced state management that properly handles search context
        across category changes.
        """
        if age_group != self.user_state.age_group:
            # Store current state before updating
            self.save_category_summary(self.user_state.current_category)
            
            # Update age group
            self.user_state.age_group = age_group
            
            # Reset search-related state
            self.user_state.reset_search_state()
            
            # Clear existing items since they're for wrong age group
            self.user_state.reset_shown_items()
            
            # Reset category since we need fresh search for new age
            self.user_state.current_category = None
            
            return True
        return False

    def handle_state_changes(self, user_message: str, current_context: Dict) -> Dict[str, Any]:
        changes = {
            "category_changed": False,
            "age_changed": False,
            "new_category": None,
            "new_age_group": None,
            "requires_requery": False
        }
        
        # Force category update on content match
        detected_category = self._detect_category_from_content(user_message.lower())
        if detected_category:
            changes["category_changed"] = True  
            changes["new_category"] = detected_category
            changes["requires_requery"] = True
            self.user_state.current_category = detected_category
        
        # Update age if found
        new_age = self.determine_age_group(user_message)
        if new_age and new_age != self.user_state.age_group:
            changes["age_changed"] = True
            changes["new_age_group"] = new_age
            self.user_state.age_group = new_age
            changes["requires_requery"] = True

        return changes

    def _confirm_category_change(self, user_message: str, current_category: str, new_category: str) -> bool:
        """Verify if user is really changing topics or just providing requested info"""
        # Don't consider it a category change if user is just providing requested info
        if self._is_providing_requested_info(user_message, self.get_last_assistant_message()):
            return False
            
        # Check if they're explicitly changing topics
        topic_change_indicators = [
            "instead", "different", "another", "change", "rather",
            "also need", "also want", "something else"
        ]
    
        return any(indicator in user_message.lower() for indicator in topic_change_indicators)

    def smart_summarize_conversation(self) -> None:
        """Intelligently summarize conversation buffer"""
        if len(self.conversation_buffer) <= self.config.MAX_CONVERSATION_BUFFER:
            return
            
        important_messages = []
        
        # Always keep last 3 messages
        important_messages.extend(self.conversation_buffer[-3:])
        
        # Keep messages with high engagement
        for msg in self.conversation_buffer[:-3]:
            if self._calculate_message_importance(msg) > 0.7:
                important_messages.append(msg)
        
        # Update buffer
        self.conversation_buffer = important_messages[-self.config.MAX_CONVERSATION_BUFFER:]

    def get_react_prompt(self, user_message: str, memory: Dict) -> str:
        """Generate ReACT prompt with full context"""
        return f"""
        Previous conversation:
        {self.format_conversation_history()}
        Current message: {user_message}
        Current context:
        - Age group: {memory['current_state']['age_group']}
        - Category: {memory['current_state']['category']}
        - Shown items: {json.dumps(memory['current_state']['shown_items'])}
        
        Analyze user needs with full context.
        1. Are they responding to a previous question?
        2. What specific information do they need?
        3. What should our next action be?
        """

    def get_conversation_memory(self, limit: int = 5) -> Dict:
        """Get formatted conversation memory for prompts"""
        memory = {
            "messages": self.conversation_buffer[-limit:],
            "current_state": {
                "category": self.user_state.current_category,
                "age_group": self.user_state.age_group,
                "shown_items": {
                    "products": list(self.user_state.shown_products),
                    "recipes": list(self.user_state.shown_recipes),
                    "health_info": list(self.user_state.shown_health_info),
                    "parenting_advice": list(self.user_state.shown_advice)
                }
            },
            "context_history": self.user_state.conversation_context
        }
        return memory

    def handle_conversation_error(self, error: ConversationError) -> str:
        """Handle conversation errors gracefully"""
        error_responses = {
            "detection_error": "I'm having trouble understanding your request. Could you please rephrase it?",
            "gpt_error": "I'm currently having difficulty processing your request. Please try again in a moment.",
            "max_retries_exceeded": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
            "validation_error": "There seems to be an issue with processing your request. Could you provide more details?"
        }
        
        # Log error details
        logging.error(f"Conversation error: {error.get_error_details()}")
        
        return error_responses.get(
            error.error_type,
            "I apologize, but I encountered an unexpected issue. Could you please rephrase your request?"
        )

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
        if not safety_result['is_safe']:
            return False, None, "I apologize, but I cannot process that type of content."
            
        return True, sanitized_message, None

    def save_conversation_history(self) -> None:
        """Save conversation history to persistent storage"""
        try:
            if not os.path.exists('conversation_history'):
                os.makedirs('conversation_history')
                
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'conversation_history/conversation_{timestamp}.json'
            
            history_data = {
                "conversation_id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "messages": self.conversation_buffer,
                "metadata": {
                    "user_state": {
                        "final_category": self.user_state.current_category,
                        "final_age_group": self.user_state.age_group,
                        "topics_discussed": list(set(m.get('topic') for m in self.conversation_buffer 
                                                if m.get('topic')))
                    },
                    "conversation_metrics": {
                        "total_messages": len(self.conversation_buffer),
                        "category_changes": self._count_category_changes(),
                        "final_engagement_score": self._calculate_user_engagement()
                    }
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(history_data, f, indent=2)
                
            logging.info(f"Conversation history saved to {filename}")
            
        except Exception as e:
            logging.error(f"Error saving conversation history: {str(e)}")

    def cleanup(self) -> None:
        """Perform cleanup operations before shutting down"""
        try:
            # Save conversation history
            if self.conversation_buffer:
                self.save_conversation_history()
                
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()
         
    def get_category_collection(self, category: str):
        """Get ChromaDB collection for a specific category"""
        chroma_config = self.config.get_chroma_config(category)
        return self.client.get_or_create_collection(
            name=chroma_config["collection_name"],
            metadata=chroma_config["settings"]
        )
        
    def verify_collection_state(self, category: str) -> bool:
        """Verify ChromaDB collection exists and is accessible"""
        try:
            collection = self.get_category_collection(category)
            return collection is not None
        except Exception as e:
            logging.error(f"Error verifying collection state for {category}: {e}")
            return False