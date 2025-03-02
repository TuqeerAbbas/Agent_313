# managers/health_concern_manager.py

from typing import Dict, List, Any, Optional
import json
import logging
import time
import chromadb
import re
from langchain_openai import OpenAIEmbeddings

from config.config import Config
from models.user_state import UserState
from utils.error_handler import ConversationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HealthConcernManager:
    """Handles health-related inquiries with strict medical boundaries."""
    
    def __init__(self, conversation_manager):
        self.config = Config()
        self.manager = conversation_manager
        self.max_items_per_response = self.config.MAX_ITEMS_PER_RESPONSE
        self.CATEGORY = "health"
        
        # Emergency detection thresholds
        self.emergency_thresholds = {
            "high_risk": 0.8,
            "medium_risk": 0.6,
            "low_risk": 0.3
        }

        # Initialize ChromaDB - CHANGED from Elasticsearch
        chroma_config = self.config.get_chroma_config(self.CATEGORY)
        self.client = chromadb.PersistentClient(path=chroma_config["path"])
        self.collection = self.client.get_or_create_collection(
            name=chroma_config["collection_name"],
            metadata=chroma_config["settings"]
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=self.config.OPENAI_API_KEY
        )
        
        # Initialize medical disclaimers
        self.initialize_medical_disclaimers()
        
        # Add state tracking for pending health queries
        if not hasattr(conversation_manager.user_state, 'pending_health_query'):
            conversation_manager.user_state.pending_health_query = None

        # Validate required methods and state
        self._validate_required_methods()
        self._validate_state()

    def _validate_required_methods(self) -> None:
        """
        Validates that all required methods are present in the class.
        Raises AttributeError if any required method is missing.
        """
        required_methods = [
            'get_next_health_batch',
            'generate_medical_redirection',
            'suggest_professional_consultation',
            'handle_emergency_situation',
            'handle_medical_attention_needed',
            'generate_health_disclaimers',
            'ensure_hallucination_free_response',
            'generate_health_info_presentation'
        ]
        
        missing_methods = [m for m in required_methods if not hasattr(self, m)]
        if missing_methods:
            error_msg = f"HealthConcernManager missing required methods: {', '.join(missing_methods)}"
            logging.error(error_msg)
            raise AttributeError(error_msg)

    def generate_medical_redirection(self, health_aspects: Dict, reasoning: str) -> str:
        """
        Generates a response directing the user to seek medical attention.
        Includes appropriate context and disclaimers.
        """
        try:
            redirect_message = [
                f"Based on your question about {health_aspects.get('topic_type', 'this health concern')}, "
                "I recommend consulting with a healthcare provider.",
                "",
                "Reason for this recommendation:",
                f"• {reasoning}",
                "",
                "While waiting to see a healthcare provider:",
                "• Monitor any symptoms",
                "• Document when symptoms started and any changes",
                "• Follow any previous medical advice you've received",
                "• If symptoms worsen, seek immediate medical attention"
            ]
            
            return "\n".join(redirect_message)
        except Exception as e:
            logging.error(f"Error generating medical redirection: {str(e)}")
            return self.generate_safe_health_fallback()

    def suggest_professional_consultation(self, health_aspects: Dict, reasoning: str) -> str:
        """
        Generates a response suggesting professional medical consultation.
        Includes context-specific guidance and appropriate disclaimers.
        """
        try:
            consultation_message = [
                "Based on your description, I recommend discussing this with your child's healthcare provider.",
                "",
                "A healthcare professional can:",
                "• Perform a proper medical evaluation",
                "• Provide personalized medical advice",
                "• Monitor your child's health over time",
                "• Address any specific concerns you have",
                "",
                f"Specifically for {health_aspects.get('topic_type', 'this health concern')}:",
                f"• {reasoning}"
            ]
            
            return "\n".join(consultation_message)
        except Exception as e:
            logging.error(f"Error suggesting professional consultation: {str(e)}")
            return self.generate_safe_health_fallback()

    def ensure_hallucination_free_response(self, response: str, context_data: Dict, category: str) -> str:
        """
        Verifies that the generated response only contains information present in the source data.
        Implements strict medical safety checks.
        """
        try:
            verification_prompt = f"""
            Task: Verify medical information accuracy.

            Response: {response}
            Source Data: {json.dumps(context_data)}
            Category: {category}

            Verify that:
            1. All medical information comes from provided data
            2. No unsupported medical claims are made
            3. All safety warnings and disclaimers are included
            4. No specific medical advice is given
            5. Appropriate referral to healthcare providers is maintained

            Return only "Yes" if verified, "No" if not.
            """
            
            verification = self.manager.get_gpt_response(verification_prompt).strip().lower()
            
            if verification == "yes":
                return response
                
            logging.warning("Response failed hallucination check - using fallback")
            return self.generate_safe_health_fallback()
            
        except Exception as e:
            logging.error(f"Error in response verification: {str(e)}")
            return self.generate_safe_health_fallback()

    """def _meets_medical_safety_requirements(self, info: Dict) -> bool:
        
        ###Verifies that health information meets all medical safety requirements.###
    
        try:
            required_safety_elements = {
                'medical_disclaimers': bool,
                'risk_level': str,
                'safety_notes': list,
                'contraindications': list
            }
            
            # Check for required safety elements
            for element, expected_type in required_safety_elements.items():
                if element not in info:
                    logging.warning(f"Missing safety element: {element}")
                    return False
                if not isinstance(info[element], expected_type):
                    logging.warning(f"Invalid type for {element}")
                    return False
            
            # Check risk level is acceptable
            if info['risk_level'].lower() == 'high':
                logging.warning("High risk content detected")
                return False
            
            # Verify disclaimers are present
            if not info['medical_disclaimers']:
                logging.warning("Missing medical disclaimers")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking medical safety requirements: {str(e)}")
            return False"""
               
    def _meets_medical_safety_requirements(self, info: Dict) -> bool:
          
        ####use this method for the current dummy data to check the flow but use the above method for the actual data###
        
        try:
            # All fields are optional with defaults
            optional_fields = {
                'risk_level': str,
                'medical_disclaimers': bool,
                'safety_notes': list,
                'contraindications': list
            }
            
            # Apply defaults if fields are missing
            if 'risk_level' not in info:
                info['risk_level'] = 'low'
                logging.info("Added default risk_level: low")
            if 'medical_disclaimers' not in info:
                info['medical_disclaimers'] = "This is general information only, not medical advice. Consult a healthcare provider for specific concerns."
                logging.info("Added default medical_disclaimers")
            if 'safety_notes' not in info:
                info['safety_notes'] = ["Monitor your child and seek professional advice if needed."]
                logging.info("Added default safety_notes")
            if 'contraindications' not in info:
                info['contraindications'] = []
                logging.info("Added default contraindications")
            
            # Type checks (relaxed to allow flexibility)
            if not isinstance(info.get('risk_level'), str):
                info['risk_level'] = 'low'
                logging.warning(f"Invalid risk_level type, defaulted to 'low'")
            if not isinstance(info.get('medical_disclaimers'), str):  # Changed to str for consistency
                info['medical_disclaimers'] = "This is general information only, not medical advice."
                logging.warning("Invalid medical_disclaimers type, defaulted")
            if not isinstance(info.get('safety_notes'), list):
                info['safety_notes'] = []
                logging.warning("Invalid safety_notes type, defaulted to empty list")
            if not isinstance(info.get('contraindications'), list):
                info['contraindications'] = []
                logging.warning("Invalid contraindications type, defaulted to empty list")
            
            # Reject high-risk content unless approved
            if info.get('risk_level', 'low').lower() == 'high' and not self.manager.user_state.get('high_risk_approved', False):
                logging.warning("High risk content detected and not approved")
                return False
            
            return True
        except Exception as e:
            logging.error(f"Error checking medical safety: {str(e)}")
            return False
        
    def log_emergency_guidance(self, assessment: Dict) -> None:
        """
        Logs emergency guidance provided to the user for audit purposes.
        """
        try:
            guidance_log = {
                'timestamp': time.time(),
                'emergency_type': assessment.get('emergency_type', 'unknown'),
                'guidance_provided': assessment.get('guidance', ''),
                'immediate_actions': assessment.get('immediate_actions', []),
                'needs_medical_attention': assessment.get('needs_medical_attention', True)
            }
            
            # Initialize log if doesn't exist
            if not hasattr(self.manager.user_state, 'emergency_guidance_log'):
                self.manager.user_state.emergency_guidance_log = []
                
            self.manager.user_state.emergency_guidance_log.append(guidance_log)
            logging.info(f"Emergency guidance logged: {assessment['emergency_type']}")
            
        except Exception as e:
            logging.error(f"Error logging emergency guidance: {str(e)}")

    def _validate_state(self) -> None:
        """
        Validates that all required state variables are present and properly initialized.
        Raises AttributeError if any required state is missing.
        """
        required_state = {
            'user_state': self.manager.user_state,
            'config': self.config,
            'client': self.client,  # ChromaDB client instead of ES
            'collection': self.collection,  # ChromaDB collection
            'emergency_thresholds': self.emergency_thresholds
        }
        
        missing_state = [k for k, v in required_state.items() if v is None]
        if missing_state:
            error_msg = f"HealthConcernManager missing required state: {', '.join(missing_state)}"
            logging.error(error_msg)
            raise AttributeError(error_msg)

    def _validate_health_response(self, response: str) -> bool:
        """
        Validates that a health-related response meets all required criteria.
        Ensures appropriate disclaimers and safety information are included.
        """
        try:
            # Check basic response validity
            if not response or len(response.strip()) < 10:
                logging.warning("Response too short or empty")
                return False
                
            # Verify required disclaimers are present
            required_disclaimers = [
                self.standard_disclaimer,
                "consult with your healthcare provider",
                "not medical advice"
            ]
            
            missing_disclaimers = [
                d for d in required_disclaimers 
                if d.lower() not in response.lower()
            ]
            
            if missing_disclaimers:
                logging.warning(f"Missing required disclaimers: {missing_disclaimers}")
                return False
                
            # Check for inappropriate medical advice
            inappropriate_terms = [
                "you should take",
                "you need to take",
                "I recommend taking",
                "you must take"
            ]
            
            if any(term in response.lower() for term in inappropriate_terms):
                logging.warning("Response contains inappropriate medical advice")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error validating health response: {str(e)}")
            return False

    def initialize_medical_disclaimers(self) -> None:
        """Initialize standard medical disclaimers"""
        self.standard_disclaimer = (
            "IMPORTANT: This information is for general educational purposes only "
            "and should not be considered medical advice. Always consult with your "
            "healthcare provider for medical diagnosis, treatment, and advice."
        )
        
        self.emergency_disclaimer = (
            "⚠️ If you believe this is a medical emergency, call emergency services "
            "(911 in the US) immediately. Do not wait for online guidance."
        )
        
        self.symptom_disclaimer = (
            "Symptoms can indicate various conditions and require proper medical "
            "evaluation for accurate diagnosis. If you're concerned about your "
            "child's symptoms, please contact your healthcare provider."
        )

    def handle_health_concern(self, user_message: str, age_group: str, requery: bool) -> str:
        """
        Main entry point for all health-related interactions. This method orchestrates
        the complete health information process using our two-step search approach,
        while enforcing strict medical safety boundaries and emergency protocols.
        
        The method follows this carefully controlled process:
        1. Check for emergencies and immediate medical needs first
        2. Get ReACT decision for safe health information handling
        3. If safe to provide information:
        - Step 1: Get all potential health info (base_search)
        - Step 2: Rank and return safest matches (get_next_health_batch)
        4. Always include appropriate medical disclaimers
        
        Args:
            user_message: The user's health-related query
            age_group: Child's age group (e.g., "0-6 months")
            requery: Whether this is a repeated search
            
        Returns:
            str: Formatted response with health information and required disclaimers
        """
        try:
            # First verify we have required age information
            if not age_group:
                logging.info("No age group provided for health query")
                
                # Store the current message as pending health query
                self.manager.user_state.pending_health_query = user_message
                self.manager.user_state.current_category = "health_concern"  # Set category explicitly
                
                return ("Could you please tell me your child's age? "
                    "This helps me provide age-appropriate health information.")
            
            # Check if we have a pending health query
            if hasattr(self.manager.user_state, 'pending_health_query') and self.manager.user_state.pending_health_query:
                # Use the pending query instead of just the age response
                original_query = self.manager.user_state.pending_health_query
                logging.info(f"Using pending health query: {original_query}")
                
                # If the user message is just an age, use the pending query instead
                if self._is_just_age_response(user_message):
                    # Store the age response for context but use original query
                    age_response = user_message
                    user_message = original_query
                    logging.info(f"Replaced age response '{age_response}' with original query: {user_message}")
                
                # Clear the pending query so we don't reuse it
                self.manager.user_state.pending_health_query = None
            
            # Make sure we maintain the category
            self.manager.user_state.current_category = "health_concern"
                
            # First perform emergency assessment before any other processing
            logging.info("Performing emergency assessment")
            emergency_assessment = self.assess_medical_urgency(user_message, age_group)
            self.log_emergency_assessment(emergency_assessment)
            
            # Immediate emergency handling takes precedence
            if emergency_assessment["is_emergency"]:
                logging.info("Emergency situation detected - providing emergency guidance")
                return f"{self.handle_emergency_situation(emergency_assessment)}\n\n{self.emergency_disclaimer}"
                
            # Handle non-emergency but medical attention needed
            if emergency_assessment["needs_medical_attention"]:
                logging.info("Medical attention needed - redirecting to healthcare provider")
                return f"{self.handle_medical_attention_needed(emergency_assessment)}\n\n{self.standard_disclaimer}"

            # Get ReACT decision about how to handle health information
            logging.info(f"Getting ReACT decision for health information - Age: {age_group}")
            react_decision = self.get_health_react_decision(
                user_message=user_message,
                age_group=age_group,
                emergency_assessment=emergency_assessment
            )
            
            # Only proceed with high-confidence decisions
            if react_decision["confidence"] >= self.config.REACT_CONFIDENCE_THRESHOLD:
                if react_decision["decision"] == "provide_info":
                    # PATH 1: SAFE HEALTH INFORMATION PROVISION
                    logging.info("Starting two-step health information search process")
                    
                    # Step 1: Get complete pool of age-appropriate health information
                    base_results = self.base_health_search(age_group)
                    if not base_results:
                        logging.info(f"No health information found for age group: {age_group}")
                        return ("I don't have specific health information for this age group. "
                            "Please consult your healthcare provider.")
                    
                    # Store complete result set for pagination
                    self.manager.user_state.all_available_health_info = base_results
                    self.manager.user_state.health_current_page = 0
                    logging.info(f"Stored {len(base_results)} health items for pagination")
                    
                    # Step 2: Get first batch using medically-aware similarity ranking
                    response = self.get_next_health_batch(
                        user_message=user_message,
                        react_decision=react_decision
                    )
                    
                elif react_decision["decision"] == "redirect_medical":
                    # PATH 2: MEDICAL REDIRECTION NEEDED
                    logging.info("Redirecting to medical professional")
                    response = self.generate_medical_redirection(
                        health_aspects=react_decision["health_aspects"],
                        reasoning=react_decision["reasoning"]
                    )
                    
                else:
                    # PATH 3: PROFESSIONAL CONSULTATION RECOMMENDED
                    logging.info("Suggesting professional consultation")
                    response = self.suggest_professional_consultation(
                        health_aspects=react_decision["health_aspects"],
                        reasoning=react_decision["reasoning"]
                    )
            else:
                # Fall back to safe handling if low confidence
                logging.info("Falling back to basic health handling due to low confidence")
                response = self.fallback_health_handling(
                    user_message=user_message,
                    age_group=age_group,
                    requery=requery
                )
                
            # Always include appropriate health disclaimers
            disclaimers = self.generate_health_disclaimers(
                health_aspects=react_decision.get("health_aspects", {}),
                required_disclaimers=react_decision.get("required_disclaimers", [])
            )
            
            return f"{response}\n\n{disclaimers}"
            
        except Exception as e:
            # Log error details and return safe fallback with disclaimer
            logging.error(f"Error in health concern handling: {str(e)}", exc_info=True)
            return f"{self.generate_safe_health_fallback()}\n\n{self.standard_disclaimer}"

    def fallback_health_handling(self, user_message: str, age_group: str, requery: bool) -> str:
        """
        Handle health concerns when primary method fails. Uses a simplified search approach
        with extra emphasis on safety disclaimers and medical guidance.
        
        Args:
            user_message: User's original request
            age_group: Child's age group (e.g., "18-24 months")
            requery: Whether this is a repeated search attempt
            
        Returns:
            str: Formatted health information with appropriate medical disclaimers
        """
        try:
            # First perform a basic safety check even in fallback mode
            safety_check = {
                "requires_professional": False,
                "safety_concerns": [],
                "sensitivity_level": "medium"
            }
            
            # Perform a basic search focusing on general health information
            basic_search = self.search_health_information(
                age_group=age_group,
                health_aspects={
                    "topic_type": "wellness",
                    "safe_discussion_points": ["general health", "well-being"]
                }
            )
            
            if basic_search:
                # Generate a presentation with additional safety context
                return self.generate_health_info_presentation(
                    selected_info=basic_search[:self.max_items_per_response],
                    user_message=user_message,
                    react_decision={
                        "health_aspects": {
                            "topic_type": "wellness",
                            "safe_discussion_points": ["general health", "well-being"],
                            "risk_factors": []
                        },
                        "information_boundaries": {
                            "safe_to_discuss": ["general wellness", "preventive care"],
                            "avoid_discussing": ["specific medical advice", "treatment recommendations"]
                        },
                        "required_disclaimers": ["general_health", "consult_professional"]
                    }
                )
                
            # If no health information found, return safe fallback message
            return self.generate_safe_health_fallback()
            
        except Exception as e:
            logging.error(f"Error in health concern fallback handling: {str(e)}")
            return self.generate_safe_health_fallback()

    def assess_medical_urgency(self, user_message: str, age_group: str) -> Dict:
        """
        Assess urgency level of health concern with comprehensive safety checks.
        
        Args:
            user_message: The user's health query
            age_group: Child's age group
            
        Returns:
            Dict: Structured assessment with all required fields
        """
        assessment_prompt = f"""
        Assess medical urgency of:
        Message: {user_message}
        Age Group: {age_group}
        
        Return JSON with ALL of these fields:
        {{
            "emergency_type": "condition type",
            "concern_type": "specific concern",
            "is_emergency": boolean,
            "needs_medical_attention": boolean,
            "immediate_actions": ["action1", "action2"],
            "guidance": "detailed guidance",
            "safety_notes": ["note1", "note2"]
        }}
        """
        
        try:
            response = self.manager.get_gpt_response(assessment_prompt)
            cleaned_response = self._clean_json_response(response)
            assessment = json.loads(cleaned_response)
            
            # Ensure all required fields are present with fallbacks
            default_assessment = {
                "emergency_type": "unspecified medical concern",
                "concern_type": "unspecified medical concern",
                "is_emergency": False,
                "needs_medical_attention": True,
                "immediate_actions": ["Consult healthcare provider"],
                "guidance": "Please consult with a healthcare provider for evaluation.",
                "safety_notes": ["When in doubt, seek professional medical advice."]
            }
            
            # Update default values with any valid data from the response
            default_assessment.update(assessment)
            
            return default_assessment
            
        except json.JSONDecodeError:
            logging.error("Failed to parse medical urgency assessment")
            return default_assessment
        except Exception as e:
            logging.error(f"Error in medical urgency assessment: {str(e)}")
            return default_assessment

    def handle_emergency_situation(self, assessment: Dict) -> str:
        """Handle detected emergency situations with clear guidance"""
        emergency_response = [
            "⚠️ MEDICAL ATTENTION REQUIRED ⚠️\n",
            "Based on the information provided, this situation requires immediate medical attention.\n",
            "IMMEDIATE STEPS:"
        ]
        
        # Add immediate actions
        for action in assessment["immediate_actions"]:
            emergency_response.append(f"• {action}")
        
        emergency_response.extend([
            "\nIf this is a medical emergency:",
            "• Call emergency services (911 in the US) immediately",
            "• Stay with your child",
            "• Follow emergency operator instructions",
            "\nThis is not medical advice. Always err on the side of caution with child health concerns."
        ])
        
        # Log emergency guidance
        self.log_emergency_guidance(assessment)
        
        return "\n".join(emergency_response)

    def get_health_react_decision(self, user_message: str, age_group: str, emergency_assessment: Dict) -> Dict:
        """Get ReACT reasoning for health information handling"""
        react_prompt = f"""
            Task: Determine how to handle health-related inquiry.

            Context:
            - User Message: {user_message}
            - Age Group: {age_group}
            - Current Topics: {json.dumps(self.manager.user_state.current_health_topics)}
            - Previous Info Shown: {list(self.manager.user_state.shown_health_info)}
            - Initial Assessment: {json.dumps(emergency_assessment)}
            
            You must respond with ONLY valid JSON in exactly this format:
            {{
                "decision": "provide_info/redirect_medical/suggest_professional",
                "confidence": 0.0-1.0,
                "health_aspects": {{
                    "topic_type": "wellness/preventive/symptom",
                    "safe_discussion_points": ["point1", "point2"],
                    "risk_factors": ["factor1", "factor2"]
                }},
                "information_boundaries": {{
                    "safe_to_discuss": ["topic1", "topic2"],
                    "avoid_discussing": ["topic1", "topic2"]
                }},
                "required_disclaimers": ["disclaimer1", "disclaimer2"],
                "reasoning": "explanation"
            }}
            """
            
        try:
            response = self.manager.get_gpt_response(react_prompt)
            # Clean the response before parsing JSON
            cleaned_response = self._clean_json_response(response)
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse health react decision: {e}")
            return {
                "decision": "redirect_medical",
                "confidence": 0.5,
                "health_aspects": {
                    "topic_type": "general",
                    "safe_discussion_points": [],
                    "risk_factors": []
                },
                "information_boundaries": {
                    "safe_to_discuss": [],
                    "avoid_discussing": ["specific medical advice"]
                },
                "required_disclaimers": ["consult_professional", "not_medical_advice"],
                "reasoning": "Failed to parse response, defaulting to medical redirection"
            }
            
    def handle_health_information(self, user_message: str, age_group: str, react_decision: Dict) -> str:
        try:
            # Step 1: First get all matching health information by age/category only
            logging.info(f"Performing base health information search for age group: {age_group}")
            base_results = self.base_health_search(age_group)
            
            if not base_results:
                return "I don't have specific health information for this age group. Please consult your healthcare provider."
                
            # Store results for pagination
            self.manager.user_state.all_available_health_info = base_results
            self.manager.user_state.health_current_page = 0
            
            # Step 2: Get first batch with similarity ranking
            return self.get_next_health_batch(user_message, react_decision)
                
        except Exception as e:
            logging.error(f"Error in health information handling: {str(e)}")
            return "I apologize, but I'm having trouble processing health information. Please consult your healthcare provider."
        
    def base_health_search(self, age_group: str) -> List[Dict]:
        """
        Enhanced base search for health information using ChromaDB.
        CHANGED: Now uses ChromaDB instead of Elasticsearch
        """
        logging.info("\n=== Step 1: Base Health Information Search ===")
        logging.info(f"Search Parameters:")
        logging.info(f"Category: '{self.CATEGORY}'")
        logging.info(f"Age Group: '{age_group}'")
        
        try:
            # Create search text and get embedding
            search_text = f"category:{self.CATEGORY} age_group:{age_group}"
            query_embedding = self.embeddings.embed_query(search_text)
            
            # Query ChromaDB with safety filters
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=7,  # Large pool for careful medical content filtering
                where={
                "$and": [
                    {"age_group": {"$eq": age_group}}, ##"age_group": "12-18 months", ##and the orignal donn't contain 'and' conditon
                    {"risk_level": {"$ne": "high"}}    ##"risk_level": {"$ne": "high"}
                ]
            }
        )
            # Add detailed logging of raw results
            logging.info(f"Raw ChromaDB results:")
            logging.info(f"Found {len(results['documents'][0])} documents")
            # Print each document and its metadata
            for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                logging.info(f"Document {i + 1}:")
                logging.info(f"Content: {doc}")
                logging.info(f"Metadata: {metadata}")
        
            
            # Process and validate health information
            processed_health_info = []
            for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                try:
                    info = json.loads(doc)
                    info.update(metadata)
                    
                    # Verify all required medical fields and safety requirements
                    if self._meets_medical_safety_requirements(info):
                        processed_health_info.append(info)
                    else:
                        logging.warning(f"Health info failed safety check: {info.get('title', 'Unknown')}")
                        
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing health info: {e}")
                    continue
            
            logging.info(f"Found {len(processed_health_info)} validated health documents")
            return processed_health_info
            
        except Exception as e:
            logging.error(f"Base health search failed: {str(e)}", exc_info=True)
            return []

    def _perform_health_diagnostic_search(self, age_group: str) -> List[Dict]:
        """
        Performs a carefully controlled diagnostic search for health information when 
        exact matching fails. This maintains strict medical safety boundaries while
        attempting to identify potential content matches.
        """
        diagnostic_query = {
            "query": {
                "bool": {
                    "should": [
                        # Look for partial matches on health category
                        {"match": {"category": "health_concern"}},
                        # Look for partial matches on age group
                        {"match": {"age_group": age_group}},
                        # Look for health information in similar age ranges
                        {"prefix": {"age_group": age_group.split()[0]}}
                    ],
                    "minimum_should_match": 1,
                    "must_not": [
                        # Maintain strict safety boundaries
                        {"term": {"emergency_only": True}},
                        {"term": {"professional_only": True}},
                        {"term": {"risk_level.keyword": "high"}}
                    ]
                }
            },
            "size": 5
        }
        
        try:
            results = self.es.search(
                index=self.config.ES_INDEX,
                body=diagnostic_query
            )
            return [hit['_source'] for hit in results['hits']['hits']]
        except Exception as e:
            logging.error(f"Health diagnostic search failed: {str(e)}")
            return []

    def _process_health_information_results(self, hits: List[Dict]) -> List[Dict]:
        """
        Processes and validates health information search results with strict medical 
        safety checks. Ensures all required medical disclaimers and safety information 
        are present.
        """
        processed_health_info = []
        required_fields = {
            'title', 'content', 'age_group', 'category'
        }
        
        for hit in hits:
            info = hit['_source']
            
            # Verify all required medical fields are present
            if all(field in info for field in required_fields):
                # Perform medical safety checks
                risk_level = info.get('risk_level', 'medium')
                if risk_level == 'high':
                    # Log warning for high-risk content
                    logging.warning(
                        f"High risk health content found: {info.get('title')} - "
                        "Ensuring all medical disclaimers are included"
                    )
                    
                    # Verify medical disclaimers are present
                    if not info.get('medical_disclaimers'):
                        logging.error(
                            f"Missing medical disclaimers for high-risk content: {info.get('title')}"
                        )
                        continue
                
                # Add validated health information
                processed_health_info.append(info)
            else:
                # Log warning for incomplete health information
                logging.warning(
                    f"Skipping incomplete health information: {info.get('title', 'Unknown')} - "
                    f"Missing required medical fields: {required_fields - set(info.keys())}"
                )
        
        return processed_health_info

    """def get_next_health_batch(self, user_message: str, react_decision: Dict) -> str:
        
        logging.info("Getting next health information batch")
        
        # Make sure we maintain the category
        self.manager.user_state.current_category = "health_concern"
        
        # Get our working set
        all_info = self.manager.user_state.all_available_health_info
        shown = self.manager.user_state.shown_health_info
        current_page = self.manager.user_state.health_current_page
        
        if not all_info:
            logging.error("No health information available in state")
            return self.generate_safe_health_fallback()
        
        # Filter out shown information
        available = [
            info for info in all_info 
            if str(info.get('_id', '')) not in shown
        ]
        
        if not available:
            return "You've seen all available health information. Would you like to explore different topics?"
        
        # Get starting index for pagination
        start_idx = current_page * 3
        logging.info(f"Page {current_page}, {len(available)} health items available")
        
        # If we have meaningful search text, use vector similarity
        if user_message.strip():
            try:
                # Get embedding for user message
                query_embedding = self.embeddings.embed_query(user_message)
                
                # Search using ChromaDB
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=3,
                    where={
                        "_id": {"$in": [str(h.get('_id', '')) for h in available]},
                        "risk_level": {"$ne": "high"}  # Safety filter
                    }
                )
                
                # Process results with medical safety checks
                selected_info = []
                for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                    try:
                        info = json.loads(doc)
                        info.update(metadata)
                        
                        # Additional safety check before including
                        if self._is_medically_safe_to_show(info, react_decision):
                            selected_info.append(info)
                            
                    except json.JSONDecodeError:
                        continue
                
                logging.info("Using similarity-based selection with medical safety checks")
                
            except Exception as e:
                logging.error(f"Error in similarity search: {e}")
                selected_info = self.get_random_health_info(available, 3)
                logging.info("Using medically safe random selection (similarity search failed)")
        else:
            # No search criteria - use medically safe random selection
            selected_info = self.get_random_health_info(available, 3)
            logging.info("Using medically safe random selection (no search criteria)")
        
        # Update state
        try:
            self.update_health_info_state(selected_info)
            self.manager.user_state.health_current_page += 1
        except Exception as e:
            logging.error(f"Error updating health info state: {str(e)}")
        
        # Generate response with required medical disclaimers
        response = self.generate_health_info_presentation(
            selected_info=selected_info,
            user_message=user_message,
            react_decision=react_decision
        )
        
        # Add pagination prompt if more information available
        if len(available) > (start_idx + 3):
            response += "\n\nWould you like to see more health information?"
        
        # Always include appropriate medical disclaimers
        disclaimers = self.generate_health_disclaimers(
            react_decision["health_aspects"],
            react_decision.get("required_disclaimers", [])
        )
        
        return f"{response}\n\n{disclaimers}" """
        
    def get_next_health_batch(self, user_message: str, react_decision: Dict) -> str:
        
        all_info = self.manager.user_state.all_available_health_info
        shown = self.manager.user_state.shown_health_info
        current_page = self.manager.user_state.health_current_page
        
        available = [info for info in all_info if str(info.get('_id', '')) not in shown]
        if not available:
            return "You've seen all available health information..."
        
        start_idx = current_page * 3
        logging.info(f"Page {current_page}, {len(available)} health items available")
        
        if user_message.strip():
            try:
                query_embedding = self.embeddings.embed_query(user_message)
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=3,
                    where={
                        "$and": [
                            #{"_id": {"$in": [str(h.get('_id', '')) for h in available]}}, ##orignal code tha ye wala
                            {"age_group": {"$eq": self.manager.user_state.age_group}},
                            {"risk_level": {"$ne": "high"}}
                        ]
                    }
                )
                selected_info = []
                for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
                    info = json.loads(doc)
                    info.update(metadata)
                    if self._is_medically_safe_to_show(info, react_decision):
                        selected_info.append(info)
                logging.info("Using similarity-based selection with medical safety checks")
            except Exception as e:
                logging.error(f"Error in similarity search: {e}")
                selected_info = self.get_random_health_info(available, 3)
                logging.info("Using medically safe random selection (similarity search failed)")
        else:
            selected_info = self.get_random_health_info(available, 3)
            logging.info("Using medically safe random selection (no search criteria)")
        
        try:
            self.update_health_info_state(selected_info)
            self.manager.user_state.health_current_page += 1
        except Exception as e:
            logging.error(f"Error updating health info state: {str(e)}")
        
        response = self.generate_health_info_presentation(selected_info, user_message, react_decision)
        disclaimers = self.generate_health_disclaimers(react_decision["health_aspects"], react_decision.get("required_disclaimers", []))
        return f"{response}\n\n{disclaimers}"    

    def _calculate_safe_health_score(self, base_score: float, info: Dict, react_decision: Dict) -> float:
            """Calculate final score with medical safety considerations"""
            score = base_score
            
            # Boost for matching safe discussion points
            safe_points = react_decision["health_aspects"].get("safe_discussion_points", [])
            if any(point.lower() in info.get("content", "").lower() for point in safe_points):
                score += 0.2
            
            # Penalty for high-risk topics unless specifically requested
            if info.get("risk_level", "low") == "high" and not react_decision["health_aspects"].get("high_risk_approved"):
                score *= 0.5
            
            # Boost for preventive/wellness content when appropriate
            if react_decision["health_aspects"].get("topic_type") == "wellness" and info.get("content_type") == "preventive":
                score += 0.15
            
            return min(1.0, score)  # Cap at 1.0

    def get_health_info_text(self, info: Dict) -> str:
        """Get searchable text representation of health information"""
        return " ".join([
            info.get("title", ""),
            info.get("content", ""),
            info.get("symptoms", ""),
            info.get("recommendations", ""),
            info.get("precautions", ""),
            info.get("category", ""),
            info.get("age_group", "")
        ]).lower()

    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using TF-IDF"""
        try:
            # Import at class level
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception as e:
            logging.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def get_random_health_info(self, info_items: List[Dict], count: int) -> List[Dict]:
        """Get random health information when no specific matching needed"""
        import random
        if len(info_items) <= count:
            return info_items
        return random.sample(info_items, count)

    def search_health_information(self, age_group: str, health_aspects: Dict) -> List[Dict]:
        """Enhanced health information search using Elasticsearch"""
        try:
            search_aspects = []
            if health_aspects.get("safe_discussion_points"):
                search_aspects.extend(health_aspects["safe_discussion_points"])
            
            search_title = " ".join(search_aspects)
            
            results = self.base_health_search(age_group)
            
            return [hit['_source'] for hit in results]
            
        except Exception as e:
            logging.error(f"Error in health information search: {str(e)}")
            return []

    def select_health_info_for_presentation(self, available_info: List[Dict], user_message: str, react_decision: Dict) -> List[Dict]:
        """Select most relevant health information"""
        sorted_info = sorted(
            available_info,
            key=lambda i: self._calculate_health_info_relevance(i, user_message, react_decision),
            reverse=True
        )
        
        return sorted_info[:self.max_items_per_response]

    def _calculate_health_info_relevance(self, info: Dict, user_message: str, 
                                       react_decision: Dict) -> float:
        """Calculate relevance score for health information"""
        score = 0.0
        
        # Match safe discussion points
        safe_points = react_decision["health_aspects"].get("safe_discussion_points", [])
        if any(point.lower() in info.get("content", "").lower() for point in safe_points):
            score += 0.3
        
        # Consider risk factors
        risk_factors = react_decision["health_aspects"].get("risk_factors", [])
        if any(factor.lower() in info.get("content", "").lower() for factor in risk_factors):
            score += 0.2
            
        # Keyword matching
        keywords = set(user_message.lower().split())
        info_text = info.get("content", "").lower()
        matching_keywords = keywords.intersection(set(info_text.split()))
        score += len(matching_keywords) * 0.1
        
        return min(1.0, score)

    def _is_medically_safe_to_show(self, info: Dict, react_decision: Dict) -> bool:
        """Determines if health information is safe to show based on medical context"""
        try:
            # Check risk levels
            risk_level = info.get('risk_level', 'medium')
            if risk_level == 'high' and not react_decision["health_aspects"].get("high_risk_approved"):
                return False
            
            # Verify medical safety constraints
            if not self._meets_medical_safety_requirements(info):
                return False
            
            # Check if information requires professional guidance
            if info.get('professional_only', False):
                return False
            
            # Verify required disclaimers are present
            if not info.get('medical_disclaimers'):
                logging.warning(f"Missing medical disclaimers for content: {info.get('title', 'Unknown')}")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking medical safety: {str(e)}")
            return False

    def _log_medical_content_delivery(self, info: Dict) -> None:
        """
        Maintains a detailed log of medical content delivery for safety tracking.
        This helps ensure responsible health information sharing.
        """
        try:
            delivery_log = {
                'timestamp': time.time(),
                'content_id': info.get('_id'),
                'risk_level': info.get('risk_level', 'medium'),
                'category': info.get('category'),
                'has_disclaimers': bool(info.get('medical_disclaimers')),
                'requires_followup': info.get('requires_followup', False)
            }
            
            #if 'medical_content_log' not in self.manager.user_state:
            if not hasattr(self.manager.user_state, 'medical_content_log'):
                self.manager.user_state.medical_content_log = []
                
            self.manager.user_state.medical_content_log.append(delivery_log)
            
        except Exception as e:
            logging.error(f"Error logging medical content delivery: {str(e)}")

    def update_health_info_state(self, health_items: List[Dict]) -> None:
        """Updates state with new health information"""
        try:
            for info in health_items:
                # Use a more reliable ID generation method
                if '_id' in info:
                    health_id = str(info['_id'])
                else:
                    # Create a stable ID from title and/or other stable attributes
                    title = info.get('title', '')
                    age_group = info.get('age_group', '')
                    health_id = f"{title}-{age_group}-{time.time()}"
                
                # Ensure shown_health_info is initialized
                if not hasattr(self.manager.user_state, 'shown_health_info'):
                    self.manager.user_state.shown_health_info = set()
                    
                self.manager.user_state.shown_health_info.add(health_id)
                
                # Track medical content delivery for safety
                self._log_medical_content_delivery(info)
            
            self.manager.user_state.current_health_topics = health_items
            self.manager.user_state.last_health_search = time.time()
            
            logging.info(f"Updated state with {len(health_items)} new health items")
            
        except Exception as e:
            logging.error(f"Error in health information state update: {str(e)}")

    def generate_health_info_presentation(self, selected_info: List[Dict], user_message: str, react_decision: Dict) -> str:
        """Generate health information presentation"""
        presentation_prompt = f"""
        Create health information response:
        Information: {json.dumps(selected_info)}
        User Message: {user_message}
        Analysis: {json.dumps(react_decision)}
        
        Response should:
        1. Present information clearly and carefully
        2. Include appropriate medical disclaimers
        3. Emphasize when to seek medical attention
        4. Maintain supportive, informative tone
        """
        
        initial_response = self.manager.get_gpt_response(presentation_prompt)
        return self.ensure_hallucination_free_response(
            initial_response,
            selected_info,
            "Health Information"
        )

    def handle_medical_attention_needed(self, assessment: Dict) -> str:
        """
        Handle situations requiring medical attention but not emergency.
        Safely processes the assessment data with fallbacks for missing information.
        
        Args:
            assessment: Dictionary containing medical assessment details
            
        Returns:
            str: Formatted medical guidance with appropriate disclaimers
        """
        try:
            # Extract assessment details with safe fallbacks
            concern_type = assessment.get("concern_type",
                assessment.get("emergency_type", "medical concern"))
            guidance = assessment.get("guidance",
                assessment.get("immediate_actions", []))
            
            # If guidance is a list, join it into a readable format
            if isinstance(guidance, list):
                guidance = "\n".join(f"• {action}" for action in guidance)
            elif not guidance:
                guidance = "Please consult with a healthcare provider for proper evaluation."
                
            return self.generate_medical_attention_guidance(concern_type, guidance)
            
        except Exception as e:
            logging.error(f"Error processing medical attention guidance: {str(e)}")
            # Provide safe fallback response
            return ("Based on the information provided, I recommend consulting with a healthcare provider. "
                    "They can properly evaluate the situation and provide appropriate medical guidance.")

    def generate_medical_attention_guidance(self, concern_type: str, guidance: str) -> str:
        """
        Generate guidance for seeking medical attention with proper medical disclaimers.
        
        Args:
            concern_type: Type of medical concern
            guidance: Specific guidance or actions to take
            
        Returns:
            str: Formatted medical guidance
        """
        try:
            response = [
                "Based on your description, I recommend having this checked by a healthcare provider.",
                "",  # Empty line for readability
                f"Regarding {concern_type}:",
                guidance,
                "",  # Empty line for readability
                "While waiting to see a healthcare provider:",
                "• Monitor your child's symptoms",
                "• Keep a log of any changes",
                "• Follow any care instructions you've previously received",
                "• If symptoms worsen, seek immediate medical attention"
            ]
            
            return "\n".join(response)
            
        except Exception as e:
            logging.error(f"Error generating medical guidance: {str(e)}")
            return ("Please consult with a healthcare provider who can properly "
                    "evaluate and provide appropriate medical guidance.")

    def generate_health_disclaimers(self, health_aspects: Dict, required_disclaimers: List[str]) -> str:
        """Generate appropriate health-related disclaimers"""
        disclaimers = [self.standard_disclaimer]
        
        if "emergency" in required_disclaimers:
            disclaimers.append(self.emergency_disclaimer)
        if "symptoms" in required_disclaimers:
            disclaimers.append(self.symptom_disclaimer)
            
        return "\n\n" + "\n".join(disclaimers)

    def log_emergency_assessment(self, assessment: Dict) -> None:
        """Log emergency assessment for tracking and analysis"""
        if assessment["emergency_type"] not in self.manager.user_state.health_emergency_history:
            self.manager.user_state.health_emergency_history[assessment["emergency_type"]] = []
            
        self.manager.user_state.health_emergency_history[assessment["emergency_type"]].append({
            "timestamp": time.time(),
            "is_emergency": assessment["is_emergency"],
            "needs_medical_attention": assessment["needs_medical_attention"],
            "immediate_actions": assessment["immediate_actions"]
        })

    def debug_search(self, index_name: str, category: str, age_group: str, health_aspects: Optional[Dict] = None, emergency_assessment: Optional[Dict] = None) -> List[Dict]:
        """
        Performs a refined health information search that considers medical safety,
        symptom matching, and risk levels. This implements step 2 of our search process,
        applying sophisticated medical-aware filtering and scoring.
        
        This function carefully balances providing helpful information while maintaining
        strict medical safety boundaries and appropriate disclaimers.
        
        Args:
            index_name: Name of the Elasticsearch index
            category: Category (should be "Health Concern")
            age_group: Age group to search within
            health_aspects: Dictionary containing safe discussion points and risk factors
            emergency_assessment: Optional emergency assessment results to consider
            
        Returns:
            List[Dict]: List of matching health information items
        """
        # Start with medical safety criteria
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"age_group.keyword": age_group}},
                        {"term": {"category.keyword": "health_concern"}}
                    ]
                }
            }
        }
        
        # Add safe discussion points if provided
        if health_aspects and health_aspects.get("safe_discussion_points"):
            query["query"]["bool"]["should"].extend([
                {
                    "match": {
                        "safe_topics": {
                            "query": point,
                            "boost": 3.0  # Prioritize safe topics
                        }
                    }
                }
                for point in health_aspects["safe_discussion_points"]
            ])
        
        # Add symptom matching with careful boosting
        if health_aspects and health_aspects.get("symptoms"):
            query["query"]["bool"]["should"].extend([
                {
                    "match": {
                        "symptoms": {
                            "query": symptom,
                            "boost": 2.0
                        }
                    }
                }
                for symptom in health_aspects["symptoms"]
            ])
        
        # Add general information matching with appropriate weights
        if health_aspects and health_aspects.get("topic_type"):
            query["query"]["bool"]["should"].extend([
                {"match": {"title": {"query": health_aspects["topic_type"], "boost": 2.0}}},
                {"match": {"content": {"query": health_aspects["topic_type"], "boost": 1.5}}},
                {"match": {"recommendations": {"query": health_aspects["topic_type"], "boost": 1.0}}}
            ])
        
        # If emergency assessment exists, adjust search accordingly
        if emergency_assessment and emergency_assessment.get("needs_medical_attention"):
            # Add medical attention guidance boost
            query["query"]["bool"]["should"].append({
                "term": {
                    "requires_medical_attention": {
                        "value": True,
                        "boost": 2.0
                    }
                }
            })
        
        # Require at least one should clause to match if we have any
        if query["query"]["bool"]["should"]:
            query["query"]["bool"]["minimum_should_match"] = 1
        
        try:
            # Log the search attempt for medical audit
            logging.info(f"\n=== Executing Refined Health Information Search ===\n"
                        f"Age Group: {age_group}\n"
                        f"Category: {category}\n"
                        f"Health Aspects: {health_aspects}\n"
                        f"Emergency Assessment: {emergency_assessment}\n"
                        f"Query:\n{json.dumps(query, indent=2)}")
            
            # Execute search
            results = self.es.search(index=index_name, body=query)
            hits = results['hits']['hits']
            
            # Log results for medical audit trail
            logging.info(f"Found {len(hits)} results in refined health information search")
            
            return hits
            
        except Exception as e:
            logging.error(f"Error in refined health information search: {str(e)}")
            logging.error(f"Query that caused error: {json.dumps(query, indent=2)}")
            return []

    def _clean_json_response(self, response: str) -> str:
        """Clean GPT response to ensure valid JSON"""
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
            return "{}"

    def generate_safe_health_fallback(self) -> str:
        """Generate safe fallback response for health-related queries"""
        # Return string instead of dict
        return ("For health-related questions, I need specific information. "
            "Could you tell me:\n"
            "• What symptoms are you observing?\n"
            "• How long has this been a concern?\n"
            "• Have you noticed any specific patterns?")
        
    def _is_just_age_response(self, message: str) -> bool:
        """Check if message is just an age response"""
        # Clean up common typos
        message = message.lower().strip().replace('monhts', 'months')
        
        # Patterns that match just age responses
        age_only_patterns = [
            r"^\d+\s*-?\s*\d*\s*(month|months|mo|year|years|yr)s?$",
            r"^(one|two|three|four|five)\s*(month|months|year|years)s?$"
        ]
        
        # Check if message matches any age-only pattern
        return any(re.match(pattern, message) for pattern in age_only_patterns)
