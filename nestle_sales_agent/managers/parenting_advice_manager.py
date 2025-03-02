# managers/parenting_advice_manager.py

from typing import Dict, List, Any, Optional
import json
import logging
import time
import chromadb
from langchain_openai import OpenAIEmbeddings

from config.config import Config
from models.user_state import UserState
from utils.error_handler import ConversationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ParentingAdviceManager:
    """
    Handles parenting advice with special attention to sensitivity, 
    developmental stages, and appropriate disclaimers.
    """
    
    def __init__(self, conversation_manager):
        self.config = Config()
        self.manager = conversation_manager
        self.max_items_per_response = self.config.MAX_ITEMS_PER_RESPONSE
        self.CATEGORY = "parenting"
        
        # Initialize sensitivity thresholds
        self.sensitivity_thresholds = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8
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

    def handle_parenting_advice(self, user_message: str, age_group: str, requery: bool) -> str:
        """
        Main entry point for all parenting advice interactions. This method orchestrates
        the complete advice recommendation process using our two-step search approach,
        while maintaining strict sensitivity awareness and appropriate disclaimers.
        
        The method follows this carefully controlled process:
        1. Verify age and check topic sensitivity
        2. Get ReACT decision to understand user needs
        3. If new advice needed:
        - Step 1: Get all potential advice (base_search)
        - Step 2: Rank and return best matches (get_next_advice_batch)
        4. Handle ongoing discussions with appropriate disclaimers
        
        Args:
            user_message: The user's input message
            age_group: User's child age group (e.g., "0-6 months")
            requery: Whether this is a repeated search
            
        Returns:
            str: Formatted response with parenting advice and appropriate disclaimers
        """
        try:
            # First verify we have required age information
            if not age_group:
                logging.info("No age group provided for parenting advice")
                return ("Could you please tell me your child's age? "
                    "This helps me provide age-appropriate parenting guidance.")

            # First check topic sensitivity before proceeding
            logging.info("Checking topic sensitivity")
            sensitivity_check = self.check_parenting_topic_sensitivity(
                user_message=user_message,
                age_group=age_group
            )
            
            # Log sensitivity assessment for tracking
            #self.log_sensitivity_check(sensitivity_check)
            
            # Handle highly sensitive topics requiring professional help
            if sensitivity_check["requires_professional"]:
                logging.info("Topic requires professional guidance")
                return self.generate_professional_referral_response(
                    topic=sensitivity_check["topic"],
                    reasoning=sensitivity_check["reasoning"]
                )

            # Get ReACT decision about user's intent
            logging.info(f"Getting ReACT decision for parenting advice - Age: {age_group}")
            react_decision = self.get_parenting_react_decision(
                user_message=user_message,
                age_group=age_group,
                sensitivity_check=sensitivity_check
            )
            
            # Generate appropriate disclaimers based on topic
            disclaimers = self.generate_parenting_disclaimers(
                parenting_aspects=react_decision["parenting_aspects"],
                sensitivity_check=sensitivity_check
            )

            # Only proceed with high-confidence decisions
            if react_decision["confidence"] >= self.config.REACT_CONFIDENCE_THRESHOLD:
                if react_decision["decision"] == "new_advice":
                    # PATH 1: NEW ADVICE SEARCH
                    logging.info("Starting two-step parenting advice search process")
                    
                    # Step 1: Get complete pool of age-appropriate advice
                    base_results = self.base_parenting_search(age_group)
                    if not base_results:
                        logging.info(f"No advice found for age group: {age_group}")
                        return ("I couldn't find any specific advice for this age group. "
                            "Would you like to explore different parenting topics?")
                    
                    # Store complete result set for pagination
                    self.manager.user_state.all_available_advice = base_results
                    self.manager.user_state.advice_current_page = 0
                    logging.info(f"Stored {len(base_results)} advice items for pagination")
                    
                    # Step 2: Get first batch using similarity ranking
                    response = self.get_next_advice_batch(user_message, react_decision)
                    
                    # Update developmental context tracking
                    self.update_developmental_context(
                        age_group=age_group,
                        developmental_areas=react_decision["parenting_aspects"].get(
                            "developmental_areas", []
                        )
                    )
                    
                    logging.info("Generated parenting advice with appropriate disclaimers")
                    return f"{response}\n\n{disclaimers}"
                    
                elif react_decision["decision"] == "continue_discussion":
                    # PATH 2: ONGOING PARENTING DISCUSSION
                    logging.info("Continuing existing parenting discussion")
                    response = self.handle_parenting_discussion(
                        user_message=user_message,
                        react_decision=react_decision
                    )
                    return f"{response}\n\n{disclaimers}"
                    
                else:
                    # PATH 3: TOPIC CHANGE NEEDED
                    logging.info("Suggesting parenting topic change based on ReACT decision")
                    return self.suggest_parenting_topic_change(react_decision["reasoning"])
            
            # Fall back to basic handling if low confidence
            logging.info("Falling back to basic parenting advice handling due to low confidence")
            return self.fallback_parenting_handling(
                user_message=user_message,
                age_group=age_group,
                requery=requery
            )
            
        except Exception as e:
            # Log error details and fall back to safe handling
            logging.error(f"Error in parenting advice handling: {str(e)}", exc_info=True)
            return self.generate_safe_parenting_fallback("Parenting Advice")

    def fallback_parenting_handling(self, user_message: str, age_group: str, requery: bool) -> str:
        """
        Handle parenting advice when primary method fails. Uses a simplified search approach
        that focuses on general age-appropriate parenting guidance.
        
        Args:
            user_message: User's original request
            age_group: Child's age group (e.g., "18-24 months")
            requery: Whether this is a repeated search attempt
            
        Returns:
            str: Formatted parenting advice response with appropriate disclaimers
        """
        try:
            # Perform a basic search using age group and any available parenting aspects
            basic_search = self.search_parenting_advice(
                age_group=age_group, 
                parenting_aspects={"primary_concern": "general guidance"}
            )
            
            if basic_search:
                # Generate a presentation with minimal context but appropriate safety
                return self.generate_parenting_advice_presentation(
                    selected_advice=basic_search[:self.max_items_per_response],
                    user_message=user_message,
                    react_decision={
                        "parenting_aspects": {
                            "primary_concern": "general guidance",
                            "developmental_areas": [],
                            "emotional_components": []
                        },
                        "approach_suggestions": ["provide general guidance"],
                        "sensitivity_considerations": [],
                        "reasoning": "Fallback guidance provided"
                    }
                )
                
            # If no advice found, return safe fallback message
            return self.generate_safe_parenting_fallback("Parenting Advice")
            
        except Exception as e:
            logging.error(f"Error in parenting advice fallback handling: {str(e)}")
            return self.generate_safe_parenting_fallback("Parenting Advice")

    def check_parenting_topic_sensitivity(self, user_message: str, age_group: str) -> Dict:
        """Analyze topic sensitivity and professional referral needs"""
        analysis_prompt = f"""
        Analyze parenting topic sensitivity:
        Message: {user_message}
        Age Group: {age_group}
        
        Return JSON with:
        {{
            "topic": "main topic",
            "sensitivity_level": "low/medium/high",
            "requires_professional": boolean,
            "safety_concerns": ["concern1", "concern2"],
            "reasoning": "explanation"
        }}
        """
        
        try:
            response = self.manager.get_gpt_response(analysis_prompt)
            cleaned_response = self._clean_json_response(response)
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            logging.error("Failed to parse sensitivity check")
            return {
                "topic": "general parenting",
                "sensitivity_level": "high",
                "requires_professional": False,
                "safety_concerns": [],
                "reasoning": "Fallback due to parsing error"
            }

    def get_parenting_react_decision(self, user_message: str, age_group: str, sensitivity_check: Dict) -> Dict:
        """
        Gets ReACT reasoning for parenting advice with enhanced error handling and data validation.
        
        This method processes the user's message through the ReACT framework to determine how to
        handle parenting advice requests while maintaining sensitivity awareness and proper
        data serialization.
        
        Args:
            user_message: The user's input message
            age_group: Child's age group (e.g., "12-18 months")
            sensitivity_check: Dictionary containing sensitivity analysis results
            
        Returns:
            Dict: Structured decision with parenting aspects and approach suggestions
        """
        try:
            # Convert sets to lists for JSON serialization
            shown_advice = list(self.manager.user_state.shown_advice) if hasattr(self.manager.user_state, 'shown_advice') else []
            
            # Safely get current topics with type checking
            current_topics = []
            if hasattr(self.manager.user_state, 'current_advice_topics'):
                if isinstance(self.manager.user_state.current_advice_topics, (list, set)):
                    current_topics = list(self.manager.user_state.current_advice_topics)
                elif isinstance(self.manager.user_state.current_advice_topics, dict):
                    current_topics = list(self.manager.user_state.current_advice_topics.keys())
                    
            # Safely convert developmental context to serializable format
            dev_context = {}
            if hasattr(self.manager.user_state, 'developmental_context'):
                for key, value in self.manager.user_state.developmental_context.items():
                    if isinstance(value, set):
                        dev_context[key] = list(value)
                    elif isinstance(value, dict):
                        dev_context[key] = {
                            k: list(v) if isinstance(v, set) else v 
                            for k, v in value.items()
                        }
                    else:
                        dev_context[key] = value

            # Construct the prompt with properly serialized data
            react_prompt = f"""
            Task: Determine how to handle parenting advice request.
            
            Context:
            - User Message: {user_message}
            - Child's Age Group: {age_group}
            - Current Topics: {json.dumps(current_topics)}
            - Previously Shown: {json.dumps(shown_advice)}
            - Sensitivity Level: {sensitivity_check.get("sensitivity_level", "medium")}
            - Developmental Context: {json.dumps(dev_context)}
            
            You must respond with ONLY valid JSON in exactly this format:
            {{
                "decision": "new_advice/continue_discussion/change_topic",
                "confidence": 0.0-1.0,
                "parenting_aspects": {{
                    "primary_concern": "description",
                    "developmental_areas": ["area1", "area2"],
                    "emotional_components": ["component1", "component2"]
                }},
                "approach_suggestions": ["suggestion1", "suggestion2"],
                "sensitivity_considerations": ["consideration1", "consideration2"],
                "reasoning": "detailed explanation"
            }}
            """
            
            # Get and validate GPT response
            try:
                response = self.manager.get_gpt_response(react_prompt)
                if not response:
                    raise ValueError("Empty response from GPT")
                    
                # Clean and parse the response
                cleaned_response = self._clean_json_response(response)
                parsed_response = json.loads(cleaned_response)
                
                # Validate required fields and data types
                validation_result = self._validate_react_decision(parsed_response)
                if not validation_result["is_valid"]:
                    logging.warning(f"Invalid parsed response: {validation_result['reason']}")
                    return self._get_fallback_decision(f"Validation failed: {validation_result['reason']}")
                    
                # Validate confidence score range
                confidence = float(parsed_response["confidence"])
                if not 0.0 <= confidence <= 1.0:
                    logging.warning(f"Invalid confidence score: {confidence}")
                    parsed_response["confidence"] = max(0.0, min(1.0, confidence))
                
                return parsed_response
                
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error: {str(e)}\nResponse: {response}")
                return self._get_fallback_decision("JSON parsing error")
                
        except Exception as e:
            logging.error(f"Error in get_parenting_react_decision: {str(e)}", exc_info=True)
            return self._get_fallback_decision("General error")
    
    def _get_fallback_decision(self, reason: str) -> Dict:
        """
        Generates a safe fallback decision when errors occur.
        
        Args:
            reason: The reason for falling back to default decision
            
        Returns:
            Dict containing a safe default decision
        """
        logging.info(f"Using fallback decision due to: {reason}")
        return {
            "decision": "new_advice",
            "confidence": 0.5,
            "parenting_aspects": {
                "primary_concern": "general parenting guidance",
                "developmental_areas": [],
                "emotional_components": []
            },
            "approach_suggestions": ["provide general guidance"],
            "sensitivity_considerations": ["proceed with standard guidance"],
            "reasoning": f"Fallback due to {reason}"
        }

    def _validate_react_decision(self, decision: Dict) -> Dict:
        """
        Validates the structure and content of a ReACT decision.
        
        Args:
            decision: The parsed decision dictionary to validate
            
        Returns:
            Dict containing validation result and reason if invalid
        """
        required_fields = {
            "decision": str,
            "confidence": (int, float),
            "parenting_aspects": dict,
            "approach_suggestions": list,
            "sensitivity_considerations": list,
            "reasoning": str
        }
        
        # Check required fields and types
        for field, expected_type in required_fields.items():
            if field not in decision:
                return {"is_valid": False, "reason": f"Missing field: {field}"}
            if not isinstance(decision[field], expected_type):
                return {"is_valid": False, "reason": f"Invalid type for {field}"}
                
        # Validate decision values
        valid_decisions = {"new_advice", "continue_discussion", "change_topic"}
        if decision["decision"] not in valid_decisions:
            return {"is_valid": False, "reason": f"Invalid decision value: {decision['decision']}"}
            
        # Validate parenting aspects structure
        required_aspects = {"primary_concern", "developmental_areas", "emotional_components"}
        if not all(aspect in decision["parenting_aspects"] for aspect in required_aspects):
            return {"is_valid": False, "reason": "Invalid parenting_aspects structure"}
            
        return {"is_valid": True, "reason": None}

    def handle_new_parenting_advice(self, user_message: str, age_group: str, react_decision: Dict) -> str:
        """
        Handles new parenting advice requests using a two-step search process to ensure relevant,
        developmentally appropriate advice while maintaining sensitivity considerations.
        """
        try:
            # First, verify we have the age group for appropriate advice
            if not age_group:
                return "Could you please tell me your child's age? This helps me provide age-appropriate guidance."

            # Step 1: Perform initial base search by age and category only
            logging.info(f"Performing base parenting advice search for age group: {age_group}")
            base_results = self.base_parenting_search(age_group)
            
            if not base_results:
                return ("I couldn't find any advice matching your child's age group. " 
                    "Would you like to explore different parenting topics?")
            
            # Store complete result set for pagination
            self.manager.user_state.all_available_advice = base_results
            self.manager.user_state.advice_current_page = 0
            
            # Step 2: Get the first batch of advice with similarity ranking
            return self.get_next_advice_batch(user_message, react_decision)
                
        except Exception as e:
            logging.error(f"Error in parenting advice search: {str(e)}")
            return self.generate_safe_parenting_fallback("Parenting Advice")

    def get_next_advice_batch(self, user_message: str, react_decision: Dict) -> str:
        """
        Get next batch of parenting advice using direct selection from available documents.
        """
        logging.info("Getting next parenting advice batch")
        
        # Get our working set
        all_advice = self.manager.user_state.all_available_advice
        shown = self.manager.user_state.shown_advice
        current_page = self.manager.user_state.advice_current_page
        
        if not all_advice:
            logging.error("No advice available in state")
            return self.generate_safe_parenting_fallback("Advice Batch Retrieval")
        
        # Filter out shown advice
        available = [
            advice for advice in all_advice 
            if str(advice.get('_id', '')) not in shown
        ]
        
        if not available:
            return "You've seen all available advice. Would you like to explore different topics?"
        
        # CHANGED: Select documents directly without similarity search
        # Take first 3 documents or all if less than 3
        selected_advice = available[:min(3, len(available))]
        logging.info(f"Directly selected {len(selected_advice)} advice documents")
        
        # Update state
        try:
            self.update_parenting_state(selected_advice)
            self.manager.user_state.advice_current_page += 1
        except Exception as e:
            logging.error(f"Error updating advice state: {str(e)}")
        
        # Generate response with appropriate disclaimers
        response = self.generate_parenting_advice_presentation(
            selected_advice=selected_advice,
            user_message=user_message,
            react_decision=react_decision
        )
        
        # Add pagination prompt if more advice available
        if len(available) > 3:
            response += "\n\nWould you like to see more advice on this topic?"
        
        return response

    def _is_advice_appropriate(self, advice: Dict, react_decision: Dict) -> bool:
        """Determine if advice is appropriate to show based on sensitivity and context"""
        try:
            # Check sensitivity levels
            sensitivity_level = advice.get('sensitivity_level', 'medium')
            if sensitivity_level == 'high' and not react_decision.get("high_sensitivity_approved"):
                return False
            
            # Extract developmental areas with safety checks
            advice_areas = advice.get('developmental_areas', [])
            target_areas = react_decision.get("parenting_aspects", {}).get("developmental_areas", [])
            
            # If either is empty, consider it appropriate
            if not advice_areas or not target_areas:
                return True
                
            # Do the developmental appropriateness check
            return len(set(advice_areas).intersection(set(target_areas))) > 0
            
        except Exception as e:
            logging.error(f"Error checking advice appropriateness: {str(e)}")

    def _is_developmentally_appropriate(self, advice_areas: List[str], target_areas: List[str]) -> bool:
        """
        Determines if advice is developmentally appropriate by comparing developmental areas.
        
        Args:
            advice_areas: List of developmental areas mentioned in the advice
            target_areas: List of developmental areas we're looking to address
            
        Returns:
            bool: True if the advice is developmentally appropriate
        """
        try:
            # If no specific areas are provided, consider it appropriate
            if not target_areas or not advice_areas:
                return True
                
            # Check for at least one matching developmental area
            matching_areas = set(advice_areas).intersection(set(target_areas))
            return len(matching_areas) > 0
            
        except Exception as e:
            logging.error(f"Error checking developmental appropriateness: {str(e)}")
            return True  # Default to including advice if check fails

    def update_parenting_state(self, advice_items: List[Dict]) -> None:
        """Update state with new parenting advice"""
        try:
            if not hasattr(self.manager.user_state, 'shown_advice'):
                self.manager.user_state.shown_advice = set()
                
            for advice in advice_items:
                # Use stored ID or generate a new one
                advice_id = advice.get('_id') or str(hash(frozenset(str(item) for item in advice.items())))
                self.manager.user_state.shown_advice.add(str(advice_id))
            
            if not hasattr(self.manager.user_state, 'current_advice'):
                self.manager.user_state.current_advice = []
                
            self.manager.user_state.current_advice = advice_items
            self.manager.user_state.last_advice_search = time.time()
            
            logging.info(f"Updated state with {len(advice_items)} new advice items")
            
        except Exception as e:
            logging.error(f"Error in advice state update: {str(e)}", exc_info=True)

    def _calculate_final_advice_score(self, base_score: float, advice: Dict, react_decision: Dict) -> float:
        """Adjust base similarity score with sensitivity and developmental considerations"""
        score = base_score
        
        # Boost score for matching developmental areas
        dev_areas = react_decision["parenting_aspects"].get("developmental_areas", [])
        matching_areas = set(dev_areas).intersection(
            set(advice.get("developmental_areas", []))
        )
        score += len(matching_areas) * 0.2
        
        # Boost for matching emotional components
        emotional_components = react_decision["parenting_aspects"].get("emotional_components", [])
        if any(comp in advice.get("content", "") for comp in emotional_components):
            score += 0.15
        
        # Penalty for high sensitivity if not specifically requested
        if advice.get("sensitivity_level", "low") == "high" and \
        react_decision.get("sensitivity_considerations", []):
            score *= 0.8
        
        return min(1.0, score)  # Cap at 1.0

    def get_advice_text(self, advice: Dict) -> str:
        """
        Creates a searchable text representation of parenting advice,
        incorporating all relevant aspects for similarity matching.
        """
        return " ".join([
            advice.get("title", ""),
            advice.get("content", ""),
            " ".join(advice.get("developmental_areas", [])),
            " ".join(advice.get("approaches", [])),
            advice.get("age_group", ""),
            " ".join(advice.get("key_points", [])),
            advice.get("context", "")
        ]).lower()

    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates the cosine similarity between two texts using TF-IDF vectorization.
        Returns a similarity score between 0 and 1.
        """
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

    def get_random_advice(self, advice_items: List[Dict], count: int) -> List[Dict]:
        """Get random advice when no specific matching needed"""
        import random
        if len(advice_items) <= count:
            return advice_items
        return random.sample(advice_items, count)

    def handle_parenting_discussion(self, user_message: str, react_decision: Dict) -> str:
        """
        Handles ongoing parenting discussions by providing contextual responses 
        based on the current conversation thread.
        
        Args:
            user_message: The user's current message
            react_decision: The ReACT analysis containing discussion context
            
        Returns:
            str: Formatted response continuing the parenting discussion
        """
        try:
            # Extract relevant aspects from the react decision
            parenting_aspects = react_decision["parenting_aspects"]
            approach_suggestions = react_decision["approach_suggestions"]
            
            # Generate discussion prompt
            discussion_prompt = f"""
            Continue parenting discussion about {parenting_aspects['primary_concern']}.
            
            Context:
            - Developmental Areas: {', '.join(parenting_aspects['developmental_areas'])}
            - Emotional Components: {', '.join(parenting_aspects['emotional_components'])}
            - Current Approach Suggestions: {', '.join(approach_suggestions)}
            
            User Message: {user_message}
            
            Provide a response that:
            1. Addresses their current question/concern
            2. Builds on previous discussion
            3. Offers practical, age-appropriate guidance
            4. Maintains a supportive, encouraging tone
            """
            
            # Get response from GPT
            initial_response = self.manager.get_gpt_response(discussion_prompt)
            
            # Validate response
            validated_response = self.ensure_hallucination_free_response(
                initial_response,
                react_decision,
                "Parenting Discussion"
            )
            
            if not validated_response:
                return self.generate_safe_parenting_fallback("Parenting Discussion")
                
            return validated_response
            
        except Exception as e:
            logging.error(f"Error handling parenting discussion: {str(e)}")
            return self.generate_safe_parenting_fallback("Parenting Discussion")

    def base_parenting_search(self, age_group: str) -> List[Dict]:
        """
        Base search for parenting advice using ChromaDB's vector similarity.
        CHANGED: Now uses ChromaDB instead of Elasticsearch
        """
        logging.info("\n=== Step 1: Base Parenting Advice Search ===")
        logging.info(f"Search Parameters:")
        logging.info(f"Category: '{self.CATEGORY}'")
        logging.info(f"Age Group: '{age_group}'")
        
        try:
            # First check if collection exists
            try:
                all_collections = self.client.list_collections()
                logging.info(f"Available collections: {all_collections}")
                
                if self.collection.name not in all_collections:
                    logging.warning(f"Collection '{self.collection.name}' not found in ChromaDB!")
            except Exception as e:
                logging.error(f"Error checking collections: {str(e)}")
            
            # Create search text and get embedding
            search_text = f"parenting advice for {age_group}"
            logging.info(f"Creating embedding for search text: '{search_text}'")
            query_embedding = self.embeddings.embed_query(search_text)
            
            # First try exact age group match
            logging.info(f"Querying with exact age_group: {age_group}")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=10,
                where={"age_group": age_group},
                include=["metadatas", "documents", "distances"]
            )
            
            # Check if we got any results
            document_count = len(results["documents"][0]) if "documents" in results and results["documents"] and results["documents"][0] else 0
            metadata_count = len(results["metadatas"][0]) if "metadatas" in results and results["metadatas"] and results["metadatas"][0] else 0
            
            logging.info(f"ChromaDB returned {document_count} documents and {metadata_count} metadata items")
            
            # If no exact matches, try a general search without age_group filter
            if document_count == 0:
                logging.info(f"No exact matches for age_group={age_group}, trying relaxed search")
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=10,
                    include=["metadatas", "documents", "distances"]
                )
                document_count = len(results["documents"][0]) if "documents" in results and results["documents"] and results["documents"][0] else 0
                metadata_count = len(results["metadatas"][0]) if "metadatas" in results and results["metadatas"] and results["metadatas"][0] else 0
                logging.info(f"Relaxed search returned {document_count} documents and {metadata_count} metadata items")
            
            # Process and validate results
            processed_advice = []
            
            if "documents" in results and results["documents"] and results["documents"][0] and "metadatas" in results and results["metadatas"] and results["metadatas"][0]:
                for i in range(len(results["documents"][0])):
                    try:
                        doc = results["documents"][0][i]
                        metadata = results["metadatas"][0][i]
                        
                        # Process document content
                        if isinstance(doc, str):
                            try:
                                content = json.loads(doc)
                            except json.JSONDecodeError:
                                content = {"content": doc}
                        elif isinstance(doc, dict):
                            content = doc
                        else:
                            content = {"content": str(doc)}
                        
                        # Update with metadata
                        content.update(metadata)
                        
                        # Add ID if missing
                        if "_id" not in content and "ids" in results and results["ids"] and results["ids"][0]:
                            content["_id"] = results["ids"][0][i]
                        
                        # Add to processed results
                        processed_advice.append(content)
                        
                    except Exception as e:
                        logging.error(f"Error processing advice item at index {i}: {str(e)}")
                        continue
            
            logging.info(f"Found {len(processed_advice)} validated advice items")
            return processed_advice
            
        except Exception as e:
            logging.error(f"Base parenting advice search failed: {str(e)}", exc_info=True)
            return []
    
    def _perform_parenting_diagnostic_search(self, age_group: str) -> List[Dict]:
        """
        Performs a broader diagnostic search when exact parenting advice matching fails.
        This helps identify potential issues with content categorization or age group matching,
        while being mindful of sensitivity levels in parenting content.
        """
        diagnostic_query = {
            "query": {
                "bool": {
                    "should": [
                        # Look for partial matches on parenting advice category
                        {"match": {"category": self.CATEGORY}},
                        # Look for partial matches on age group
                        {"match": {"age_group": age_group}},
                        # Look for advice in similar age ranges
                        {"prefix": {"age_group": age_group.split()[0]}}
                    ],
                    "minimum_should_match": 1,
                    "must_not": [
                        # Exclude high-sensitivity content during diagnostic search
                        {"term": {"sensitivity_level.keyword": "high"}}
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
            logging.error(f"Parenting advice diagnostic search failed: {str(e)}")
            return []

    def _process_parenting_advice_results(self, hits: List[Dict]) -> List[Dict]:
        """
        Processes and validates parenting advice search results with flexible validation.
        """
        processed_advice = []
        core_required_fields = {'title', 'content'}  # Minimum required fields
        optional_fields = {'age_group', 'category', 'sensitivity_level', 'developmental_areas'}
        
        for hit in hits:
            advice = hit['_source']
            
            # Check for minimum required fields
            if all(field in advice for field in core_required_fields):
                # Add default values for missing optional fields
                if 'age_group' not in advice:
                    advice['age_group'] = self.manager.user_state.age_group
                if 'category' not in advice:
                    advice['category'] = self.CATEGORY
                if 'sensitivity_level' not in advice:
                    advice['sensitivity_level'] = 'medium'
                if 'developmental_areas' not in advice:
                    advice['developmental_areas'] = []
                    
                processed_advice.append(advice)
            else:
                logging.warning(
                    f"Skipping incomplete advice: {advice.get('title', 'Unknown')} - "
                    f"Missing required fields: {core_required_fields - set(advice.keys())}"
                )
        
        return processed_advice

    def search_parenting_advice(self, age_group: str, parenting_aspects: Dict) -> List[Dict]:
        """Enhanced parenting advice search using Elasticsearch"""
        try:
            # Convert aspects to search terms
            search_aspects = []
            if parenting_aspects.get("primary_concern"):
                search_aspects.append(parenting_aspects["primary_concern"])
            if parenting_aspects.get("developmental_areas"):
                search_aspects.extend(parenting_aspects["developmental_areas"])
            
            search_title = " ".join(search_aspects)
            
            results = self.base_parenting_search(age_group)
            
            return [hit['_source'] for hit in results]
            
        except Exception as e:
            logging.error(f"Error in parenting advice search: {str(e)}")
            return []

    def update_developmental_context(self, age_group: str, developmental_areas: List[str]) -> None:
        """Update stored developmental context with new insights"""
        context = self.manager.user_state.developmental_context
        
        if age_group not in context:
            context[age_group] = {
                "areas_discussed": set(),
                "last_updated": time.time()
            }
            
        context[age_group]["areas_discussed"].update(developmental_areas)
        context[age_group]["last_updated"] = time.time()

    def generate_professional_referral_response(self, topic: str, reasoning: str) -> str:
        """Generate response recommending professional consultation"""
        return f"""
        Based on your question about {topic}, I recommend consulting with a healthcare provider 
        or child development specialist who can provide personalized guidance.
        
        While I can offer general information, a professional can:
        • Evaluate your specific situation
        • Provide personalized recommendations
        • Monitor progress over time
        • Address any specific concerns
        
        Would you like me to provide some general information about this topic in the meantime?
        """

    def generate_parenting_disclaimers(self, parenting_aspects: Dict, 
                                     sensitivity_check: Dict) -> str:
        """Generate appropriate disclaimers based on topic sensitivity"""
        disclaimers = ["This advice is for informational purposes only."]
        
        if sensitivity_check["sensitivity_level"] == "high":
            disclaimers.append(
                "Consider consulting with a child development specialist for personalized guidance."
            )
        
        if parenting_aspects.get("developmental_areas"):
            disclaimers.append(
                "Every child develops at their own pace. Consult your pediatrician with specific concerns."
            )
            
        return "\n\n" + "\n".join(disclaimers)

    def select_advice_for_presentation(self, available_advice: List[Dict], user_message: str, react_decision: Dict) -> List[Dict]:
        """Select most relevant parenting advice"""
        sorted_advice = sorted(
            available_advice,
            key=lambda a: self._calculate_advice_relevance(a, user_message, react_decision),
            reverse=True
        )
        
        return sorted_advice[:self.max_items_per_response]

    def _calculate_advice_relevance(self, advice: Dict, user_message: str, 
                                  react_decision: Dict) -> float:
        """Calculate relevance score for parenting advice"""
        score = 0.0
        
        # Match developmental areas
        dev_areas = react_decision["parenting_aspects"].get("developmental_areas", [])
        matching_areas = set(dev_areas).intersection(
            set(advice.get("developmental_areas", []))
        )
        score += len(matching_areas) * 0.3
        
        # Consider emotional components
        emotional_components = react_decision["parenting_aspects"].get("emotional_components", [])
        if any(comp in advice.get("content", "") for comp in emotional_components):
            score += 0.2
            
        # Keyword matching
        keywords = set(user_message.lower().split())
        advice_text = advice.get("content", "").lower()
        matching_keywords = keywords.intersection(set(advice_text.split()))
        score += len(matching_keywords) * 0.1
        
        return min(1.0, score)

    def generate_parenting_advice_presentation(self, selected_advice: List[Dict], user_message: str, react_decision: Dict) -> str:
        """Generate parenting advice presentation"""
        try:
            presentation_prompt = f"""
            Create parenting advice response:
            Advice: {json.dumps(selected_advice)}
            User Message: {user_message}
            Analysis: {json.dumps(react_decision)}
            
            Response should:
            1. Address the specific concern
            2. Provide practical guidance
            3. Consider emotional aspects
            4. Maintain supportive tone
            5. Include relevant developmental context
            6. Reference appropriate disclaimers
            """
            
            initial_response = self.manager.get_gpt_response(presentation_prompt)
            validated_response = self.ensure_hallucination_free_response(
                initial_response,
                selected_advice,
                "Parenting Advice"
            )
            
            if not validated_response:
                return self.generate_safe_parenting_fallback("Parenting Advice")
                
            disclaimers = self.generate_parenting_disclaimers(
                react_decision["parenting_aspects"],
                {"sensitivity_level": "medium"}  # Safe default
            )
            
            return f"{validated_response}\n\n{disclaimers}"
            
        except Exception as e:
            logging.error(f"Error in parenting advice presentation: {str(e)}")
            return self.generate_safe_parenting_fallback("Parenting Advice")

    def ensure_hallucination_free_response(self, response: str, context_data: Dict, category: str) -> str:
        """Verify response against context data"""
        try:
            verification_prompt = f"""
            Task: Verify parenting advice accuracy.

            Response: {response}
            Original Data: {json.dumps(context_data)}
            Category: {category}

            Verify that:
            1. All mentioned advice exists in data
            2. All claims are supported
            3. Safety considerations maintained

            Return only "Yes" if verified, "No" if not.
            """
            
            verification = self.manager.get_gpt_response(verification_prompt).strip().lower()
            
            if verification == "yes":
                return response
                
            return None
            
        except Exception as e:
            logging.error(f"Error in response verification: {str(e)}")
            return None

    def debug_search(self, index_name: str, category: str, age_group: str, parenting_aspects: Optional[Dict] = None, sensitivity_level: Optional[str] = None) -> List[Dict]:
        """
        Performs a refined parenting advice search that considers developmental stages,
        sensitivity levels, and specific parenting concerns. This function implements
        step 2 of our search process, applying more sophisticated filtering and scoring.
        
        Args:
            index_name: Name of the Elasticsearch index
            category: Category (should be "Parenting Advice")
            age_group: Age group to search within
            parenting_aspects: Dictionary containing developmental areas and concerns
            sensitivity_level: Optional sensitivity threshold to apply
            
        Returns:
            List[Dict]: List of matching parenting advice items
        """
        # Start with base criteria
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"age_group.keyword": age_group}},
                        {"term": {"category.keyword": self.CATEGORY}}
                    ],
                    "should": [],  # Initialize scoring criteria
                    "must_not": []  # Initialize exclusion criteria
                }
            }
        }
        
        # Add sensitivity filtering if specified
        if sensitivity_level:
            threshold = self.sensitivity_thresholds.get(sensitivity_level, 0.5)
            query["query"]["bool"]["must_not"].append({
                "range": {
                    "sensitivity_score": {
                        "gt": threshold  # Exclude items above sensitivity threshold
                    }
                }
            })
        
        # Add developmental area matching if provided
        if parenting_aspects and parenting_aspects.get("developmental_areas"):
            # High boost for developmental area matches
            query["query"]["bool"]["should"].extend([
                {
                    "match": {
                        "developmental_areas": {
                            "query": area,
                            "boost": 3.0  # Prioritize developmental matches
                        }
                    }
                }
                for area in parenting_aspects["developmental_areas"]
            ])
        
        # Add emotional component matching if provided
        if parenting_aspects and parenting_aspects.get("emotional_components"):
            # Medium boost for emotional component matches
            query["query"]["bool"]["should"].extend([
                {
                    "match": {
                        "emotional_aspects": {
                            "query": component,
                            "boost": 2.0
                        }
                    }
                }
                for component in parenting_aspects["emotional_components"]
            ])
        
        # Add general content matching with appropriate weights
        if parenting_aspects and parenting_aspects.get("primary_concern"):
            query["query"]["bool"]["should"].extend([
                {"match": {"title": {"query": parenting_aspects["primary_concern"], "boost": 2.0}}},
                {"match": {"content": {"query": parenting_aspects["primary_concern"], "boost": 1.5}}},
                {"match": {"key_points": {"query": parenting_aspects["primary_concern"], "boost": 1.0}}}
            ])
        
        # Require at least one should clause to match if we have any
        if query["query"]["bool"]["should"]:
            query["query"]["bool"]["minimum_should_match"] = 1
        
        try:
            # Log the search attempt
            logging.info(f"\n=== Executing Refined Parenting Advice Search ===\n"
                        f"Age Group: {age_group}\n"
                        f"Category: {category}\n"
                        f"Parenting Aspects: {parenting_aspects}\n"
                        f"Sensitivity Level: {sensitivity_level}\n"
                        f"Query:\n{json.dumps(query, indent=2)}")
            
            # Execute search
            results = self.es.search(index=index_name, body=query)
            hits = results['hits']['hits']
            
            # Log results summary
            logging.info(f"Found {len(hits)} results in refined parenting advice search")
            
            return hits
            
        except Exception as e:
            logging.error(f"Error in refined parenting advice search: {str(e)}")
            logging.error(f"Query that caused error: {json.dumps(query, indent=2)}")
            return []

    def suggest_parenting_topic_change(self, reasoning: str) -> str:
        """Suggest alternative parenting topics"""
        return f"""
        Based on our discussion, {reasoning}
        
        Would you like to:
        • Explore different parenting approaches?
        • Learn about child development milestones?
        • Discuss general parenting strategies?
        """

    def fallback_parenting_advice_selection(self, advice_data: List[Dict], user_message: str) -> str:
        """Handle advice selection when primary method fails"""
        try:
            if advice_data:
                basic_advice = advice_data[:self.max_items_per_response]
                return self.generate_parenting_advice_presentation(
                    basic_advice,
                    user_message,
                    {"parenting_aspects": {"primary_concern": "general advice"}}
                )
            return self.generate_safe_parenting_fallback("Parenting Advice")
        except Exception as e:
            logging.error(f"Error in fallback advice selection: {str(e)}")
            return self.generate_safe_parenting_fallback("Parenting Advice")
    
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

    def generate_safe_parenting_fallback(self, category: str) -> str:
        """Generate safe fallback response"""
        return ("I'd like to help you with your parenting question. Could you provide more details about:\n"
                "• Your specific concern\n"
                "• Your child's age\n"
                "• Any relevant background information")
        
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
        
    def _meets_medical_safety_requirements(self, content: Dict) -> bool:
        """
        Verifies that advice content meets safety requirements.
        This is a basic implementation that should be expanded based on your needs.
        """
        # Check for minimum required content
        if not content:
            return False
            
        # Check for content or title
        if not content.get("content") and not content.get("title"):
            return False
        
        # Check for explicitly marked unsafe content
        if content.get("unsafe", False):
            return False
        
        return True
    
    def generate_fallback_advice(self, age_group: str) -> str:
        """Generate advice using LLM when no database matches found"""
        prompt = f"""
        Generate helpful, evidence-based parenting advice for a {age_group} child.
        
        Include:
        1. Age-appropriate developmental information
        2. Common challenges at this age
        3. Practical parenting strategies
        4. Safety considerations
        
        Format as a helpful, conversational response that a parent would find valuable.
        """
        
        return self.manager.get_gpt_response(prompt)