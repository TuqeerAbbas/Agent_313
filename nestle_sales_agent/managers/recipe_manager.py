# managers/recipe_manager.py

from typing import Dict, List, Any, Optional
import json
import logging
import time
import random
import chromadb
from langchain_openai import OpenAIEmbeddings

from config.config import Config
from models.user_state import UserState
from utils.error_handler import ConversationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RecipeManager:
    """Handles recipe-related functionality including recommendations and dietary considerations."""
    
    def __init__(self, conversation_manager):
        self.config = Config()
        self.manager = conversation_manager
        self.max_items_per_response = self.config.MAX_ITEMS_PER_RESPONSE
        self.CATEGORY = "recipe"

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
        
    def handle_recipe_recommendation(self, user_message: str, age_group: str, requery: bool) -> str:
        """
        Main entry point for recipe interactions. This method:
        1. Gets ReACT decision about what user wants
        2. Routes to appropriate handling method
        3. Returns formatted response with recipes or guidance
        """
        try:
            # Get ReACT decision
            react_decision = self.get_recipe_react_decision(user_message, age_group)
            
            if react_decision["confidence"] >= self.config.REACT_CONFIDENCE_THRESHOLD:
                if react_decision["decision"] == "new_search":
                    # Path 1: New Recipe Search
                    logging.info("Starting new recipe search process")
                    
                    # Update dietary preferences if needed
                    if dietary_considerations := react_decision.get("dietary_considerations"):
                        self.update_dietary_preferences(dietary_considerations)
                    
                    # Step 1: Get base recipe pool
                    base_results = self.base_search(age_group)
                    if not base_results:
                        return "I couldn't find any recipes for this age group. Would you like to explore different options?"
                    
                    # Store for pagination
                    self.manager.user_state.all_available_recipes = base_results
                    self.manager.user_state.recipe_current_page = 0
                    
                    # Step 2: Get first batch
                    return self.get_next_recipe_batch(user_message, react_decision)
                    
                elif react_decision["decision"] == "continue_discussion":
                    # Path 2: Recipe Discussion
                    return self.handle_recipe_discussion(user_message, react_decision)
                else:
                    # Path 3: Topic Change
                    return self.suggest_topic_change(react_decision["reasoning"])
            
            # Fall back to basic handling if low confidence
            return self.fallback_recipe_handling(user_message, age_group, requery)
            
        except Exception as e:
            logging.error(f"Error in recipe handling: {str(e)}", exc_info=True)
            return self.fallback_recipe_handling(user_message, age_group, requery)

    def get_recipe_react_decision(self, user_message: str, age_group: str) -> Dict:
        """Get ReACT reasoning for recipe handling with dietary awareness"""
        react_prompt = f"""
        Task: Determine how to handle recipe recommendation request.

        Context:
        - User Message: {user_message}
        - Age Group: {age_group}
        - Current Recipes: {json.dumps(self.manager.user_state.current_recipes)}
        - Previously Shown: {list(self.manager.user_state.shown_recipes)}
        - Existing Dietary Preferences: {json.dumps(self.manager.user_state.dietary_preferences)}
        
        Think through:
        1. Is user asking about current recipes or needs new recommendations?
        2. Are there new dietary restrictions/preferences mentioned?
        3. Any preparation preferences mentioned?
        4. Does this build on previous recipe discussion?
        
        Respond with structured JSON:
        {{
            "decision": "new_search/continue_discussion/change_topic",
            "confidence": 0.0-1.0,
            "dietary_considerations": ["consideration1", "consideration2"],
            "preparation_preferences": {{
                "time": "quick/medium/long",
                "complexity": "easy/medium/difficult"
            }},
            "reasoning": "explanation of decision"
        }}
        """
        
        response = self.manager.get_gpt_response(react_prompt)
        # Clean the response before parsing JSON
        cleaned_response = self._clean_json_response(response)
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            logging.error("Failed to parse GPT response as JSON")
            return {
                "decision": "new_search",
                "confidence": 0.5,
                "dietary_considerations": [],
                "preparation_preferences": {
                    "time": "medium",
                    "complexity": "easy"
                },
                "reasoning": "Fallback due to parsing error"
            }

    def _clean_json_response(self, response: str) -> str:
        """Clean GPT response to ensure valid JSON"""
        # Remove any markdown formatting
        response = response.replace('```json\n', '').replace('```', '')
        # Remove any leading/trailing whitespace
        response = response.strip()
        return response

    def handle_new_recipe_search(self, user_message: str, age_group: str, react_decision: Dict) -> str:
        try:
            if not age_group:
                return "Could you please tell me your baby's age? This helps me recommend appropriate recipes."

            # Step 1: First search using only age_group and category
            logging.info(f"Performing base recipe search for age group: {age_group}")
            base_results = self.base_search(age_group)
            
            if not base_results:
                return ("I couldn't find any recipes matching your criteria for a " + 
                    f"{age_group} child. Would you like to explore different options?")
            
            # Store all results for pagination
            self.manager.user_state.all_available_recipes = base_results
            self.manager.user_state.recipe_current_page = 0
            
            # Step 2: Get first batch with similarity ranking
            return self.get_next_recipe_batch(user_message, react_decision)
                
        except Exception as e:
            logging.error(f"Error in recipe search: {str(e)}")
            return self.generate_safe_recipe_fallback("Recipe Recommendation")

    def get_next_recipe_batch(self, user_message: str, react_decision: Dict) -> str:
        """
        Step 2: Get next batch of recipes using vector similarity.
        CHANGED: Now uses ChromaDB for similarity search
        """
        logging.info("Getting next recipe batch")
        
        # Get our working set of recipes
        all_recipes = self.manager.user_state.all_available_recipes
        shown = self.manager.user_state.shown_recipes
        current_page = self.manager.user_state.recipe_current_page
        
        if not all_recipes:
            logging.error("No recipes available in state")
            return self.generate_safe_recipe_fallback("Recipe Batch Retrieval")
        
        # Filter out shown recipes
        available = [recipe for recipe in all_recipes if str(recipe.get('_id', '')) not in shown]
        
        logging.info(f"All recipes: {len(all_recipes)}, Shown recipes: {len(shown)}, Available: {len(available)}")
        
        if not available:
            return "You've seen all available recipes. Would you like to explore different options?"
        
        # Get starting index for this batch
        start_idx = current_page * 3
        logging.info(f"Page {current_page}, {len(available)} recipes available")
        
        # If we have meaningful search text, use vector similarity
        selected_recipes = []
        if user_message.strip():
            try:
                # Get embedding for user message
                query_embedding = self.embeddings.embed_query(user_message)
                
                # Get IDs of available recipes
                available_ids = [str(r.get('_id', '')) for r in available]
                
                # Debug logging
                logging.info(f"Searching with query: '{user_message}'")
                logging.info(f"Available recipe IDs: {available_ids[:5]}...")
                
                # Search using ChromaDB
                if available_ids:
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(3, len(available_ids)),
                        where={"_id": {"$in": available_ids}},
                        include=["metadatas", "documents", "distances"]
                    )
                    
                    # Debug logging for results
                    logging.info(f"ChromaDB result keys: {results.keys()}")
                    if 'distances' in results and results['distances']:
                        logging.info(f"Distances: {results['distances'][0]}")
                    
                    # Process results into recipes
                    if 'metadatas' in results and results['metadatas'] and results['metadatas'][0]:
                        for i in range(len(results['metadatas'][0])):
                            try:
                                # Start with metadata
                                recipe = results['metadatas'][0][i].copy()
                                
                                # Add document content
                                if 'documents' in results and results['documents'] and len(results['documents'][0]) > i:
                                    doc_content = results['documents'][0][i]
                                    # Try parsing as JSON if it's a string
                                    if isinstance(doc_content, str):
                                        try:
                                            doc_data = json.loads(doc_content)
                                            if isinstance(doc_data, dict):
                                                recipe.update(doc_data)
                                        except json.JSONDecodeError:
                                            recipe['content'] = doc_content
                                    elif isinstance(doc_content, dict):
                                        recipe.update(doc_content)
                                
                                # Add ID if missing
                                if '_id' not in recipe and 'ids' in results and results['ids'][0]:
                                    recipe['_id'] = results['ids'][0][i]
                                    
                                selected_recipes.append(recipe)
                                
                            except Exception as e:
                                logging.error(f"Error processing recipe result at index {i}: {str(e)}")
                                continue
                        
                        logging.info(f"Successfully processed {len(selected_recipes)} recipes from results")
                
                logging.info("Using similarity-based selection")
                
            except Exception as e:
                logging.error(f"Error in similarity search: {e}", exc_info=True)
                selected_recipes = self.get_random_recipes(available, 3)
                logging.info("Using random selection (similarity search failed)")
        else:
            # No search criteria - use random selection
            selected_recipes = self.get_random_recipes(available, 3)
            logging.info("Using random selection (no search criteria)")
        
        # Fallback to random if similarity search returned no results
        if not selected_recipes and available:
            selected_recipes = self.get_random_recipes(available, 3)
            logging.info("Fallback to random selection - similarity search returned no results")
        
        # Update state
        self.update_recipe_state(selected_recipes)
        self.manager.user_state.recipe_current_page += 1
        
        # Generate response
        response = self.generate_recipe_presentation(
            recipes=selected_recipes,
            user_message=user_message,
            react_decision=react_decision
        )
        
        # Add pagination prompt if more recipes available
        if len(available) > (start_idx + 3):
            response += "\n\nWould you like to see more recipe options?"
        
        return response

    def get_recipe_text(self, recipe: Dict) -> str:
        """Get searchable text representation of a recipe"""
        return " ".join([
            recipe.get("name", ""),
            recipe.get("description", ""),
            " ".join(recipe.get("ingredients", [])),
            " ".join(recipe.get("dietary_flags", [])),
            recipe.get("preparation_time", ""),
            recipe.get("difficulty", ""),
            recipe.get("category", ""),
            recipe.get("age_group", "")
        ]).lower()

    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using TF-IDF"""
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception as e:
            logging.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def get_random_recipes(self, recipes: List[Dict], count: int) -> List[Dict]:
        """Get random recipes when no specific matching needed"""
        if len(recipes) <= count:
            return recipes
        return random.sample(recipes, count)

    def base_search(self, age_group: str) -> List[Dict]:
        """
        Step 1: Base search for recipes using ChromaDB.
        CHANGED: Now uses ChromaDB instead of Elasticsearch
        """
        logging.info("\n=== Step 1: Base Recipe Search ===")
        logging.info(f"Search Parameters:")
        logging.info(f"Category: '{self.CATEGORY}'")
        logging.info(f"Age Group: '{age_group}'")
        
        try:
            # Create search text and get embedding
            search_text = f"recipes for {age_group} baby"
            query_embedding = self.embeddings.embed_query(search_text)
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=20,  # Large pool for second step
                where={"age_group": age_group},  # Metadata filtering
                include=["metadatas", "documents", "distances"]  # Ensure we get all data
            )
            
            # Debug logging
            logging.info(f"ChromaDB result keys: {results.keys()}")
            if 'ids' in results and results['ids']:
                logging.info(f"Number of recipe IDs found: {len(results['ids'][0])}")
            
            # Process results
            processed_results = []
            
            if 'metadatas' in results and results['metadatas'] and results['metadatas'][0]:
                for i in range(len(results['metadatas'][0])):
                    try:
                        # Start with metadata (always a dict)
                        recipe = results['metadatas'][0][i].copy()
                        
                        # Add document content if available
                        if 'documents' in results and results['documents'] and len(results['documents'][0]) > i:
                            doc_content = results['documents'][0][i]
                            # Try parsing as JSON if it's a string
                            if isinstance(doc_content, str):
                                try:
                                    doc_data = json.loads(doc_content)
                                    if isinstance(doc_data, dict):
                                        recipe.update(doc_data)
                                except json.JSONDecodeError:
                                    # Not JSON, store as content
                                    recipe['content'] = doc_content
                            elif isinstance(doc_content, dict):
                                # If already a dict, update directly
                                recipe.update(doc_content)
                        
                        # Ensure recipe has an ID
                        if '_id' not in recipe and 'ids' in results and results['ids'][0]:
                            recipe['_id'] = results['ids'][0][i]
                            
                        processed_results.append(recipe)
                        
                    except Exception as e:
                        logging.error(f"Error processing recipe at index {i}: {str(e)}")
                        continue
            
            logging.info(f"Found {len(processed_results)} matching recipes")
            return processed_results
            
        except Exception as e:
            logging.error(f"Base recipe search failed: {str(e)}", exc_info=True)
            return []

    def _perform_recipe_diagnostic_search(self, age_group: str) -> List[Dict]:
        """
        Performs a broader diagnostic search when exact recipe matching fails.
        This helps identify potential issues with recipe categorization or age group matching.
        """
        diagnostic_query = {
            "query": {
                "bool": {
                    "should": [
                        # Look for partial matches on recipe category
                        {"match": {"category": "recipe"}},
                        # Look for partial matches on age group
                        {"match": {"age_group": age_group}},
                        # Look for recipes in similar age ranges
                        {"prefix": {"age_group": age_group.split()[0]}}
                    ],
                    "minimum_should_match": 1
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
            logging.error(f"Recipe diagnostic search failed: {str(e)}")
            return []

    def _process_recipe_results(self, hits: List[Dict]) -> List[Dict]:
        """
        Processes and validates recipe search results to ensure they contain
        required recipe-specific fields and are properly formatted.
        """
        processed_recipes = []
        required_fields = {'age_group', 'category'}
        
        for hit in hits:
            recipe = hit['_source']
            
            # Verify required fields are present
            if all(field in recipe for field in required_fields):
                # Add recipe to processed results
                processed_recipes.append(recipe)
            else:
                # Log warning for incomplete recipes
                logging.warning(
                    f"Skipping incomplete recipe: {recipe.get('title', 'Unknown')} - "
                    f"Missing fields: {required_fields - set(recipe.keys())}"
                )
        
        return processed_recipes

    def search_recipes(self, age_group: str, dietary_considerations: List[str], preparation_preferences: Dict) -> List[Dict]:
        """Enhanced recipe search using Elasticsearch"""
        try:
            # Combine search criteria
            search_aspects = []
            if dietary_considerations:
                search_aspects.extend(dietary_considerations)
            if preparation_preferences:
                search_aspects.append(preparation_preferences["time"])
                search_aspects.append(preparation_preferences["complexity"])
            
            search_title = " ".join(search_aspects)
            
            results = self.base_search(age_group)
            
            return [hit['_source'] for hit in results]
            
        except Exception as e:
            logging.error(f"Error in recipe search: {str(e)}")
            return []

    def debug_search(self, index_name: str, category: str, age_group: str, dietary_considerations: Optional[List[str]] = None, preparation_preferences: Optional[Dict] = None) -> List[Dict]:
        """
        Performs a refined recipe search with additional criteria beyond base matching.
        This function handles the more complex search requirements including dietary
        preferences and preparation considerations.
        
        Args:
            index_name: Name of the Elasticsearch index
            category: Category of items (should be "recipe")
            age_group: Age group to search within
            dietary_considerations: List of dietary requirements or preferences
            preparation_preferences: Dictionary containing preparation time and complexity preferences
            
        Returns:
            List[Dict]: List of matching recipes
        """
        # Start with the basic must-match criteria
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"age_group.keyword": age_group}},
                        {"term": {"category.keyword": self.CATEGORY}}
                    ]
                }
            }
        }
        
        # Add dietary consideration matching if provided
        if dietary_considerations and any(dietary_considerations):
            # Create a should clause for dietary flags with high boost
            query["query"]["bool"]["should"].extend([
                {
                    "match": {
                        "dietary_flags": {
                            "query": consideration,
                            "boost": 3.0  # Give high importance to dietary matches
                        }
                    }
                }
                for consideration in dietary_considerations
                if consideration
            ])
        
        # Add preparation preference matching if provided
        if preparation_preferences:
            # Add time preference matching
            if preparation_preferences.get("time"):
                query["query"]["bool"]["should"].append({
                    "term": {
                        "preparation_time.keyword": {
                            "value": preparation_preferences["time"],
                            "boost": 2.0
                        }
                    }
                })
            
            # Add complexity preference matching
            if preparation_preferences.get("complexity"):
                query["query"]["bool"]["should"].append({
                    "term": {
                        "difficulty.keyword": {
                            "value": preparation_preferences["complexity"],
                            "boost": 2.0
                        }
                    }
                })
        
        # Add additional fields for general matching with appropriate weights
        query["query"]["bool"]["should"].extend([
            {"match": {"name": {"query": " ".join(dietary_considerations or []), "boost": 2.0}}},
            {"match": {"description": {"query": " ".join(dietary_considerations or []), "boost": 1.5}}},
            {"match": {"ingredients": {"query": " ".join(dietary_considerations or []), "boost": 1.0}}}
        ])
        
        # If we have any should clauses, require at least one to match
        if query["query"]["bool"]["should"]:
            query["query"]["bool"]["minimum_should_match"] = 1
        
        try:
            # Log the search attempt
            logging.info(f"\n=== Executing Refined Recipe Search ===\n"
                        f"Age Group: {age_group}\n"
                        f"Category: {category}\n"
                        f"Dietary Considerations: {dietary_considerations}\n"
                        f"Preparation Preferences: {preparation_preferences}\n"
                        f"Query:\n{json.dumps(query, indent=2)}")
            
            # Execute search
            results = self.es.search(index=index_name, body=query)
            hits = results['hits']['hits']
            
            # Log results summary
            logging.info(f"Found {len(hits)} results in refined recipe search")
            
            return hits
            
        except Exception as e:
            logging.error(f"Error in refined recipe search: {str(e)}")
            logging.error(f"Query that caused error: {json.dumps(query, indent=2)}")
            return []
    
    def select_recipes_for_presentation(self, available_recipes: List[Dict], 
                                      user_message: str, react_decision: Dict) -> List[Dict]:
        """Select most relevant recipes for presentation"""
        sorted_recipes = sorted(
            available_recipes,
            key=lambda r: self._calculate_recipe_relevance(r, user_message, react_decision),
            reverse=True
        )
        
        return sorted_recipes[:self.max_items_per_response]

    def _calculate_recipe_relevance(self, recipe: Dict, user_message: str, 
                                  react_decision: Dict) -> float:
        """Calculate relevance score for a recipe"""
        score = 0.0
        
        # Check dietary considerations match
        dietary_prefs = self.manager.user_state.dietary_preferences
        if dietary_prefs:
            matching_prefs = set(recipe.get("dietary_flags", [])).intersection(dietary_prefs.keys())
            score += len(matching_prefs) * 0.3
        
        # Check preparation preferences match
        prep_prefs = react_decision.get("preparation_preferences", {})
        if prep_prefs.get("time") == recipe.get("preparation_time"):
            score += 0.2
        if prep_prefs.get("complexity") == recipe.get("difficulty"):
            score += 0.2
            
        # Consider keywords from user message
        keywords = set(user_message.lower().split())
        recipe_text = " ".join([
            recipe.get("name", ""),
            recipe.get("description", ""),
            " ".join(recipe.get("ingredients", []))
        ]).lower()
        
        matching_keywords = keywords.intersection(set(recipe_text.split()))
        score += len(matching_keywords) * 0.1
        
        return min(1.0, score)

    def update_recipe_state(self, recipes: List[Dict]) -> None:
        """
        Update recipe state with new recipes, safely handling recipe identification.
        
        This function:
        1. Adds recipes to shown list using safe identifiers
        2. Updates current recipes in state
        3. Updates timestamp for tracking
        """
        try:
            for recipe in recipes:
                # Use the stored ID or generate a new one
                recipe_id = recipe.get('_id') or str(hash(frozenset(sorted(recipe.items()))))
                self.manager.user_state.shown_recipes.add(str(recipe_id))
            
            self.manager.user_state.current_recipes = recipes
            self.manager.user_state.last_recipe_search = time.time()
            
            logging.info(f"Updated state with {len(recipes)} new recipes")
            
        except Exception as e:
            logging.error(f"Error in recipe state update: {str(e)}", exc_info=True)

        """Update recipe state with new recipes"""
        try:
            for recipe in recipes:
                # Use stored ID or generate a new one
                recipe_id = recipe.get('_id') or str(hash(frozenset(recipe.items())))
                self.manager.user_state.shown_recipes.add(recipe_id)
            
            self.manager.user_state.current_recipes = recipes
            self.manager.user_state.last_recipe_search = time.time()
            
            logging.info(f"Updated state with {len(recipes)} new recipes")
            
        except Exception as e:
            logging.error(f"Error in recipe state update: {str(e)}")

    def update_dietary_preferences(self, new_preferences: List[str]) -> None:
        """Update stored dietary preferences with new information"""
        current_preferences = self.manager.user_state.dietary_preferences
        
        for pref in new_preferences:
            if pref not in current_preferences:
                current_preferences[pref] = {
                    "first_mentioned": time.time(),
                    "last_confirmed": time.time()
                }
            else:
                current_preferences[pref]["last_confirmed"] = time.time()

    def generate_recipe_presentation(self, recipes: List[Dict], user_message: str, 
                                   react_decision: Dict) -> str:
        """Generate natural, informative recipe presentation"""
        presentation_prompt = f"""
        Task: Present recipes in a helpful, engaging way.
        
        Recipes: {json.dumps(recipes)}
        User Message: {user_message}
        Dietary Considerations: {react_decision.get('dietary_considerations')}
        Preparation Preferences: {react_decision.get('preparation_preferences')}
        
        Create a response that:
        1. Introduces each recipe naturally
        2. Highlights nutritional benefits for the age group
        3. Notes preparation time and complexity
        4. Mentions any dietary considerations
        5. Encourages questions about preparation
        6. Maintains friendly, helpful tone
        """
        
        initial_response = self.manager.get_gpt_response(presentation_prompt)
        return self.ensure_hallucination_free_response(
            initial_response,
            recipes,
            "Recipe Presentation"
        )

    def ensure_hallucination_free_response(self, response: str, context_data: Dict, 
                                         category: str) -> str:
        """Verify response against context data"""
        verification_prompt = f"""
        Task: Verify recipe response accuracy.

        Response: {response}
        Recipe Data: {json.dumps(context_data)}
        Category: {category}

        Verify that:
        1. All mentioned recipes exist in data
        2. All ingredients and steps are accurate
        3. Nutritional claims are supported
        4. Safety considerations are maintained

        Return only "Yes" if verified, "No" if not.
        """
        
        max_attempts = 4
        for attempt in range(max_attempts):
            verification = self.manager.get_gpt_response(verification_prompt).strip().lower()
            
            if verification == "yes":
                return response
            
            refinement_prompt = f"""
            Task: Revise recipe response for accuracy.
            
            Original Response: {response}
            Recipe Data: {json.dumps(context_data)}
            Category: {category}
            
            Create new response using only verified information.
            Ensure all recipe details are accurate.
            """
            
            response = self.manager.get_gpt_response(refinement_prompt)
        
        return self.generate_safe_recipe_fallback(category)

    def generate_safe_recipe_fallback(self, category: str) -> str:
        # Return string instead of dict
        return ("I'm having difficulty finding recipes that match your criteria. " +
                "Would you like to try a different search? " +
                "You can tell me your baby's age, any dietary requirements, " +
                "or if you prefer simpler recipes.")

    def suggest_topic_change(self, reasoning: str) -> str:
        """Suggest changing topics when recipe discussion isn't productive"""
        return f"""
        Based on our discussion, {reasoning}
        
        Would you like to:
        • Explore different types of recipes?
        • Learn about meal planning?
        • Discuss dietary guidelines for your baby's age?
        """

    def fallback_recipe_handling(self, user_message: str, age_group: str, requery: bool) -> str:
        """Handle recipe recommendation when primary method fails"""
        try:
            # Simple keyword-based search
            basic_search = self.search_recipes(age_group)
            if basic_search:
                return self.generate_recipe_presentation(
                    basic_search[:self.max_items_per_response],
                    user_message,
                    {"dietary_considerations": [], "preparation_preferences": {}}
                )
            return self.generate_safe_recipe_fallback("Recipe Recommendation")
        except Exception as e:
            logging.error(f"Error in fallback handling: {str(e)}")
            return self.generate_safe_recipe_fallback("Recipe Recommendation")
    