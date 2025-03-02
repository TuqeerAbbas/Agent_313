# managers/product_manager.py

from typing import Dict, List, Any, Optional
import json
import logging
import time
import chromadb
from elasticsearch import Elasticsearch
from langchain_openai import OpenAIEmbeddings

from config.config import Config
from models.user_state import UserState
from utils.error_handler import ConversationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ProductManager:
    """Handles all product-related functionality including recommendations and upsells."""
    
    def __init__(self, conversation_manager):
        self.config = Config()
        self.manager = conversation_manager
        self.max_items_per_response = self.config.MAX_ITEMS_PER_RESPONSE
        self.CATEGORY = "product"
        
        # Initialize ChromaDB
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

    def handle_product_recommendation(self, user_message: str, age_group: str, requery: bool) -> str:
        """
        Main entry point for all product-related interactions. This method orchestrates 
        the complete product recommendation process using our two-step search approach.
        
        The method follows this process:
        1. Get ReACT decision to understand user intent
        2. If new search needed:
        - Step 1: Get all potential products (base_search)
        - Step 2: Rank and return best matches (get_next_product_batch)
        3. If continuing discussion:
        - Handle ongoing product conversation
        4. Handle upsell opportunities when appropriate
        
        Args:
            user_message: The user's input message
            age_group: User's child age group (e.g., "0-6 months")
            requery: Whether this is a repeated search
            
        Returns:
            str: Formatted response with product recommendations or guidance
        """
        try:
            # First verify we have required age information
            if not age_group:
                logging.info("No age group provided for product recommendation")
                return "Could you please tell me your baby's age? This helps me recommend appropriate products."

            # Get ReACT decision about user's intent
            logging.info(f"Getting ReACT decision for product handling - Age: {age_group}")
            react_decision = self.get_product_react_decision(user_message, age_group)
            
            # Only proceed with high-confidence decisions
            if react_decision["confidence"] >= self.config.REACT_CONFIDENCE_THRESHOLD:
                if react_decision["decision"] == "new_search":
                    # PATH 1: NEW PRODUCT SEARCH
                    logging.info("Starting two-step product search process")
                    
                    # Step 1: Get complete pool of age-appropriate products
                    base_results = self.base_search(age_group)
                    if not base_results:
                        logging.info(f"No products found for age group: {age_group}")
                        return ("I couldn't find any products for this age group. "
                            "Would you like to explore different options?")
                    
                    # Store complete result set for pagination
                    self.manager.user_state.all_available_products = base_results
                    self.manager.user_state.current_page = 0
                    logging.info(f"Stored {len(base_results)} products for pagination")
                    
                    # Step 2: Get first batch using similarity ranking
                    response = self.get_next_items_batch(user_message, react_decision)
                    
                    # Check for upsell opportunities if we found products
                    if response:
                        upsell_response = self.check_upsell_opportunity(
                            user_message=user_message,
                            age_group=age_group,
                            react_decision=react_decision
                        )
                        
                        # Add upsell suggestions if appropriate
                        if upsell_response:
                            response = f"{response}\n\n{upsell_response}"
                    
                    logging.info("Generated product recommendations with upsell check")
                    return response
                    
                elif react_decision["decision"] == "continue_discussion":
                    # PATH 2: ONGOING PRODUCT DISCUSSION
                    logging.info("Continuing existing product discussion")
                    return self.handle_product_discussion(
                        user_message=user_message,
                        react_decision=react_decision
                    )
                    
                else:
                    # PATH 3: TOPIC CHANGE NEEDED
                    logging.info("Suggesting topic change based on ReACT decision")
                    return self.suggest_topic_change(react_decision["reasoning"])
            
            # Fall back to basic handling if low confidence
            logging.info("Falling back to basic product handling due to low confidence")
            return self.fallback_product_handling(
                user_message=user_message,
                age_group=age_group,
                requery=requery
            )
            
        except Exception as e:
            # Log error details and fall back to safe handling
            logging.error(f"Error in product handling: {str(e)}", exc_info=True)
            return self.fallback_product_handling(
                user_message=user_message,
                age_group=age_group,
                requery=requery
            )

    def _clean_json_response(self, response: str) -> str:
        """Clean GPT response to ensure valid JSON"""
        # Remove any markdown formatting
        response = response.replace('```json\n', '').replace('```', '')
        # Remove any leading/trailing whitespace
        response = response.strip()
        return response

    def get_product_react_decision(self, user_message: str, age_group: str) -> Dict:
        """Get ReACT reasoning for product handling"""
        react_prompt = f"""
        Task: Determine how to handle product recommendation request.

        Context:
        - User Message: {user_message}
        - Age Group: {age_group}
        - Current Products: {json.dumps(self.manager.user_state.current_products)}
        - Previously Shown: {list(self.manager.user_state.shown_products)}
        
        Think through:
        1. Is user asking about current products or needs new recommendations?
        2. What specific aspects of products are they interested in?
        3. Should we show new products or discuss existing ones?
        4. Are there potential upsell opportunities?
        
        Respond with structured JSON:
        {{
            "decision": "new_search/continue_discussion/change_topic",
            "confidence": 0.0-1.0,
            "focus_aspects": ["aspect1", "aspect2"],
            "upsell_potential": true/false,
            "reasoning": "explanation of decision"
        }}
        """
        
        response = self.manager.get_gpt_response(react_prompt)
        cleaned_response = self._clean_json_response(response)
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse product react decision: {e}")
            return {
                "decision": "new_search",
                "confidence": 0.5,
                "focus_aspects": [],
                "upsell_potential": False,
                "reasoning": "Fallback due to parsing error"
            }

    def base_search(self, age_group: str) -> List[Dict]:
        """
        Step 1 of our search process: Get ALL products matching age and category.
        Uses ChromaDB instead of Elasticsearch.
        """
        logging.info("\n=== Step 1: Base Product Search ===")
        logging.info(f"Search Parameters:")
        logging.info(f"Category: '{self.CATEGORY}'")
        logging.info(f"Age Group: '{age_group}'")
        
        try:
            # Add debug logging for collection
            logging.info(f"Collection name: {self.collection.name}")
            logging.info(f"Collection count: {self.collection.count()}")
            
            # Get sample data to verify format
            sample_data = self.collection.get(limit=1)
            logging.info(f"Sample data: {json.dumps(sample_data, indent=2)}")

            # Create search text and get embedding
            search_text = f"category:{self.CATEGORY} age_group:{age_group}"
            query_embedding = self.embeddings.embed_query(search_text)
            
            # First try without metadata filter
            logging.info("Querying without metadata filter...")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=7
            )
            logging.info(f"Results without filter: {len(results['documents'][0]) if results['documents'] else 0} items")

            # Then try with metadata filter
            logging.info("Querying with metadata filter...")
            filtered_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=7,
                where={"age_group": age_group}
            )
            logging.info(f"Results with filter: {len(filtered_results['documents'][0]) if filtered_results['documents'] else 0} items")
            
            # Process results
            processed_results = []
            for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                try:
                    content = json.loads(doc)
                    content.update(metadata)
                    processed_results.append(content)
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing product data: {e}")
                    continue
            
            logging.info(f"Found {len(processed_results)} matching products")
            return processed_results
            
        except Exception as e:
            logging.error(f"Base product search failed: {str(e)}", exc_info=True)
            return []
        
    def _perform_diagnostic_search(self, age_group: str) -> List[Dict]:
        """
        Helper method to perform a broader diagnostic search when exact matching fails.
        This helps identify why our main search might not be finding products.
        """
        diagnostic_query = {
            "query": {
                "bool": {
                    "should": [
                        # Look for partial matches on category
                        {"match": {"category": self.CATEGORY}},
                        # Look for partial matches on age group
                        {"match": {"age_group": age_group}},
                        # Look for products in nearby age groups
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
            logging.error(f"Diagnostic search failed: {str(e)}")
            return []

    def calculate_product_similarity(self, product: Dict, user_message: str) -> float:
        """Calculate cosine similarity between product text and user message"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Create searchable product text
            product_text = " ".join([
                product.get("name", ""),
                product.get("description", ""),
                " ".join(product.get("features", [])),
                product.get("category", ""),
                product.get("age_group", "")
            ]).lower()
            
            # Calculate similarity
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([product_text, user_message])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
            
        except Exception as e:
            logging.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def handle_new_product_search(self, user_message: str, age_group: str, react_decision: Dict) -> str:
        try:
            if not age_group:
                return "Could you please tell me your baby's age? This helps me recommend appropriate products."

            # Step 1: Get base results by age/category
            base_results = self.base_search(age_group)
            if not base_results:
                return "I couldn't find any products for this age group. Would you like to explore different options?"
                
            # Store all results for pagination
            self.manager.user_state.all_available_products = base_results
            self.manager.user_state.current_page = 0
            
            # Step 2: Get first batch
            return self.get_next_items_batch(user_message, react_decision)
                
        except Exception as e:
                logging.error(f"Error in product search: {str(e)}")
                return self.generate_safe_fallback_response("Product Recommendation")

    def get_next_items_batch(self, user_message: str, react_decision: Dict) -> str:
        """
        Step 2 of our search process: Apply vector similarity ranking and handle pagination.
        """
        logging.info("=== Step 2: Getting Next Product Batch ===")
        
        # Get our working set of products
        all_items = self.manager.user_state.all_available_products
        shown = self.manager.user_state.shown_products
        current_page = self.manager.user_state.current_page
        
        if not all_items:
            logging.error("No products available in state")
            return self.generate_safe_fallback_response("Product Batch Retrieval")
        
        # Filter out shown products
        available = []
        for item in all_items:
            # Create a stable, unique identifier for the product based on its content
            product_id = str(hash(frozenset({
                k: str(v) for k, v in item.items()
                if k in ['title', 'name', 'description', 'age_group', 'category']
            }.items())))
            
            # Store ID in product for future reference
            item['_id'] = product_id
            
            # Check if we've shown this product before
            if product_id not in shown:
                available.append(item)
        
        if not available:
            return "You've seen all available products. Would you like to explore different options?"
        
        # Get starting index for this batch
        start_idx = current_page * 3
        logging.info(f"Page {current_page}, {len(available)} products available")
        
        # If we have meaningful search text, use vector similarity
        if user_message.strip():
            try:
                # Get embedding for user message
                query_embedding = self.embeddings.embed_query(user_message)
                
                # Search using ChromaDB
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=3,
                    where={"age_group": self.manager.user_state.age_group}
                )
                
                # Process results
                selected_products = []
                for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                    try:
                        product = json.loads(doc)
                        product.update(metadata)
                        selected_products.append(product)
                    except json.JSONDecodeError:
                        logging.error(f"Error parsing product data at index {i}")
                        continue
                
                logging.info("Using similarity-based selection")
                
            except Exception as e:
                logging.error(f"Error in similarity search: {e}")
                # Fallback to random selection
                selected_products = self.get_random_products(available, 3)
                logging.info("Using random selection (similarity search failed)")
        else:
            # No search criteria - use random selection
            selected_products = self.get_random_products(available, 3)
            logging.info("Using random selection (no search criteria)")
        
        # Ensure we have products to display
        if not selected_products and available:
            selected_products = available[:3]
            logging.info("Using first 3 available products as fallback")
        
        # Update state with proper error handling
        try:
            self.update_product_state(selected_products)
            self.manager.user_state.current_page += 1
        except Exception as e:
            logging.error(f"Error updating product state: {str(e)}")
        
        # Debug log the selected products
        logging.info(f"Selected products for presentation: {[p.get('title', 'Unknown') for p in selected_products]}")
        
        # Generate response
        response = self.generate_product_presentation(
            products=selected_products,  # Pass the actual products!
            user_message=user_message,
            react_decision=react_decision
        )
        
        # Add pagination prompt if more products available
        if len(available) > (start_idx + 3):
            response += "\n\nWould you like to see more options?"
        
        return response

    def get_random_products(self, products: List[Dict], count: int) -> List[Dict]:
        """Get random products when no specific matching needed"""
        import random
        if len(products) <= count:
            return products
        return random.sample(products, count)

    def get_upsell_react_decision(self, user_message: str, age_group: str) -> Dict:
        """Get ReACT reasoning for upsell opportunities"""
        react_prompt = f"""
        Task: Determine upsell/cross-sell opportunities.

        Context:
        - User Message: {user_message}
        - Age Group: {age_group}
        - Original Products: {json.dumps(self.manager.user_state.original_product_context)}
        - Previous Products: {list(self.manager.user_state.shown_products)}
        
        Return JSON with:
        {{
            "should_upsell": boolean,
            "confidence": 0.0-1.0,
            "product_aspects": ["aspect1", "aspect2"],
            "reasoning": "explanation"
        }}
        """
        
        try:
            response = self.manager.get_gpt_response(react_prompt)
            cleaned_response = self._clean_json_response(response)
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            return {
                "should_upsell": False,
                "confidence": 0.0,
                "product_aspects": [],
                "reasoning": "Error parsing upsell decision"
            }

    def search_products(self, age_group: str, aspects: List[str]) -> List[Dict]:
        """Enhanced product search using Elasticsearch"""
        try:
            logging.info(f"Starting product search with aspects: {aspects}")
            logging.info(f"Age group: {age_group}")
            
            results = self.base_search(age_group)
            
            # Validate results
            if not isinstance(results, list):
                logging.error(f"Unexpected results format: {type(results)}")
                return []
                
            logging.info(f"Search completed with {len(results)} results")
            return results
            
        except Exception as e:
            logging.error(f"Error in product search: {str(e)}")
            logging.error(f"Search parameters - age_group: {age_group}, aspects: {aspects}")
            return []

    def debug_search(self, index_name: str, category: str, age_group: str, search_aspects: Optional[List[str]] = None) -> List[Dict]:
        """
        Performs a refined search with additional criteria beyond base matching.
        This is used for searching with specific aspects or features.
        
        Args:
            index_name: Name of the Elasticsearch index
            category: Category of items to search for
            age_group: Age group to search within
            search_aspects: Optional list of specific aspects to search for
            
        Returns:
            List[Dict]: List of matching items
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
        
        # Add aspect-based searching if provided
        if search_aspects and any(search_aspects):
            # Convert aspects to a space-separated string
            aspect_text = " ".join(filter(None, search_aspects))
            
            if aspect_text.strip():
                # Add should clauses to search in relevant fields
                query["query"]["bool"]["should"] = [
                    # Search in multiple relevant fields with different weights
                    {"match": {"name": {"query": aspect_text, "boost": 3.0}}},
                    {"match": {"description": {"query": aspect_text, "boost": 2.0}}},
                    {"match": {"features": {"query": aspect_text, "boost": 2.0}}},
                    {"match": {"tags": {"query": aspect_text, "boost": 1.0}}}
                ]
                
                # Require at least one should clause to match
                query["query"]["bool"]["minimum_should_match"] = 1
        
        try:
            # Log the search attempt
            logging.info(f"\n=== Executing Refined Product Search ===\n"
                        f"Age Group: {age_group}\n"
                        f"Category: {category}\n"
                        f"Search Aspects: {search_aspects}\n"
                        f"Query:\n{json.dumps(query, indent=2)}")
            
            # Execute search
            results = self.es.search(index=index_name, body=query)
            hits = results['hits']['hits']
            
            # Log results summary
            logging.info(f"Found {len(hits)} results in refined search")
            
            # Return results
            return hits
            
        except Exception as e:
            logging.error(f"Error in refined product search: {str(e)}")
            logging.error(f"Query that caused error: {json.dumps(query, indent=2)}")
            return []

    def log_sensitivity_check(self, sensitivity_check: Dict) -> None:
        """Log sensitivity check results for tracking"""
        logging.info(f"Sensitivity Check Results:\n"
                    f"Topic: {sensitivity_check['topic']}\n"
                    f"Level: {sensitivity_check['sensitivity_level']}\n"
                    f"Professional Required: {sensitivity_check['requires_professional']}")

    def generate_safe_parenting_fallback(self, category: str) -> str:
        """Generate safe fallback response when errors occur"""
        return ("I'd like to help you with your parenting question. "
                "Could you provide more details about:\n"
                "• Your specific concern\n"
                "• Your child's age\n"
                "• Any relevant background information")

    def select_products_for_presentation(self, available_products: List[Dict], user_message: str, react_decision: Dict) -> List[Dict]:
        """Select most relevant products for presentation"""
        sorted_products = sorted(
            available_products,
            key=lambda p: self._calculate_product_relevance(p, user_message, react_decision),
            reverse=True
        )
        
        return sorted_products[:self.max_items_per_response]

    def _calculate_product_relevance(self, product: Dict, user_message: str, react_decision: Dict) -> float:
        """Calculate relevance score for a product"""
        score = 0.0
        
        # Check focus aspects match
        for aspect in react_decision.get("focus_aspects", []):
            if aspect.lower() in product.get("description", "").lower():
                score += 0.3
                
        # Check age group match
        if product.get("age_group") == self.manager.user_state.age_group:
            score += 0.4
            
        # Consider user message keywords
        keywords = set(user_message.lower().split())
        product_text = " ".join([
            product.get("name", ""),
            product.get("description", "")
        ]).lower()
        
        matching_keywords = keywords.intersection(set(product_text.split()))
        score += len(matching_keywords) * 0.1
        
        return min(1.0, score)

    def update_product_state(self, products: List[Dict]) -> None:
        """
        Update product state with new products, safely handling product identification.
        This method ensures we properly track which products we've shown to the user,
        even when products don't have built-in IDs.
        """
        try:
            for product in products:
                # Use stored ID or generate a new one
                product_id = product.get('_id') or str(hash(frozenset({
                    k: str(v) for k, v in product.items()
                    if k in ['title', 'name', 'description', 'age_group', 'category']
                }.items())))
                
                self.manager.user_state.shown_products.add(product_id)
            
            self.manager.user_state.current_products = products
            self.manager.user_state.last_product_search = time.time()
            
            logging.info(f"Updated state with {len(products)} new products")
            
        except Exception as e:
            logging.error(f"Error in product state update: {str(e)}")

    def handle_product_discussion(self, user_message: str, react_decision: Dict) -> str:
        """Handle discussion about current products"""
        prompt = f"""
        Create response discussing current products:
        Products: {json.dumps(self.manager.user_state.current_products)}
        User Message: {user_message}
        Analysis: {json.dumps(react_decision)}
        
        Response should:
        1. Address specific questions/concerns
        2. Provide relevant details
        3. Suggest related products if appropriate
        4. Maintain helpful tone
        """
        
        return self.manager.get_gpt_response(prompt)

    def check_upsell_opportunity(self, user_message: str, age_group: str, react_decision: Dict) -> Optional[str]:
        """Check for and handle upsell opportunities"""
        if not react_decision.get("upsell_potential"):
            return None
            
        try:
            self.manager.user_state.original_product_context = {
                "products": self.manager.user_state.current_products,
                "message": user_message,
                "aspects": react_decision["focus_aspects"]
            }
            
            return self.handle_upsell_cross_sell(user_message, age_group)
            
        except Exception as e:
            logging.error(f"Error in upsell check: {str(e)}")
            return None

    def handle_upsell_cross_sell(self, user_message: str, age_group: str) -> Optional[str]:
        """
        Handle upsell/cross-sell recommendations using two-step search process.
        First gets all potential related products, then uses similarity to find best matches.
        """
        try:
            logging.info("Starting upsell/cross-sell process")
            
            # Get upsell decision from ReACT
            react_decision = self.get_upsell_react_decision(user_message, age_group)
            
            if not react_decision["should_upsell"] or \
            react_decision["confidence"] < self.config.REACT_CONFIDENCE_THRESHOLD:
                logging.info("Upsell not recommended by ReACT decision")
                return None
                
            # Step 1: Get base pool of potential related products
            base_products = self.base_search(age_group)
            if not base_products:
                logging.info("No base products found for upsell")
                return None
                
            # Filter out current products
            current_ids = {str(p.get('id')) for p in self.manager.user_state.current_products}
            available_products = [p for p in base_products if str(p.get('id')) not in current_ids]
            
            if not available_products:
                logging.info("No additional products available for upsell")
                return None
                
            # Step 2: Find most relevant products using similarity
            products_with_scores = []
            search_context = " ".join(react_decision["product_aspects"])
            
            for product in available_products:
                similarity = self.calculate_product_similarity(product, search_context)
                products_with_scores.append((product, similarity))
                
            # Sort by similarity and get top matches
            products_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Use top 2-3 most relevant products for upsell
            top_products = [p[0] for p in products_with_scores[:2]]
            
            if not top_products:
                logging.info("No relevant products found for upsell")
                return None
                
            logging.info(f"Found {len(top_products)} relevant products for upsell")
            
            # Generate upsell presentation
            return self.generate_upsell_presentation(
                top_products,
                react_decision,
                user_message
            )
                
        except Exception as e:
            logging.error(f"Error in upsell handling: {str(e)}", exc_info=True)
            return None

    def generate_product_presentation(self, products: List[Dict], user_message: str, react_decision: Dict) -> str:
        """Generate engaging product presentation"""
        try:
            # Add logging to see what products we're working with
            logging.info(f"Generating presentation for {len(products)} products")
            for idx, prod in enumerate(products):
                logging.info(f"Product {idx+1} title: {prod.get('title', 'Unknown')}")
            
            # If no products, return early with a helpful message
            if not products:
                return "I'm sorry, I couldn't find specific products matching your criteria. Could you provide more details about what you're looking for?"
            
            presentation_prompt = f"""
            Create engaging product presentation:
            Products: {json.dumps(products)}
            User Message: {user_message}
            Analysis: {json.dumps(react_decision)}
            
            Presentation should:
            1. Introduce each product naturally
            2. Highlight key features and benefits
            3. Consider age appropriateness
            4. Maintain helpful, informative tone
            5. Include relevant safety information
            """
            
            initial_response = self.manager.get_gpt_response(presentation_prompt)
            
            # Verify the response against the actual products
            verified_response = self.ensure_hallucination_free_response(
                initial_response,
                products,  # Pass the actual products for verification
                "Product Presentation"
            )
            
            # If verification failed, create a simpler response based directly on the data
            if not verified_response or verified_response == "No":
                return self.generate_simple_product_response(products)
                
            return verified_response
            
        except Exception as e:
            logging.error(f"Error in product presentation: {e}")
            return self.generate_safe_fallback_response("Product Presentation")

    def ensure_hallucination_free_response(self, response: str, context_data: List[Dict], category: str) -> str:
        """Verify response against context data"""
        try:
            # If no context data, can't verify
            if not context_data:
                logging.warning("No context data provided for verification")
                return response
                
            verification_prompt = f"""
            Task: Verify response accuracy against provided data.

            Response: {response}
            Context Data: {json.dumps(context_data)}
            Category: {category}

            Verify that:
            1. All mentioned products/features exist in context
            2. All claims are supported by data
            3. Response stays within category boundaries

            Return only "Yes" if verified, "No" if not.
            """
            
            verification = self.manager.get_gpt_response(verification_prompt).strip().lower()
            
            if verification == "yes":
                return response
            
            logging.warning(f"Response failed verification: {verification}")
            
            # Return the original response anyway, as we'll handle fallback in the calling method
            return response
        except Exception as e:
            logging.error(f"Error in response verification: {str(e)}")
            return response

    def generate_safe_fallback_response(self, category: str) -> str:
        """Generate safe fallback response when normal processing fails"""
        # Return string instead of dict
        return ("I'm having trouble finding specific product recommendations. "
                "Would you like to start over with your product search? "
                "Please tell me your baby's age and what type of product you're looking for.")

    def fallback_product_handling(self, user_message: str, age_group: str, requery: bool) -> str:
        """Handle product recommendation when primary method fails"""
        try:
            # Simple keyword-based search
            basic_search = self.search_products(age_group, user_message.split())
            if basic_search:
                return self.generate_product_presentation(
                    basic_search[:self.max_items_per_response],
                    user_message,
                    {"focus_aspects": [], "confidence": 0.5}
                )
            return self.generate_safe_fallback_response("Product Recommendation")
        except Exception as e:
            logging.error(f"Error in fallback handling: {str(e)}")
            return self.generate_safe_fallback_response("Product Recommendation")

    def search_related_products(self, age_group: str, aspects: List[str], 
                              current_products: List[Dict]) -> List[Dict]:
        """Search for related products for upsell/cross-sell"""
        try:
            # Extract features from current products
            current_features = []
            for product in current_products:
                current_features.extend(product.get("features", []))
            
            # Combine with new aspects
            search_aspects = list(set(aspects + current_features))
            
            results = self.search_products(age_group, search_aspects)
            
            # Filter out current products
            current_ids = {str(p.get('id')) for p in current_products}
            return [p for p in results if str(p.get('id')) not in current_ids]
            
        except Exception as e:
            logging.error(f"Error in related products search: {str(e)}")
            return []    
        
    def generate_upsell_presentation(self, selected_products: List[Dict], react_decision: Dict, user_message: str) -> str:
        """Generate upsell presentation for complementary products"""
        presentation_prompt = f"""
        Create natural upsell suggestions:
        Products: {json.dumps(selected_products)}
        User Message: {user_message}
        Analysis: {json.dumps(react_decision)}
        
        Response should:
        1. Naturally introduce complementary products
        2. Highlight additional benefits
        3. Keep helpful, informative tone
        4. Include relevant safety notes
        """
        
        try:
            initial_response = self.manager.get_gpt_response(presentation_prompt)
            verified_response = self.ensure_hallucination_free_response(
                initial_response,
                selected_products,
                "Product Upsell"
            )
            return verified_response
        except Exception as e:
            logging.error(f"Error generating upsell presentation: {e}")
            return ""
        
    def generate_simple_product_response(self, products: List[Dict]) -> str:
        """Generate a simple response based directly on the product data"""
        try:
            response_parts = ["Here are some products suitable for your baby:"]
            
            for idx, product in enumerate(products, 1):
                title = product.get('title', 'Product')
                description = ""
                
                if 'content' in product:
                    # Extract first 200 characters as a brief description
                    description = product['content'][:200] + "..."
                
                response_parts.append(f"\n{idx}. **{title}**")
                response_parts.append(f"   {description}")
            
            response_parts.append("\nWould you like more details about any of these products?")
            return "\n".join(response_parts)
            
        except Exception as e:
            logging.error(f"Error in simple product response: {e}")
            return "I found some products for your baby but am having trouble displaying them. Please try again."