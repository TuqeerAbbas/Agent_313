# managers/general_qa_manager.py

from typing import Dict, List, Any, Optional
import json
import logging
import time
import chromadb
from langchain_openai import OpenAIEmbeddings

from config.config import Config
from models.user_state import UserState
from utils.error_handler import ConversationError

class GeneralQAManager:
    """
    Handles general questions while intelligently redirecting specialized topics.
    Acts as both a general information provider and a router to specialized handlers.
    """
    
    def __init__(self, conversation_manager):
        self.config = Config()
        self.manager = conversation_manager
        self.max_items_per_response = self.config.MAX_ITEMS_PER_RESPONSE
        
        # Initialize ChromaDB - CHANGED from Elasticsearch
        chroma_config = self.config.get_chroma_config('general')
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
        
        # Initialize specialized handlers mapping
        self.specialized_handlers = {
            "health_concern": self.manager.health_manager,
            "parenting_advice": self.manager.parenting_manager,
            "product_recommendation": self.manager.product_manager,
            "recipe_recommendation": self.manager.recipe_manager
        }

    def _clean_json_response(self, response: str) -> str:
        """Clean GPT response to ensure valid JSON"""
        response = response.replace('```json\n', '').replace('```', '')
        return response.strip()

    def handle_general_qa(self, user_message: str, age_group: str, requery: bool) -> str:
        """Main Q&A handler with specialized topic detection and routing"""
        try:
            # First, analyze question to determine appropriate handler
            category_assessment = self.assess_question_category(user_message, age_group)
            
            # Log assessment for analysis
            self.log_category_assessment(category_assessment)
            
            # Handle specialized redirections
            if category_assessment["needs_specialized_handler"]:
                return self.handle_specialized_redirection(
                    user_message,
                    age_group,
                    category_assessment
                )

            # Get ReACT reasoning for general question handling
            react_decision = self.get_qa_react_decision(
                user_message,
                age_group,
                category_assessment
            )
            
            # Generate contextual notes
            context_notes = self.generate_qa_context_notes(
                react_decision["question_analysis"],
                age_group
            )
            
            if react_decision["confidence"] >= self.config.REACT_CONFIDENCE_THRESHOLD:
                if react_decision["decision"] == "provide_answer":
                    response = self.handle_general_answer(
                        user_message,
                        age_group,
                        react_decision
                    )
                elif react_decision["decision"] == "request_clarification":
                    response = self.generate_clarification_request(
                        react_decision["question_analysis"],
                        react_decision["reasoning"]
                    )
                else:
                    response = self.suggest_related_topics(
                        react_decision["response_strategy"]["related_topics"],
                        react_decision["reasoning"]
                    )
                
                # Add context notes when appropriate
                if context_notes:
                    response = f"{response}\n\n{context_notes}"
                
                return response
                
            return self.fallback_qa_handling(user_message, age_group, requery)
            
        except Exception as e:
            logging.error(f"Error in general Q&A handling: {str(e)}")
            return self.generate_safe_qa_fallback(user_message, age_group)

    def get_qa_react_decision(self, user_message: str, age_group: str, 
                            category_assessment: Dict) -> Dict:
        """Get ReACT reasoning for Q&A handling with category awareness"""
        react_prompt = f"""
        Task: Determine how to handle general question.

        Context:
        - User Message: {user_message}
        - Age Group: {age_group}
        - Current Topics: {json.dumps(self.manager.user_state.current_qa_topics)}
        - Previous Info: {list(self.manager.user_state.shown_qa_info)}
        - Category Assessment: {json.dumps(category_assessment)}
        - Topic History: {json.dumps(self.manager.user_state.topic_history)}
        
        Think through:
        1. What's the core question being asked?
        2. How does this relate to previous topics?
        3. What developmental context is relevant?
        4. What practical guidance is appropriate?
        
        Return structured JSON response including comprehensive analysis.
        """
        
        response = self.manager.get_gpt_response(react_prompt)
        return json.loads(response)

    def handle_specialized_redirection(self, user_message: str, age_group: str, assessment: Dict) -> str:
        """Handle redirection to specialized handlers with context preservation"""
        try:
            # Log redirection for tracking
            self.log_redirection(assessment)
            
            # Get appropriate handler
            handler = self.specialized_handlers.get(
                assessment["recommended_handler"]
            )
            
            if not handler:
                logging.error(f"No handler found for: {assessment['recommended_handler']}")
                return self.generate_safe_qa_fallback("handler_not_found")
            
            handler_method = getattr(handler, f"handle_{assessment['recommended_handler']}")
            return handler_method(user_message, age_group, True)
            
        except Exception as e:
            logging.error(f"Error in specialized redirection: {str(e)}")
            return self.generate_safe_qa_fallback("redirection_failed")
        
    def handle_general_answer(self, user_message: str, age_group: str, react_decision: Dict) -> str:
        """Handle general Q&A with comprehensive responses using ChromaDB"""
        try:
            # Search for relevant QA information
            qa_info = self.search_qa_information(age_group, react_decision["question_analysis"])
            
            if not qa_info:
                return "I understand your question, but I need a bit more context to provide a helpful answer. Could you tell me more?"
            
            # Get embedding for user message
            query_embedding = self.embeddings.embed_query(user_message)
            
            # Get most relevant information using ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                where={"_id": {"$in": [str(info.get('_id', '')) for info in qa_info]}}
            )
            
            # Process results
            selected_info = []
            for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                try:
                    info = json.loads(doc)
                    info.update(metadata)
                    selected_info.append(info)
                except json.JSONDecodeError:
                    continue
            
            # Update state before generating response
            self.update_qa_state(selected_info)
            if age_group:
                self.update_developmental_insights(age_group, react_decision["question_analysis"])
            
            response = self.generate_qa_response(selected_info, user_message, react_decision)
            
            if not response:
                return self.generate_safe_qa_fallback("response_generation_failed")
                
            return response
            
        except Exception as e:
            logging.error(f"Error in general answer handling: {str(e)}")
            return self.generate_safe_qa_fallback("error_handling")

    def search_qa_information(self, age_group: str, question_analysis: Dict) -> List[Dict]:
        """
        Enhanced Q&A information search using ChromaDB.
        CHANGED: Now uses ChromaDB instead of Elasticsearch
        """
        try:
            # Build search text from analysis
            search_aspects = []
            if question_analysis.get("core_topic"):
                search_aspects.append(question_analysis["core_topic"])
            if question_analysis.get("sub_topics"):
                search_aspects.extend(question_analysis["sub_topics"])
            
            search_text = " ".join(search_aspects)
            
            # Get embedding for search
            query_embedding = self.embeddings.embed_query(search_text)
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=7,
                where={"age_group": age_group} if age_group else None
            )
            
            # Process results
            processed_results = []
            for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                try:
                    content = json.loads(doc)
                    content.update(metadata)
                    processed_results.append(content)
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing QA data: {e}")
                    continue
            
            return processed_results
            
        except Exception as e:
            logging.error(f"Error in QA information search: {str(e)}")
            return []

    def update_developmental_insights(self, age_group: str, question_analysis: Dict) -> None:
        """Update developmental insights based on question analysis"""
        insights = self.manager.user_state.developmental_insights
        
        if age_group not in insights:
            insights[age_group] = {
                'topics_discussed': set(),
                'developmental_areas': set(),
                'last_updated': time.time()
            }
        
        if question_analysis.get("developmental_context"):
            insights[age_group]['developmental_areas'].update(
                question_analysis["developmental_context"]
            )
        
        insights[age_group]['topics_discussed'].add(
            question_analysis.get("core_topic")
        )
        insights[age_group]['last_updated'] = time.time()

    def update_qa_state(self, selected_info: List[Dict]) -> None:
        """Update Q&A state with new information"""
        # Update shown info tracking
        for info in selected_info:
            if 'id' in info:
                self.manager.user_state.shown_qa_info.add(str(info['id']))
            
            # Track topics in history
            if 'topic' in info:
                if info['topic'] not in self.manager.user_state.topic_history:
                    self.manager.user_state.topic_history[info['topic']] = []
                    
                self.manager.user_state.topic_history[info['topic']].append({
                    'timestamp': time.time(),
                    'info_id': info.get('id'),
                    'context': info.get('context')
                })

    def generate_qa_response(self, selected_info: List[Dict], user_message: str, react_decision: Dict) -> str:
        """Generate comprehensive Q&A response"""
        response_prompt = f"""
        Create Q&A response:
        Information: {json.dumps(selected_info)}
        User Message: {user_message}
        Analysis: {json.dumps(react_decision)}
        
        Response should:
        1. Answer the core question clearly
        2. Provide relevant context
        3. Include helpful examples
        4. Suggest related topics
        """
        
        return self.manager.get_gpt_response(response_prompt)

    def generate_safe_qa_fallback(self, category: str) -> str:
        """Generate safe fallback response for Q&A failures"""
        return ("I'd like to help answer your question more effectively. "
                "Could you provide more details? What specific information "
                "are you looking for?")

    def assess_question_category(self, user_message: str, age_group: str) -> Dict:
        """Assess if question needs specialized handling"""
        assessment_prompt = f"""
        Analyze question category:
        Message: {user_message}
        Age Group: {age_group}
        
        Return JSON with:
        {{
            "needs_specialized_handler": boolean,
            "recommended_handler": "handler type",
            "confidence": 0.0-1.0,
            "reasoning": "explanation"
        }}
        """
        
        try:
            response = self.manager.get_gpt_response(assessment_prompt)
            cleaned_response = self._clean_json_response(response)
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            logging.error("Failed to parse category assessment")
            return {
                "needs_specialized_handler": False,
                "recommended_handler": "general",
                "confidence": 0.5,
                "reasoning": "Fallback due to parsing error"
            }

    def log_category_assessment(self, assessment: Dict) -> None:
        """Log category assessment results"""
        topic = assessment.get("recommended_handler", "general")
        if topic not in self.manager.user_state.topic_history:
            self.manager.user_state.topic_history[topic] = []
            
        self.manager.user_state.topic_history[topic].append({
            "timestamp": time.time(),
            "confidence": assessment["confidence"],
            "reasoning": assessment["reasoning"]
        })

    def generate_transition_message(self, assessment: Dict) -> str:
        """Generate smooth transition message for redirection"""
        return f"""I notice you're asking about a specific topic. Let me connect you with our 
        specialized {assessment['recommended_handler']} expert to better assist you."""

    def generate_qa_context_notes(self, question_analysis: Dict, age_group: str) -> str:
        """Generate contextual notes for Q&A responses"""
        if not question_analysis.get("requires_context"):
            return ""
            
        context_prompt = f"""
        Create helpful context notes for:
        Analysis: {json.dumps(question_analysis)}
        Age Group: {age_group}
        
        Notes should:
        1. Add relevant background information
        2. Explain age-specific considerations
        3. Suggest related topics to explore
        """
        
        return self.manager.get_gpt_response(context_prompt)