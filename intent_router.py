import json
import os
from typing import Dict, Any, Optional, List
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


INTENT_CATALOG = {
    "price": {
        "description": "User wants to find the price or cost of a product or service",
        "examples": [
            "How much does iPhone 15 cost?",
            "What's the price of Tesla Model 3?",
            "Find me the cheapest laptop"
        ],
        "tool": "search_web"
    },
    "news": {
        "description": "User wants current news or recent events about a topic",
        "examples": [
            "What's the latest news about AI?",
            "Tell me today's headlines",
            "What happened in the election?"
        ],
        "tool": "search_web"
    },
    "translate": {
        "description": "User wants to translate text from one language to another",
        "examples": [
            "Translate 'hello' to Spanish",
            "How do you say 'thank you' in French?",
            "What does 'bonjour' mean?"
        ],
        "tool": "translate"
    },
    "weather": {
        "description": "User wants weather information for a location",
        "examples": [
            "What's the weather in New York?",
            "Will it rain today?",
            "Temperature in London"
        ],
        "tool": "get_current_weather"
    },
    "math": {
        "description": "User wants to perform mathematical calculations or solve equations",
        "examples": [
            "What is 2+2?",
            "Calculate 15% of 200",
            "Solve x^2 + 5x + 6 = 0"
        ],
        "tool": "solve_math"
    },
    "email": {
        "description": "User wants to send, read, or manage emails",
        "examples": [
            "Send an email to John",
            "Check my inbox",
            "Draft an email about the meeting"
        ],
        "tool": "gmail"
    },
    "general_question": {
        "description": "User has a general knowledge question that doesn't require tools",
        "examples": [
            "Who was the first president?",
            "Explain quantum physics",
            "What is photosynthesis?"
        ],
        "tool": None
    }
}


class IntentRouter:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize Intent Router with Groq API
        
        Recommended models:
        - llama-3.3-70b-versatile (best for complex reasoning)
        - llama-3.1-70b-versatile (fast and accurate)
        - mixtral-8x7b-32768 (good balance of speed/accuracy)
        - llama-3.1-8b-instant (fastest, for simple intents)
        """
        self.client = Groq(api_key=api_key)
        self.model = model
        self.intent_catalog = INTENT_CATALOG
    
    def _build_classification_prompt(self, query: str) -> str:
        """Build a structured prompt for intent classification"""
        
        intent_descriptions = "\n".join([
            f"- {intent}: {config['description']}\n  Examples: {', '.join(config['examples'][:2])}"
            for intent, config in self.intent_catalog.items()
        ])
        
        return f"""You are an intent classification system. Analyze the user query and determine the most appropriate intent.

Available intents:
{intent_descriptions}

User query: "{query}"

Respond with ONLY a valid JSON object (no markdown, no explanation) containing:
- "intent": the matched intent name (must be one of: {', '.join(self.intent_catalog.keys())})
- "confidence": a number between 0 and 1 indicating your confidence
- "entities": any relevant entities extracted as a JSON object
- "reasoning": brief explanation of why you chose this intent

Example response format:
{{"intent": "math", "confidence": 0.95, "entities": {{"operation": "addition", "numbers": [2, 2]}}, "reasoning": "Clear mathematical calculation request"}}

Your JSON response:"""

    def detect_intent(self, query: str, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Detect intent from user query using LLM. For example: if the query is "what is 2+2", the intent is "additon" and "number:[2,2]" with a reasoning:"Clear mathematical calculation request".
        
        Args:
            query: User's input query
            temperature: Lower = more deterministic (0.1 recommended for classification)
        """
        
        if not query or not query.strip():
            return {
                "intent": "unknown",
                "confidence": 46.0,
                "entities": {},
                "tool": None,
                "needs_clarification": True,
                "clarification_question": "What would you like to do?",
                "reasoning": "Empty query"
            }
        
        try:
            # Call Groq API for intent classification
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert intent classification system. Always respond with valid JSON only, no other text."
                    },
                    {
                        "role": "user",
                        "content": self._build_classification_prompt(query)
                    }
                ],
                model=self.model,
                temperature=temperature,
                max_tokens=512,
            )
            
            # Parse the response
            response_text = chat_completion.choices[0].message.content.strip()
            
            # Clean up potential markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            # Validate intent is in catalog
            intent = result.get("intent", "unknown")
            if intent not in self.intent_catalog:
                intent = "general_question"  # fallback
                result["intent"] = intent
            
            # Enrich with tool mapping
            result["tool"] = self.intent_catalog.get(intent, {}).get("tool")
            result["needs_clarification"] = result.get("confidence", 0) < 0.6
            
            if result["needs_clarification"]:
                result["clarification_question"] = f"Did you want to {self.intent_catalog.get(intent, {}).get('description', 'do something')}?"
            else:
                result["clarification_question"] = None
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {response_text}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "tool": None,
                "error": f"Failed to parse response: {str(e)}",
                "needs_clarification": True,
                "clarification_question": "I'm having trouble understanding. Can you rephrase?",
                "reasoning": "Parse error"
            }
        except Exception as e:
            print(f"Error in intent detection: {e}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "entities": {},
                "tool": None,
                "error": str(e),
                "needs_clarification": True,
                "clarification_question": "I'm having trouble understanding. Can you rephrase?",
                "reasoning": "API error"
            }
    
    def add_intent(self, intent_name: str, description: str, examples: List[str], tool: Optional[str] = None):
        """Dynamically add new intents to the catalog"""
        self.intent_catalog[intent_name] = {
            "description": description,
            "examples": examples,
            "tool": tool
        }
        print(f"Added intent: {intent_name}")
    
    def remove_intent(self, intent_name: str):
        """Remove an intent from the catalog"""
        if intent_name in self.intent_catalog:
            del self.intent_catalog[intent_name]
            print(f"✓ Removed intent: {intent_name}")
    
    def list_intents(self):
        """List all available intents"""
        return list(self.intent_catalog.keys())


# Global router instance for standalone functions
_router_instance = None


def get_router_instance():
    """Get or create the global IntentRouter instance"""
    global _router_instance
    if _router_instance is None:
        # Try to get Groq API key from environment
        api_keys = [
            os.getenv("GROQ_API_KEY_1"),
            os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"),
            os.getenv("GROQ_API_KEY")
        ]
        
        api_key = next((key for key in api_keys if key), None)
        
        if not api_key:
            raise ValueError(
                "No Groq API key found in environment. "
                "Please set GROQ_API_KEY, GROQ_API_KEY_1, GROQ_API_KEY_2, or GROQ_API_KEY_3 in .env file"
            )
        
        _router_instance = IntentRouter(
            api_key=api_key,
            model="llama-3.3-70b-versatile"
        )
    
    return _router_instance


def detect_intent(query: str, temperature: float = 0.1) -> Dict[str, Any]:
    """
    Standalone function to detect intent using the global router instance.
    Compatible with existing code that imports this function.
    
    Args:
        query: User's input query
        temperature: Lower = more deterministic (0.1 recommended for classification)
    
    Returns:
        Dict containing intent, confidence, entities, tool, etc.
    """
    router = get_router_instance()
    return router.detect_intent(query, temperature)


def resolve_tool(intent_name: str) -> Optional[str]:
    """
    Resolve the tool name for a given intent.
    
    Args:
        intent_name: The intent name (e.g., "math", "weather", "email")
    
    Returns:
        Tool name string or None
    """
    router = get_router_instance()
    return router.intent_catalog.get(intent_name, {}).get("tool")


if __name__ == "__main__":
    # Initialize router with Groq API from environment
    api_keys = [
        os.getenv("GROQ_API_KEY_1"),
        os.getenv("GROQ_API_KEY_2"),
        os.getenv("GROQ_API_KEY_3"),
        os.getenv("GROQ_API_KEY")
    ]
    
    api_key = next((key for key in api_keys if key), None)
    
    if not api_key:
        print("ERROR: No Groq API key found!")
        exit(1)
    
    router = IntentRouter(
        api_key=api_key,
        model="llama-3.3-70b-versatile"  # or "llama-3.1-8b-instant" for faster responses
    )
    
    # Test queries
    test_queries = [
        "what is 2+2",
        "How much does iPhone 15 cost?",
        "Translate hello to Spanish",
        "What's the weather in Paris?",
        "Send email to boss about quarterly report",
        "Who invented the telephone?",
        "What's happening in tech news today?",
        "Calculate 15% tip on $87.50"
    ]
    
    print("=" * 60)
    print("INTENT ROUTER TEST")
    print("=" * 60)
    
    for query in test_queries:
        result = router.detect_intent(query)
        print(f"\n Query: {query}")
        print(f" Intent: {result['intent']}")
        print(f" Confidence: {result['confidence']:.2f}")
        print(f" Tool: {result.get('tool', 'None')}")
        print(f" Entities: {result.get('entities', {})}")
        print(f" Reasoning: {result.get('reasoning', 'N/A')}")
        if result.get('needs_clarification'):
            print(f" Clarification: {result.get('clarification_question')}")
        print("-" * 60)
    
    # Example: Adding a custom intent dynamically
    print("\n\n Adding custom intent...")
    router.add_intent(
        intent_name="reminder",
        description="User wants to set a reminder or schedule something",
        examples=[
            "Remind me to call mom at 5pm",
            "Set a reminder for tomorrow",
            "Schedule a meeting for next week"
        ],
        tool="calendar"
    )
    
    # Test the new intent
    result = router.detect_intent("Remind me to buy groceries")
    print(f"\n Query: Remind me to buy groceries")
    print(f" Intent: {result['intent']}")
    print(f" Tool: {result.get('tool')}")