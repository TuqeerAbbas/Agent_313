from managers import ConversationManager
from utils.logging_utils import setup_logging
import logging

def main():
    # Setup logging with more detailed format
    setup_logging()
    
    # Initialize conversation manager
    manager = ConversationManager()
    
    print("\nNestlé Baby and Me AI Agent")
    print("Type 'exit' to quit")
    
    try:
        while True:
            user_input = input("\nUser: ").strip()
            
            if user_input.lower() == 'exit':
                break
                
            # Log the start of processing
            print("\nProcessing your request...")
            
            # Get and display response
            response = manager.handle_conversation(user_input)
            print("\nAgent:", response)
            
            # Display separator for readability
            print("\n" + "-"*50)
            
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        manager.cleanup()

if __name__ == "__main__":
    main()