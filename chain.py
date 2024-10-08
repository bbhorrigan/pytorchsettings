# Chain of Thought Simulator

def understand_question(question):
    """
    Step 1: Understand the question's intent, context, and details.
    """
    if "explain" in question.lower():
        return "explanation"
    elif "what is" in question.lower() or "who is" in question.lower():
        return "factual"
    elif "generate" in question.lower() or "create" in question.lower():
        return "creative"
    elif "calculate" in question.lower() or "analyze" in question.lower():
        return "analytical"
    else:
        return "unknown"

def choose_approach(intent):
    """
    Step 2: Choose the right approach based on the question's intent.
    """
    if intent == "explanation":
        return "Breaking it down step by step..."
    elif intent == "factual":
        return "Looking up specific details and facts..."
    elif intent == "creative":
        return "Generating something based on a creative prompt..."
    elif intent == "analytical":
        return "Performing calculations or analyzing data..."
    else:
        return "Not sure what you're asking. Could you clarify?"

def generate_response(approach):
    """
    Step 3: Generate a response based on the approach chosen.
    """
    responses = {
        "Breaking it down step by step...": "Let me explain this in detail...",
        "Looking up specific details and facts...": "Here’s what I found based on available data...",
        "Generating something based on a creative prompt...": "Let’s create something unique...",
        "Performing calculations or analyzing data...": "Crunching the numbers for you...",
        "Not sure what you're asking. Could you clarify?": "Could you give more details?"
    }
    return responses.get(approach, "Hmm, I need to rethink this.")

def main():
    print("Welcome to the Chain of Thought Simulator!")
    question = input("Please enter your question: ")

    # Step 1: Understand the question
    intent = understand_question(question)
    print(f"Step 1: Understanding the question... (Intent: {intent})")

    # Step 2: Choose an approach
    approach = choose_approach(intent)
    print(f"Step 2: Choosing the approach... (Approach: {approach})")

    # Step 3: Generate a response
    response = generate_response(approach)
    print(f"Step 3: Generating the response... (Response: {response})")

if __name__ == "__main__":
    main()
