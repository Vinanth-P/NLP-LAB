# --- Part 1: DFA Logic ---

def process_dfa(input_string):
    # Handle empty string edge case
    if not input_string:
        return "Not Accepted"

    state = "q0"
    
    for char in input_string:
        # Determine the character class
        is_lower = 'a' <= char <= 'z'
        
        # State transitions
        if state == "q0":
            if is_lower:
                state = "q1"
            else:
                state = "q2"
        elif state == "q1":
            if is_lower:
                state = "q1"
            else:
                state = "q2"
        elif state == "q2":
            # Once in the trap state, we can't escape
            break 

    # Check final state
    if state == "q1":
        return "Accepted"
    else:
        return "Not Accepted"

# Test Cases
test_words = ["cat", "dog", "a", "zebra", "dog1", "1dog", "DogHouse", "Dog_house", " cats"]

print("--- DFA Output ---")
for word in test_words:
    print(f"'{word}': {process_dfa(word)}")


# --- Part 2: Visualization using visual-automata ---
# Note: You must `pip install visual-automata` first.

try:
    from visual_automata.fa.dfa import VisualDFA
    
    # Define the DFA for visualization
    # We use "a-z" to represent lowercase and "other" to represent invalid characters
    dfa = VisualDFA(
        states={"q0", "q1", "q2"},
        input_symbols={"a-z", "other"},
        transitions={
            "q0": {"a-z": "q1", "other": "q2"},
            "q1": {"a-z": "q1", "other": "q2"},
            "q2": {"a-z": "q2", "other": "q2"}
        },
        initial_state="q0",
        final_states={"q1"}
    )
    
    # This will display the graph in a Jupyter Notebook or generate a visual representation
    dfa.show_diagram()
    print("\nVisualization generated successfully.")
except ImportError:
    print("\nNote: Install 'visual-automata' (pip install visual-automata) to see the diagram.")