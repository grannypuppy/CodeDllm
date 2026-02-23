import os
import numpy as np
from tree_sitter import Language, Parser
import tree_sitter_python
from transformers import AutoTokenizer
import graphviz

# --- Configuration ---
    # Example code snippet to analyze
sample_code = """
A, B, C, X, Y = list[int](map(int, input().split<|mask|>ans = 5000 * 2 * 10 ** 5<|mask|>for i in range(10 ** 5 + 1):
    s = i * 2 * C + max(0, X - i) * A + max(0, Y - i) * B
    ans = min(ans, s<|mask|>print(ans)
"""

# 1. Initialize the tree-sitter parser for Python
PYTHON_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PYTHON_LANGUAGE)

# --- Core Implementation ---

def find_best_matching_token(node_span, tokens, offset_mapping):
    """
    Finds the token that best matches a given node's character span.
    The "best match" is defined as the token with the maximum overlap.
    """
    node_start, node_end = node_span
    best_match_token = ""
    max_overlap = -1

    for i, (tok_start, tok_end) in enumerate(offset_mapping):
        # Calculate the overlapping region
        overlap = max(0, min(node_end, tok_end) - max(node_start, tok_start))
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_match_token = tokens[i]

    # Clean up special characters from the token for display
    # return best_match_token.replace('Ġ', ' ').replace('Ċ', '\\n')
    return best_match_token

def add_ast_nodes_edges(node, dot, tokens, offset_mapping):
    """Recursively traverses the AST to build a Graphviz graph."""
    node_id = str(id(node))
    
    if node.child_count == 0:  # This is a leaf node
        node_span = (node.start_byte, node.end_byte)
        token_text = find_best_matching_token(node_span, tokens, offset_mapping)    
        # Create a label with both node type and matched token
        # Escape special characters for Graphviz using backslash
        node_label = f'{node.type}\n{token_text}\n{sample_code[node.start_byte:node.end_byte]}\n({node.start_byte},{node.end_byte})'
        dot.node(node_id, label=node_label, shape='box', style='filled', fillcolor='lightblue')
    else:  # This is an intermediate node
        node_label = node.type
        dot.node(node_id, label=node_label, shape='ellipse')
        
    # Add edges to children and recurse
    for child in node.children:
        child_id = str(id(child))
        dot.edge(node_id, child_id)
        add_ast_nodes_edges(child, dot, tokens, offset_mapping)

def visualize_ast_with_tokens(code_string: str, tokenizer, output_filename="ast_with_tokens"):
    """
    Generates a visualization of the AST with leaf nodes mapped to tokens.
    
    Args:
        code_string (str): The Python code to analyze.
        tokenizer: The tokenizer to use for splitting code into tokens.
        output_filename (str): The path (without extension) to save the output graph.
    """
    print(f"\n--- Visualizing AST and Token Mapping ---")
    
    # 1. Parse code to get the AST root
    tree = parser.parse(bytes(code_string, "utf8"))
    root_node = tree.root_node
    
    # 2. Tokenize the code to get tokens and their character offsets
    if tokenizer.is_fast:
        encoding = tokenizer(code_string, return_offsets_mapping=True)
        offset_mapping = encoding["offset_mapping"]
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    else:
        tokens, offset_mapping = get_offsets_for_slow_tokenizer(code_string, tokenizer)

    # 3. Create and build the graph using Graphviz
    dot = graphviz.Digraph('AST', comment='Abstract Syntax Tree with Tokens')
    dot.attr(rankdir='TB')

    add_ast_nodes_edges(root_node, dot, tokens, offset_mapping)

    # 4. Render and save the graph to a file
    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        dot.render(output_filename, format='png', view=False, cleanup=False)        
        print(f"AST visualization saved to '{output_filename}.png'")
    except Exception as e:
        print(f"\n--- Graphviz Error ---")
        print(f"Error generating AST visualization: {e}")

def get_offsets_for_slow_tokenizer(code_string: str, tokenizer):
    """
    Manually creates an offset mapping for a slow tokenizer by searching for
    each token in the original string. This is a workaround for tokenizers
    that do not support `return_offsets_mapping=True`.
    """
    tokens = tokenizer.tokenize(code_string)
    
    offset_mapping = []
    current_pos = 0
    for token in tokens:
        # GPT-2/BPE tokenizers use 'Ġ' to represent a space.
        # We need to convert this to a findable string.
        findable_token = token.replace('Ġ', ' ').replace('Ċ', '\n')
        
        try:
            # Find the token in the string from the current position
            start = code_string.index(findable_token, current_pos)
            end = start + len(findable_token)
            offset_mapping.append((start, end))
            current_pos = end
        except ValueError:
            # This can happen if tokenization rules are complex.
            # As a fallback, we'll just map to an empty range.
            offset_mapping.append((current_pos, current_pos))
            current_pos += 1

    return tokens, offset_mapping

def find_max_depth(node, current_depth):
    """Recursively finds the maximum depth of the AST."""
    if node.child_count == 0:
        return current_depth
    
    max_child_depth = current_depth
    for child in node.children:
        child_depth = find_max_depth(child, current_depth + 1)
        if child_depth > max_child_depth:
            max_child_depth = child_depth
    
    return max_child_depth

def apply_leaf_node_weights(node, char_weights_array, depth):
    """
    Recursively traverses the AST, applying depth weight only to leaf nodes.

    Args:
        node: The current tree-sitter AST node to process.
        char_weights_array (np.ndarray): A NumPy array to store weights for each character.
        depth (int): The current depth of the `node` in the AST.
    """
    # Only leaf nodes (nodes with no children) assign their depth as a weight.
    if node.child_count == 0:
        start = node.start_byte
        end = node.end_byte
        char_weights_array[start:end] = depth
    else:
        # Intermediate nodes just pass the traversal down to their children.
        for child in node.children:
            apply_leaf_node_weights(child, char_weights_array, depth + 1)

def generate_token_weights_from_dag(code_string: str, tokenizer):
    """
    Generates an order-weight for each token based on the code's AST leaf node depth.

    Args:
        code_string (str): The input Python code snippet to analyze.
        tokenizer: An initialized Hugging Face tokenizer (fast or slow).

    Returns:
        A tuple containing:
        - tokens (list[str]): The list of generated token strings.
        - token_ids (list[int]): The list of corresponding token IDs.
        - token_weights (list[float]): The final calculated order-weight for each token.
        - offset_mapping (list[tuple[int, int]]): The character offsets for each token.
    """
    tree = parser.parse(bytes(code_string, "utf8"))
    root_node = tree.root_node
    
    # 1. Find the maximum depth of the entire tree.
    max_depth = find_max_depth(root_node, current_depth=1)
    
    # 2. Create a char-level weight array, initialized with a penalty weight.
    # This high value signifies the lowest priority for non-leaf characters.
    penalty_weight = max_depth + 1
    char_weights = np.full(len(code_string), penalty_weight, dtype=float)
    
    # 3. Apply weights only from leaf nodes.
    apply_leaf_node_weights(root_node, char_weights, depth=1)
    
    # Apply maximum depth weight to all whitespace characters (spaces and newlines)
    # Apply maximum depth weight to consecutive whitespace characters (3 or more)
    i = 0
    while i < len(code_string):
        if code_string[i] in (' ', '\n', '\t', '\r'):
            # Found a whitespace character, check if there are consecutive ones
            start = i
            while i < len(code_string) and code_string[i] in (' ', '\n', '\t', '\r'):
                i += 1
            # If we have 3 or more consecutive whitespace characters
            if i - start >= 3:
                char_weights[start:i] = penalty_weight
        else:
            i += 1

    # 5. Tokenize the code and get the offset mapping.
    if tokenizer.is_fast:
        encoding = tokenizer(
            code_string,
            return_offsets_mapping=True,
        )
        token_ids = encoding["input_ids"]
        offset_mapping = encoding["offset_mapping"]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
    else:
        # print("Warning: Using slow tokenizer. Manually creating offset mapping.")
        token_ids = tokenizer.encode(code_string)
        tokens, offset_mapping = get_offsets_for_slow_tokenizer(code_string, tokenizer)

    # 6. Aggregate character weights into token weights using the 'max' strategy.
    token_weights = []
    for start, end in offset_mapping:
        if start == end:
            token_weights.append(penalty_weight)
            continue
        
        relevant_char_weights = char_weights[start:end]
        token_weight = np.min(relevant_char_weights)
        token_weights.append(token_weight)
        
    return tokens, token_ids, token_weights, offset_mapping

def generate_char_weights_from_dag(code_string: str):
    """
    Generates an order-weight for each token based on the code's AST leaf node depth.

    Args:
        code_string (str): The input Python code snippet to analyze.
        tokenizer: An initialized Hugging Face tokenizer (fast or slow).

    Returns:
        A tuple containing:
        - tokens (list[str]): The list of generated token strings.
        - token_ids (list[int]): The list of corresponding token IDs.
        - token_weights (list[float]): The final calculated order-weight for each token.
        - offset_mapping (list[tuple[int, int]]): The character offsets for each token.
    """
    tree = parser.parse(bytes(code_string, "utf8"))
    root_node = tree.root_node
    
    # 1. Find the maximum depth of the entire tree.
    max_depth = find_max_depth(root_node, current_depth=1)
    
    # 2. Create a char-level weight array, initialized with a penalty weight.
    # This high value signifies the lowest priority for non-leaf characters.
    penalty_weight = max_depth + 1
    char_weights = np.full(len(code_string), penalty_weight, dtype=float)
    
    # 3. Apply weights only from leaf nodes.
    apply_leaf_node_weights(root_node, char_weights, depth=1)
    
    # Apply maximum depth weight to all whitespace characters (spaces and newlines)
    # Apply maximum depth weight to consecutive whitespace characters (3 or more)
    i = 0
    while i < len(code_string):
        if code_string[i] in (' ', '\n', '\t', '\r'):
            # Found a whitespace character, check if there are consecutive ones
            start = i
            while i < len(code_string) and code_string[i] in (' ', '\n', '\t', '\r'):
                i += 1
            # If we have 3 or more consecutive whitespace characters
            if i - start >= 3:
                char_weights[start:i] = penalty_weight
        else:
            i += 1
        
    return char_weights

# --- Demonstration ---

if __name__ == "__main__":
    # Path to your local DiffuCoder model/tokenizer
    model_path = "../dLLM-RL/local_models/dream-7b-base"

    print(f"Loading tokenizer from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    except Exception as e:
        print(f"Error loading a tokenizer: {e}")

    print("\n--- Analyzing Sample Code ---")
    print(sample_code)
    print("-" * 30)
    
    # Generate the weights using the new DAG-based method
    tokens, token_ids, weights, offsets = generate_token_weights_from_dag(
        sample_code, tokenizer
    )
    
    # Print the results in a formatted table
    print(f"\n{'Token':<20} {'ID':<8} {'Offset':<12} {'Weight (Structure-First)'}")
    print("=" * 60)
    for token, token_id, weight, offset in zip(tokens, token_ids, weights, offsets):
        print(f"{token:<20} {token_id:<8} {str(offset):<12} {weight:.2f}")

    print("\nAnalysis complete. Higher weight means higher in the AST (more structural).")

    # Generate the AST visualization
    visualize_ast_with_tokens(
        sample_code, 
        tokenizer, 
        output_filename="ast_visualizations/ast_with_tokens"
    )
