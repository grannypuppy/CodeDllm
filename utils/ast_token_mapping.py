"""
AST-based token-to-node mapping for contextual tokenization.

Given: full response string = "Here is the optimized code:\n```python\n" + tgt_code + "\n```" + eos
We tokenize the FULL string (contextual tokenization), then map each response token
to its corresponding AST node in tgt_code. This enables node-level masking.
"""

from typing import Any, Dict, List, Optional, Tuple

from tree_sitter import Language, Parser
import tree_sitter_python

PYTHON_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PYTHON_LANGUAGE)

# Import get_offsets_for_slow_tokenizer from ast_dag for slow tokenizers
try:
    from utils.ast_dag import get_offsets_for_slow_tokenizer
except ImportError:
    try:
        from CodeDllm.utils.ast_dag import get_offsets_for_slow_tokenizer
    except ImportError:
        get_offsets_for_slow_tokenizer = None


def _compute_node_depth(root, node, depth_map: Dict, current_depth: int = 0):
    """Recursively compute depth of each node. Modifies depth_map in place."""
    node_id_val = id(node)
    if node_id_val not in depth_map:
        depth_map[node_id_val] = current_depth
    for child in node.children:
        _compute_node_depth(root, child, depth_map, current_depth + 1)


def get_offsets_for_full_string(full_string: str, tokenizer) -> Tuple[List, List[Tuple[int, int]]]:
    """
    Get tokens and offset_mapping for full_string.
    Works with both fast and slow tokenizers (reuses ast_dag.get_offsets_for_slow_tokenizer).
    """
    try:
        encoding = tokenizer(full_string, return_offsets_mapping=True, add_special_tokens=False)
        offset_mapping = encoding["offset_mapping"]
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        if offset_mapping:
            return tokens, offset_mapping
    except Exception:
        pass
    if get_offsets_for_slow_tokenizer is not None:
        tokens, offset_mapping = get_offsets_for_slow_tokenizer(full_string, tokenizer)
        return tokens, offset_mapping
    raise RuntimeError("Need get_offsets_for_slow_tokenizer from ast_dag for slow tokenizer")


def _collect_leaf_nodes(node, leaves: List):
    """Collect all leaf nodes (nodes with no children) from AST."""
    if node.child_count == 0:
        leaves.append(node)
    else:
        for child in node.children:
            _collect_leaf_nodes(child, leaves)


def get_response_token_to_node_mapping(
    tgt_code: str,
    full_response_string: str,
    code_start_in_full: int,
    tokenizer,
    max_response_tokens: Optional[int] = None,
) -> Tuple[Dict[int, List[int]], List[Dict[str, Any]]]:
    """
    Map response tokens to AST nodes in tgt_code.

    Args:
        tgt_code: The Python code string (target).
        full_response_string: Full response = prefix + tgt_code + suffix.
        code_start_in_full: Character offset where tgt_code starts in full_response_string.
        tokenizer: HF tokenizer (used with full_response_string for contextual tokenization).
        max_response_tokens: If set, only include token indices < this (for truncated responses).

    Returns:
        node_to_token_indices: Dict mapping node_id (int) -> list of token indices (0-based in response).
        ast_node_info: List of full node info dicts, one per leaf node. Each dict contains:
            - node_id: int
            - type: str (AST node type)
            - start_byte, end_byte: int (in tgt_code)
            - start_point: (line, column) 0-based
            - end_point: (line, column) 0-based
            - text: str (code span)
            - depth: int (depth in AST)
            - token_indices: list[int] (response-local token indices)
    """
    code_end_in_full = code_start_in_full + len(tgt_code)

    try:
        tree = parser.parse(bytes(tgt_code, "utf-8"))
        root = tree.root_node
    except Exception:
        return {}, []

    leaves = []
    _collect_leaf_nodes(root, leaves)
    if not leaves:
        return {}, []

    depth_map: Dict[int, int] = {}
    _compute_node_depth(root, root, depth_map)

    tokens, offset_mapping = get_offsets_for_full_string(full_response_string, tokenizer)

    node_to_token_indices: Dict[int, List[int]] = {}
    ast_node_info: List[Dict[str, Any]] = []

    for node_id, node in enumerate(leaves):
        node_start_full = code_start_in_full + node.start_byte
        node_end_full = code_start_in_full + node.end_byte
        token_indices_this: List[int] = []

        for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if max_response_tokens is not None and tok_idx >= max_response_tokens:
                continue
            if tok_start == tok_end:
                continue
            overlap_start = max(tok_start, node_start_full)
            overlap_end = min(tok_end, node_end_full)
            if overlap_start < overlap_end:
                if node_id not in node_to_token_indices:
                    node_to_token_indices[node_id] = []
                node_to_token_indices[node_id].append(tok_idx)
                token_indices_this.append(tok_idx)

        node_text = tgt_code[node.start_byte : node.end_byte]
        depth = depth_map.get(id(node), 0)

        ast_node_info.append({
            "node_id": node_id,
            "type": node.type,
            "start_byte": node.start_byte,
            "end_byte": node.end_byte,
            "start_point": (node.start_point[0], node.start_point[1]),
            "end_point": (node.end_point[0], node.end_point[1]),
            "text": node_text,
            "depth": depth,
            "token_indices": token_indices_this,
        })

    return node_to_token_indices, ast_node_info


def get_node_based_masking(
    node_to_token_indices: Dict[int, List[int]],
    num_nodes: int,
    ratio: float,
    response_start_pos: int,
    device,
):
    """
    Select `ratio` fraction of nodes, return the set of token indices (global) to mask.

    Args:
        node_to_token_indices: node_id -> list of token indices in response (0-based).
        num_nodes: Total number of nodes.
        ratio: Fraction of nodes to mask (0 to 1).
        response_start_pos: Start position of response in full sequence.
        device: torch device.

    Returns:
        token_indices_to_mask: Set of global token indices to mask, or None if no valid nodes.
    """
    import torch

    if not node_to_token_indices or num_nodes <= 0:
        return None

    num_to_mask = max(1, int(ratio * num_nodes))
    node_ids = list(node_to_token_indices.keys())
    if num_to_mask >= len(node_ids):
        selected_nodes = set(node_ids)
    else:
        perm = torch.randperm(len(node_ids), device=device)
        selected_nodes = {node_ids[i] for i in perm[:num_to_mask].tolist()}

    token_indices_set = set()
    for nid in selected_nodes:
        for tok_idx in node_to_token_indices[nid]:
            token_indices_set.add(response_start_pos + tok_idx)

    return token_indices_set


def get_depth_based_masking(
    ast_node_info: List[Dict[str, Any]],
    num_response_tokens: int,
    ratio: float,
    response_start_pos: int,
) -> Optional[List[int]]:
    """
    Select `ratio` fraction of response tokens to mask, based on AST depth ordering.
    Tokens with deeper depth (higher depth value = lower in tree) are masked first.

    Args:
        ast_node_info: List of node info dicts from get_response_token_to_node_mapping,
            each with "depth" and "token_indices".
        num_response_tokens: Total number of response tokens (maskable region).
        ratio: Fraction of tokens to mask (0 to 1).
        response_start_pos: Start position of response in full sequence.

    Returns:
        List of global token indices to mask, or None if no valid tokens.
    """
    if num_response_tokens <= 0:
        return None

    # Build token_idx (0-based in response) -> depth
    # Tokens not in any AST node (e.g. prefix/suffix) get depth=0 (masked last)
    token_depth = [0] * num_response_tokens
    for info in ast_node_info:
        d = info.get("depth", 0)
        for tok_idx in info.get("token_indices", []):
            if 0 <= tok_idx < num_response_tokens:
                token_depth[tok_idx] = d

    num_to_mask = max(1, int(ratio * num_response_tokens))
    if num_to_mask >= num_response_tokens:
        return [response_start_pos + i for i in range(num_response_tokens)]

    # Sort by depth descending (deeper first); break ties by position (earlier first)
    indices_with_depth = [(i, token_depth[i]) for i in range(num_response_tokens)]
    indices_with_depth.sort(key=lambda x: (-x[1], x[0]))

    selected = [response_start_pos + indices_with_depth[i][0] for i in range(num_to_mask)]
    return selected


def get_code_token_indices_set(ast_node_info: List[Dict[str, Any]], max_token_idx: int) -> set:
    """Collect all response-local token indices that belong to code (have AST mapping)."""
    code_set = set()
    for info in ast_node_info:
        for tok_idx in info.get("token_indices", []):
            if 0 <= tok_idx < max_token_idx:
                code_set.add(tok_idx)
    return code_set


def get_depth_based_code_masking(
    ast_node_info: List[Dict[str, Any]],
    code_token_indices: set,
    num_to_mask_code: int,
    response_start_pos: int,
) -> List[int]:
    """
    Select `num_to_mask_code` code tokens by depth descending (deeper first).
    Only considers tokens in code_token_indices (response-local).

    Returns:
        List of global token indices to mask.
    """
    if num_to_mask_code <= 0 or not code_token_indices:
        return []

    token_depth: Dict[int, int] = {}
    for info in ast_node_info:
        d = info.get("depth", 0)
        for tok_idx in info.get("token_indices", []):
            if tok_idx in code_token_indices:
                token_depth[tok_idx] = d

    code_list = list(code_token_indices)
    if num_to_mask_code >= len(code_list):
        return [response_start_pos + i for i in code_list]

    indices_with_depth = [(i, token_depth.get(i, 0)) for i in code_list]
    indices_with_depth.sort(key=lambda x: (-x[1], x[0]))

    selected = [response_start_pos + indices_with_depth[i][0] for i in range(num_to_mask_code)]
    return selected
