import argparse
import shutil
from pathlib import Path
import torch

def convert_file(path: Path):
    data = torch.load(path)
    if "depth_labels" in data:
        print(f"{path.name}: already has depth_labels, skip.")
        return

    required_keys = ["input_ids", "labels", "p_mask_lm", "start_pos", "ast_node_info", "sample_to_original_idx"]
    for k in required_keys:
        if k not in data:
            print(f"{path.name}: missing key {k}, skip.")
            return

    input_ids = data["input_ids"]
    B, L = input_ids.shape
    start_pos = data["start_pos"]
    ast_node_info = data["ast_node_info"]  # list per original sample
    sample_to_original_idx = data["sample_to_original_idx"]

    # ensure list form
    if isinstance(sample_to_original_idx, torch.Tensor):
        sample_to_original_idx = sample_to_original_idx.tolist()

    depth_labels = torch.full((B, L), -1.0, dtype=torch.float)

    for i in range(B):
        orig = sample_to_original_idx[i]
        if orig is None:
            raise ValueError(f"Sample {i} has no original index mapping.")
        nodes = ast_node_info[orig]
        resp_start = int(start_pos[i].item()) if isinstance(start_pos[i], torch.Tensor) else int(start_pos[i])
        for node in nodes:
            depth = node.get("depth", None)
            if depth is None:
                raise ValueError(f"Node in sample {i} has no depth info.")
            token_idxs = node.get("token_indices", []) or []
            for tok in token_idxs:
                full_idx = resp_start + int(tok)
                if 0 <= full_idx < L:
                    depth_labels[i, full_idx] = float(depth)

    # backup and save
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    data["depth_labels"] = depth_labels
    torch.save(data, path)
    print(f"{path.name}: converted, backup => {bak.name}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", help="preprocessed dir containing train.pt / val.pt")
    args = p.parse_args()
    d = Path(args.dir)
    if not d.is_dir():
        print("Not a directory:", d)
        return
    for fname in ["train.pt", "val.pt"]:
        f = d / fname
        if f.exists():
            convert_file(f)
        else:
            print(f"{fname} not found in {d}")

if __name__ == "__main__":
    main()