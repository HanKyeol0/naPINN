# run: scripts/load_model_exp.sh

import os, yaml, argparse, json, torch
from pinnlab.registry import get_experiment, get_model
from pinnlab.utils.seed import seed_everything
import torch

def load_yaml(path):
    with open(path, 'r', encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(args):
    folder_path = args.folder_path
    device = args.device if torch.cuda.is_available() else "cpu"
    
    def str2bool(v):
        return str(v).lower() in ("yes", "true", "t", "1")
    
    do_train = str2bool(args.train)
    do_evaluate = str2bool(args.evaluate)
    do_make_video = str2bool(args.make_video)

    cfg_path = os.path.join(folder_path, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found at {cfg_path}")
    
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    base_cfg = cfg["base"]
    exp_cfg = cfg["experiment"]
    model_cfg = cfg["model"]

    in_features  = exp_cfg.get("in_features", model_cfg.get("in_features"))
    out_features = exp_cfg.get("out_features", model_cfg.get("out_features"))
    model_cfg["in_features"]  = in_features
    model_cfg["out_features"] = out_features

    seed_everything(base_cfg["seed"])

    exp = get_experiment(args.experiment_name)(exp_cfg, device)
    model = get_model(args.model_name)(model_cfg).to(device)
    
    checkpoint_path = os.path.join(folder_path, "best.pt")
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from: {checkpoint_path}")
    
    # A. Load Model
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # Fallback if the ckpt is just the state dict itself (rare but possible)
        try:
            model.load_state_dict(ckpt)
        except:
            print("Warning: Could not find model state dict in checkpoint.")

    # B. Load Experiment State (running_std, EBM, Gate, Offset)
    # This replaces the manual _exp_extra loop completely.
    if hasattr(exp, "load_state_dict"):
        if "experiment" in ckpt:
            print("Restoring experiment state (EBM, Gate, Std)...")
            exp.load_state_dict(ckpt["experiment"])
        elif "experiment_state_dict" in ckpt:
            exp.load_state_dict(ckpt["experiment_state_dict"])
        else:
            print("Warning: No experiment state found in checkpoint. EBM/Gate will use random init.")

    # --- EVALUATION BLOCK ---
    if do_evaluate:
        print("Starting Evaluation...")
        
        # 1. Standard Metric Evaluation (Relative L2)
        grid = base_cfg["eval"]["grid"]
        model.eval()
        with torch.no_grad():
            rel_l2 = exp.relative_l2_on_grid(model, grid)
        print(f"Relative L2 Error: {rel_l2:.5e}")
        
        # 2. Gate Performance Evaluation (Sigmoid + Confusion Matrix)
        if hasattr(exp, "evaluate_gate_performance"):
            print("Evaluating Gate Performance...")
            exp.evaluate_gate_performance(model, folder_path)
        else:
            print("Experiment does not support 'evaluate_gate_performance'. Skipping.")

    if do_make_video:
        model.eval()
        print("making video...")
        if args.video_grid:
            grid = args.video_grid
            vid_grid = {'nx': grid['nx'], 'ny': grid['ny'], 'nt': grid['nt']}
        else:
            vid_grid = {'nx': base_cfg["eval"]['nx'], 'ny': base_cfg["eval"]['ny'], 'nt': base_cfg["eval"]['nt']}
        fps = 10
        try:
            vid_path = exp.make_video(
                model, vid_grid, out_dir=folder_path,
                filename=args.video_file_name, fps=fps,
                phase=2, # for making noise analysis video
            )
            print(f"Video saved to: {vid_path}")
        except Exception as e:
            print(f"Error creating video: {e}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--folder_path", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--evaluate", required=True)
    parser.add_argument("--make_video", required=True)
    parser.add_argument("--video_grid", type=json.loads, required=False)
    parser.add_argument("--video_file_name", required=True)
    args = parser.parse_args()
    main(args)