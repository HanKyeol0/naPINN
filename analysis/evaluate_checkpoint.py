# Run via: python -m analysis.evaluate_checkpoint

import argparse
import json
import os

import torch
import yaml

from pinnlab.registry import get_experiment, get_model
from pinnlab.utils.seed import seed_everything

def main(args):
    folder_path = args.folder_path
    device = args.device if torch.cuda.is_available() else "cpu"
    
    def str2bool(v):
        return str(v).lower() in ("yes", "true", "t", "1")
    
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
    
    checkpoint_path = os.path.join(folder_path, "final.pt")
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from: {checkpoint_path}")
    
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        try:
            model.load_state_dict(ckpt)
        except RuntimeError:
            print("Warning: Could not find model state dict in checkpoint.")

    if hasattr(exp, "load_state_dict"):
        if "experiment" in ckpt:
            print("Restoring experiment state (EBM, Gate, Std)...")
            exp.load_state_dict(ckpt["experiment"])
        elif "experiment_state_dict" in ckpt:
            exp.load_state_dict(ckpt["experiment_state_dict"])
        else:
            print("Warning: No experiment state found in checkpoint. EBM/Gate will use random init.")

    if do_evaluate:
        print("Starting Evaluation...")

        grid = base_cfg["eval"]["grid"]
        model.eval()
        with torch.no_grad():
            eval_result  = exp.eval_on_grid(model, grid)
            rMAE, rMSE = eval_result["rMAE"], eval_result["rMSE"]
        print(f"rMAE: {rMAE:.5e}, rMSE: {rMSE:.5e}")
        
        if hasattr(exp, "evaluate_gate_performance"):
            print("Evaluating Gate Performance...")
            exp.evaluate_gate_performance(model=model, out_dir=folder_path, filename_prefix="remade")
        else:
            print("Experiment does not support 'evaluate_gate_performance'. Skipping.")

    if do_make_video:
        model.eval()
        print("making video...")
        if args.video_grid:
            grid = args.video_grid
            vid_grid = {'nx': grid['nx'], 'ny': grid['ny'], 'nt': grid['nt']}
        else:
            vid_grid = dict(base_cfg["eval"]["grid"])
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
    parser.add_argument("--evaluate", required=True)
    parser.add_argument("--make_video", required=True)
    parser.add_argument("--video_grid", type=json.loads, required=False)
    parser.add_argument("--video_file_name", required=True)
    args = parser.parse_args()
    main(args)
