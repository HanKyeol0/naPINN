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
    train = args.train
    evaluate = args.evaluate
    make_video = args.make_video

    cfg_path = os.path.join(folder_path, "config.yaml")
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
    exp_extra = ckpt.pop("_exp_extra", None)
    model.load_state_dict(ckpt)

    if make_video:
        print("making video...")
        if args.video_grid:
            grid = args.video_grid
            vid_grid = {'nx': grid['nx'], 'ny': grid['ny'], 'nt': grid['nt']}
        else:
            vid_grid = {'nx': base_cfg["eval"]['nt'], 'ny': base_cfg["eval"]['ny'], 'nt': base_cfg["eval"]['nt']}
        fps = 10
        vid_path = exp.make_video(model, vid_grid, out_dir=folder_path,
                                  filename=args.video_file_name, fps=fps)
    
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