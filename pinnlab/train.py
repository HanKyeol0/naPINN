import os, time, yaml, argparse, sys, json, math
import torch
import wandb
from tqdm import trange
from pinnlab.registry import get_model, get_experiment
from pinnlab.utils.seed import seed_everything
from pinnlab.utils.wandb_utils import setup_wandb, wandb_log, wandb_finish

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _save_yaml(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f)

def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")

def state_to_cpu(state):
    if isinstance(state, torch.Tensor):
        return state.detach().cpu()
    elif isinstance(state, dict):
        return {k: state_to_cpu(v) for k, v in state.items()}
    elif isinstance(state, list):
        return [state_to_cpu(v) for v in state]
    return state

def main(args):
    base_cfg = load_yaml(args.common_config)
    model_cfg = load_yaml(args.model_config)
    exp_cfg   = load_yaml(args.exp_config)

    # Allow experiment to override in/out dims if needed
    in_features  = exp_cfg.get("in_features", model_cfg.get("in_features"))
    out_features = exp_cfg.get("out_features", model_cfg.get("out_features"))
    model_cfg["in_features"]  = in_features
    model_cfg["out_features"] = out_features

    seed_everything(base_cfg["seed"])

    if exp_cfg.get("device"):
        base_cfg["device"] = exp_cfg["device"]
    device = torch.device(base_cfg["device"] if torch.cuda.is_available() else "cpu")
    # torch.cuda.reset_peak_memory_stats(device)
    
    exp = get_experiment(args.experiment_name)(exp_cfg, device)
    model = get_model(args.model_name)(model_cfg).to(device)
    
    tag = exp_cfg.get("tag", None)
    if tag:
        file_name = f"{args.experiment_name}_{args.model_name}_{tag}"
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"{args.experiment_name}_{args.model_name}_{ts}"
    
    out_dir = os.path.join(base_cfg["log"]["out_dir"], args.experiment_name, file_name)
    os.makedirs(out_dir, exist_ok=True)

    _save_yaml(os.path.join(out_dir, "config.yaml"), {
        "base": base_cfg, "model": model_cfg, "experiment": exp_cfg
    })
    
    # Optimizer
    params = list(model.parameters())
    if hasattr(exp, "extra_params"):
        params += list(exp.extra_params())

    opt_cfg = base_cfg["train"]["optimizer"]
    if opt_cfg["name"].lower() == "adam":
        optimizer = torch.optim.Adam(params, lr=opt_cfg["lr"], weight_decay=opt_cfg.get("weight_decay", 0.0))
    else:
        raise ValueError("Only Adam is wired in, add more in train.py.")

    # WandB
    if base_cfg["log"]["wandb"]["enabled"]:
        if exp_cfg.get("wandb_project"):
            base_cfg["log"]["wandb"]["project"] = exp_cfg["wandb_project"]
        setup_wandb(base_cfg["log"]["wandb"], args, out_dir, config={
            "base": base_cfg, "model": model_cfg, "experiment": exp_cfg
        })

    epochs = base_cfg["train"]["epochs"]
    eval_every = int(base_cfg.get("eval").get("every", 100))
    use_phase = exp_cfg["phase"]["enabled"]
    if use_phase:
        phase1_epochs = exp_cfg["phase"]["phase1_epochs"]
        phase2_epochs = exp_cfg["phase"]["phase2_epochs"]
        print(f"Using phased training: phase 1 for {phase1_epochs} epochs, phase 2 for {phase2_epochs} epochs.")

    best_metric = float("inf")
    best_model_state = None
    best_exp_state = None

    w_res = base_cfg["train"]["loss_weights"]["res"]
    w_data = base_cfg["train"]["loss_weights"]["data"]

    n_f = exp_cfg.get("batch", {}).get("n_f", base_cfg["train"]["batch"]["n_f"])
    
    # Make video
    enable_video = exp_cfg.get("video", {}).get("enabled", False)
    make_video_every = exp_cfg.get("video", {}).get("every", eval_every)

    use_tty = sys.stdout.isatty()
    
    if use_phase:
        epochs = phase1_epochs
        phase = 1
        pbar1 = trange(
            phase1_epochs,
            desc="Phase 1 Training",
            ncols=120,
            dynamic_ncols=True,
            leave=False,          # don't leave old bars behind
            disable=not use_tty,  # if output is piped, avoid multiline spam
        )
        pbar2 = trange(
            phase2_epochs,
            desc="Phase 2 Training",
            ncols=120,
            dynamic_ncols=True,
            leave=False,          # don't leave old bars behind
            disable=not use_tty,  # if output is piped, avoid multiline spam
        )
    else:
        phase = 0
        pbar1 = trange(
            epochs,
            desc="Training",
            ncols=120,
            dynamic_ncols=True,
            leave=False,          # don't leave old bars behind
            disable=not use_tty,  # if output is piped, avoid multiline spam
        )

    # Training loop
    print("training started")
    training_start_time = time.time()
    global_step = 0
    phase1_iter_times = []          # per-iteration wall-clock time in Phase 1 (seconds)
    phase1_mem_alloc_history = []   # per-iteration GPU memory allocated in Phase 1 (MB)
    for ep in pbar1:
        iter_start = time.time()
        model.train()
        batch = exp.sample_batch(n_f=n_f)

        loss_res = exp.pde_residual_loss(model, batch).mean() if batch.get("X_f") is not None else torch.tensor(0., device=device)
        loss_data = exp.data_loss(model, batch, phase).mean() if batch.get("X_d") is not None else torch.tensor(0., device=device)
        
        loss_res_s = loss_res.mean() if torch.is_tensor(loss_res) and loss_res.dim() > 0 else loss_res # scalar
        loss_data_s = loss_data.mean() if torch.is_tensor(loss_data) and loss_data.dim() > 0 else loss_data

        total_loss = w_res * loss_res + w_data * loss_data

        kl_loss = None
        if hasattr(model, "kl_loss"):
            kl_weight = float(getattr(model, "kl_weight", 1.0))
            kl_loss = model.kl_loss()
            total_loss = total_loss + kl_weight * kl_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        phase1_iter_times.append(time.time() - iter_start)
        if torch.cuda.is_available():
            phase1_mem_alloc_history.append(float(torch.cuda.memory_allocated(device)) / (1024**2))

        # Log
        it_per_sec = pbar1.format_dict.get("rate", None)
        elapsed_s  = pbar1.format_dict.get("elapsed", None)
        gpu_now = {
            "gpu/mem_alloc_mb": float(torch.cuda.memory_allocated(device)) / (1024**2),
            "gpu/mem_reserved_mb": float(torch.cuda.memory_reserved(device)) / (1024**2),
        }
        log_payload = {
            "loss/total": float(total_loss.detach().cpu()),
            "loss/res": float(loss_res_s.detach().cpu()),
            "loss/data": float(loss_data_s.detach().cpu()),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch": ep,
            "perf/it_per_sec_tqdm": it_per_sec if it_per_sec is not None else 0.0,
            "perf/elapsed_sec": elapsed_s if elapsed_s is not None else 0.0,
            **gpu_now,
        }
        if kl_loss is not None:
            log_payload["loss/kl"] = float(kl_loss.detach().cpu())
            log_payload["loss/kl_weight"] = kl_weight
        
        if hasattr(exp, "nu") and isinstance(exp.nu, torch.nn.Parameter): # Burgers
            log_payload["pde/nu"] = float(exp.nu.detach().cpu())
        if hasattr(exp, "eps") and isinstance(exp.eps, torch.nn.Parameter): # Allen-Cahn
            log_payload["pde/eps"] = float(exp.eps.detach().cpu())
        wandb_log(log_payload, commit=True)
        pbar1.set_postfix({k: f"{v:.3e}" for k,v in log_payload.items() if "loss" in k})
        global_step += 1

        # Simple validation metric (rMAE and rMSE on a fixed grid)
        if (ep % eval_every == 0 or ep == epochs-1) and (ep > 0):
            print("Evaluating...")
            model.eval()
            with torch.no_grad():
                eval_result = exp.eval_on_grid(model, base_cfg["eval"]["grid"])
                rMAE, rMSE = eval_result["rMAE"], eval_result["rMSE"]
            wandb_log({"eval/rMAE": rMAE, "eval/rMSE": rMSE, "epoch": ep})

            best_path = os.path.join(out_dir, "best.pt")
            if rMSE < best_metric:
                best_metric = rMSE
                
                best_model_state = state_to_cpu(model.state_dict())
                if hasattr(exp, "state_dict"):
                    best_exp_state = state_to_cpu(exp.state_dict())

                # Save checkpoint with structured dict
                save_dict = {
                    "model": best_model_state,
                }
                if best_exp_state is not None:
                    save_dict["experiment"] = best_exp_state
                
                torch.save(save_dict, best_path)

        if enable_video and (ep % make_video_every == 0 and ep > 0):
            print(f"Making video...")
            vid_grid = exp_cfg.get("video", {}).get("grid", base_cfg["eval"]["grid"])
            fps      = exp_cfg.get("video", {}).get("fps", 10)
            out_fmt  = exp_cfg.get("video", {}).get("format", "mp4")  # "mp4" or "gif"
            vid_path = exp.make_video(
                model, vid_grid, out_dir, fps=fps,
                filename=f"eval_ep{ep}.{out_fmt}",
                phase=phase
            )
            
    if enable_video:
        vid_grid = exp_cfg.get("video", {}).get("grid", base_cfg["eval"]["grid"])
        fps      = exp_cfg.get("video", {}).get("fps", 10)
        out_fmt  = exp_cfg.get("video", {}).get("format", "mp4")  # "mp4" or "gif"
        if use_phase:
            vid_filename = f"phase1_result.{out_fmt}"
        else:
            vid_filename = f"final_evolution.{out_fmt}"
        vid_path = exp.make_video(
            model, vid_grid, out_dir,
            fps=fps, filename=vid_filename,
            phase=phase
        )
        wandb_log({"video/evolution": wandb.Video(vid_path, format=out_fmt)})
        
        base, ext = os.path.splitext(os.path.basename(vid_path))
        noise_true = os.path.join(out_dir, f"{base}_noise_true{ext}")
        noise_ebm  = os.path.join(out_dir, f"{base}_noise_ebm{ext}")
        if os.path.exists(noise_true):
            wandb_log({"video/noise_true": wandb.Video(noise_true, format=out_fmt)})
        if os.path.exists(noise_ebm):
            wandb_log({"video/noise_ebm": wandb.Video(noise_ebm, format=out_fmt)})  
        
    if use_phase:
        exp.initialize_EBM(model)
        phase = 2

        print("[Optimizer] Resetting Adam state for Phase 2 fine-tuning.")
        phase2_lr = opt_cfg["lr"]
        params = list(model.parameters())
        if hasattr(exp, "extra_params"):
            params += list(exp.extra_params())

        optimizer = torch.optim.Adam(params, lr=phase2_lr, weight_decay=opt_cfg.get("weight_decay", 0.0))

        phase2_iter_times = []          # per-iteration wall-clock time in Phase 2 (seconds)
        phase2_mem_alloc_history = []   # per-iteration GPU memory allocated in Phase 2 (MB)
        for ep in pbar2:
            iter_start = time.time()
            model.train()
            batch = exp.sample_batch(n_f=n_f)

            loss_res = exp.pde_residual_loss(model, batch).mean() if batch.get("X_f") is not None else torch.tensor(0., device=device)
            loss_data = exp.data_loss(model, batch, phase).mean() if batch.get("X_d") is not None else torch.tensor(0., device=device)

            loss_res_s = loss_res.mean() if torch.is_tensor(loss_res) and loss_res.dim() > 0 else loss_res # scalar
            loss_data_s = loss_data.mean() if torch.is_tensor(loss_data) and loss_data.dim() > 0 else loss_data

            total_loss = w_res * loss_res + w_data * loss_data

            kl_loss = None
            if hasattr(model, "kl_loss"):
                kl_weight = float(getattr(model, "kl_weight", 1.0))
                kl_loss = model.kl_loss()
                total_loss = total_loss + kl_weight * kl_loss
            
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            phase2_iter_times.append(time.time() - iter_start)
            if torch.cuda.is_available():
                phase2_mem_alloc_history.append(float(torch.cuda.memory_allocated(device)) / (1024**2))

            # Log
            it_per_sec = pbar2.format_dict.get("rate", None)
            elapsed_s  = pbar2.format_dict.get("elapsed", None)
            gpu_now = {
                "gpu/mem_alloc_mb": float(torch.cuda.memory_allocated(device)) / (1024**2),
                "gpu/mem_reserved_mb": float(torch.cuda.memory_reserved(device)) / (1024**2),
            }
            log_payload = {
                "loss/total": float(total_loss.detach().cpu()),
                "loss/res": float(loss_res_s.detach().cpu()),
                "loss/data": float(loss_data_s.detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": ep + phase1_epochs,
                "perf/it_per_sec_tqdm": it_per_sec if it_per_sec is not None else 0.0,
                "perf/elapsed_sec": elapsed_s if elapsed_s is not None else 0.0,
                **gpu_now,
            }
            if kl_loss is not None:
                log_payload["loss/kl"] = float(kl_loss.detach().cpu())
                log_payload["loss/kl_weight"] = kl_weight
            if hasattr(exp, "running_std"):
                log_payload["running_std"] = float(exp.running_std.detach().cpu())
            if hasattr(exp, "nu") and isinstance(exp.nu, torch.nn.Parameter):
                log_payload["pde/nu"] = float(exp.nu.detach().cpu())
            if hasattr(exp, "eps") and isinstance(exp.eps, torch.nn.Parameter):
                log_payload["pde/eps"] = float(exp.eps.detach().cpu())
            if hasattr(exp, "gate_module") and exp.gate_module is not None:
                log_payload["gate/cutoff"] = float(exp.gate_module.cutoff_alpha.detach().cpu())
                log_payload["gate/steepness"]  = float(exp.gate_module.steepness.detach().cpu())
            if hasattr(exp, "threshold_gate") and exp.threshold_gate is not None:
                log_payload["gate/threshold"] = float(exp.threshold_gate.raw_threshold.detach().cpu())
                log_payload["gate/steepness"] = float(exp.threshold_gate.raw_steepness.detach().cpu())
            if hasattr(exp, "_last_n_filtered") and hasattr(exp, "_last_n_total"):
                log_payload["gate/n_filtered"] = exp._last_n_filtered
                log_payload["gate/pct_filtered"] = 100.0 * exp._last_n_filtered / max(exp._last_n_total, 1)
            wandb_log(log_payload, commit=True)
            pbar2.set_postfix({k: f"{v:.3e}" for k,v in log_payload.items() if "loss" in k})
            global_step += 1
            
            # Simple validation metric (rMAE and rMSE on a fixed grid)
            if (ep % eval_every == 0 or ep == phase2_epochs-1):
                model.eval()
                with torch.no_grad():
                    eval_result = exp.eval_on_grid(model, base_cfg["eval"]["grid"])
                    rMAE, rMSE = eval_result["rMAE"], eval_result["rMSE"]
                wandb_log({"eval/rMAE": rMAE, "eval/rMSE": rMSE, "epoch": ep + phase1_epochs})
                
                best_path = os.path.join(out_dir, "best.pt")
                if rMSE < best_metric:
                    best_metric = rMSE
                    
                    best_model_state = state_to_cpu(model.state_dict())
                    if hasattr(exp, "state_dict"):
                        best_exp_state = state_to_cpu(exp.state_dict())
                    
                    save_dict = {"model": best_model_state}
                    if best_exp_state is not None:
                        save_dict["experiment"] = best_exp_state
                    torch.save(save_dict, best_path)
                
            if enable_video and ((ep + phase1_epochs) % make_video_every == 0 and ep > 0):
                vid_grid = exp_cfg.get("video", {}).get("grid", base_cfg["eval"]["grid"])
                fps      = exp_cfg.get("video", {}).get("fps", 10)
                out_fmt  = exp_cfg.get("video", {}).get("format", "mp4")  # "mp4" or "gif"
                vid_path = exp.make_video(
                    model, vid_grid, out_dir, fps=fps,
                    filename=f"eval_ep{ep + phase1_epochs}.{out_fmt}",
                    phase=phase
                )
                if hasattr(exp, "evaluate_gate_performance"):
                    exp.evaluate_gate_performance(model, out_dir, filename_prefix=f"eval_ep{ep + phase1_epochs}")
                
        final_path = os.path.join(out_dir, "final.pt")
        final_model_state = state_to_cpu(model.state_dict())
        if hasattr(exp, "state_dict"):
            final_exp_state = state_to_cpu  (exp.state_dict())
        save_dict = {"model": final_model_state}
        if final_exp_state is not None:
            save_dict["experiment"] = final_exp_state
        torch.save(save_dict, final_path)
        
        if enable_video:
            vid_grid = exp_cfg.get("video", {}).get("grid", base_cfg["eval"]["grid"])
            fps      = exp_cfg.get("video", {}).get("fps", 10)
            out_fmt  = exp_cfg.get("video", {}).get("format", "mp4")  # "mp4" or "gif"
            vid_path = exp.make_video(
                model, vid_grid, out_dir,
                fps=fps, filename=f"final_evolution.{out_fmt}",
                phase=phase
            )
            wandb_log({"video/evolution": wandb.Video(vid_path, format=out_fmt)})
            
            base, ext = os.path.splitext(os.path.basename(vid_path))
            noise_true = os.path.join(out_dir, f"{base}_noise_true{ext}")
            noise_ebm  = os.path.join(out_dir, f"{base}_noise_ebm{ext}")
            if os.path.exists(noise_true):
                wandb_log({"video/noise_true": wandb.Video(noise_true, format=out_fmt)})
            if os.path.exists(noise_ebm):
                wandb_log({"video/noise_ebm": wandb.Video(noise_ebm, format=out_fmt)}) 
                
        if hasattr(exp, "evaluate_gate_performance"):
            exp.evaluate_gate_performance(model, out_dir, filename_prefix="final")
    
    model.eval()
    with torch.no_grad():
        eval_result = exp.eval_on_grid(model, base_cfg["eval"]["grid"])
        rMAE, rMSE = eval_result["rMAE"], eval_result["rMSE"]
    wandb_log({"eval/final_rMAE": rMAE, "eval/final_rMSE": rMSE})

    wandb_log({"eval/best_rMSE": best_metric})
    training_end_time = time.time()
    
    all_iter_times = phase1_iter_times + (phase2_iter_times if use_phase else [])
    final_perf = {
        "perf/total_time_sec": training_end_time - training_start_time,
        "perf/avg_sec_per_iter": sum(all_iter_times) / len(all_iter_times) if all_iter_times else 0.0,
        "gpu/peak_mem_alloc_mb": float(torch.cuda.max_memory_allocated(device)) / (1024**2),
        "gpu/peak_mem_reserved_mb": float(torch.cuda.max_memory_reserved(device)) / (1024**2),
        # Phase 1
        "perf/phase1/total_iter_time_sec": sum(phase1_iter_times),
        "perf/phase1/avg_sec_per_iter": sum(phase1_iter_times) / len(phase1_iter_times) if phase1_iter_times else 0.0,
    }
    if phase1_mem_alloc_history:
        final_perf["gpu/phase1/min_mem_alloc_mb"] = min(phase1_mem_alloc_history)
        final_perf["gpu/phase1/avg_mem_alloc_mb"] = sum(phase1_mem_alloc_history) / len(phase1_mem_alloc_history)
        final_perf["gpu/phase1/max_mem_alloc_mb"] = max(phase1_mem_alloc_history)
    # Phase 2 (only when phased training is enabled)
    if use_phase:
        final_perf["perf/phase2/total_iter_time_sec"] = sum(phase2_iter_times)
        final_perf["perf/phase2/avg_sec_per_iter"] = sum(phase2_iter_times) / len(phase2_iter_times) if phase2_iter_times else 0.0
        if phase2_mem_alloc_history:
            final_perf["gpu/phase2/min_mem_alloc_mb"] = min(phase2_mem_alloc_history)
            final_perf["gpu/phase2/avg_mem_alloc_mb"] = sum(phase2_mem_alloc_history) / len(phase2_mem_alloc_history)
            final_perf["gpu/phase2/max_mem_alloc_mb"] = max(phase2_mem_alloc_history)

    wandb_log(final_perf)
    _save_json(
        os.path.join(out_dir, "metrics.json"),
        {
            "status": "complete",
            "experiment_name": args.experiment_name,
            "model_name": args.model_name,
            "tag": tag,
            "seed": int(base_cfg["seed"]),
            "device": str(device),
            "pinn_update_steps": int(global_step),
            "field_rMAE": float(rMAE),
            "field_rMSE": float(rMSE),
            "best_field_rMSE": (
                None if not math.isfinite(best_metric) else float(best_metric)
            ),
            **final_perf,
        },
    )

    # Restore best
    if best_model_state:
        print("[Restore] Loading best model state...")
        model.load_state_dict(best_model_state)
        if best_exp_state and hasattr(exp, "load_state_dict"):
            print("[Restore] Loading best experiment state...")
            exp.load_state_dict(best_exp_state)

    model.eval()
    wandb_finish()
    print(f"Artifacts saved to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--common_config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--exp_config", required=True)
    args = parser.parse_args()
    main(args)
