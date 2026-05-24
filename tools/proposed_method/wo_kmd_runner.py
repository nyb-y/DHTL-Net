import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from wo_kmd_common import set_seed, parse_int_list
from wo_kmd_data import make_loaders, make_class_weights
from wo_kmd_model import AdaptiveInterpDynamic
from wo_kmd_training import load_source_weights, set_trainable_params, count_trainable_params, evaluate, train_one_epoch
from wo_kmd_artifacts import try_compute_flops_and_params, profile_batchsize_latency_vram, save_qualitative_probability_examples
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root directory with train/val/test subfolders.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Source-task pretrained checkpoint path.")
    parser.add_argument("--out_dir", type=str, default="runs_binary_iqa/dynamic_alpha")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alpha_lr_mult", type=float, default=0.5)
    parser.add_argument("--lambda_prior", type=float, default=0.1)
    parser.add_argument("--lambda_smooth", type=float, default=0.05)
    parser.add_argument("--lambda_ent", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ft_mode", type=str, default="l34", choices=["l4", "l34", "l234", "all"])
    parser.add_argument("--use_class_weights", action="store_true", help="Use inverse-frequency class weights in CE.")
    parser.add_argument("--use_weighted_sampler", action="store_true", help="Use weighted sampler for training.")

    # Output artifacts for plotting and qualitative analysis.
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--profile_batch_sizes", type=str, default="8,16,32,64",
                        help="Comma-separated batch sizes for latency/VRAM profiling. Empty disables profiling.")
    parser.add_argument("--profile_warmup", type=int, default=10)
    parser.add_argument("--profile_iters", type=int, default=30)
    parser.add_argument("--flops_train_multiplier", type=float, default=3.0,
                        help="Approximate training FLOPs multiplier relative to forward FLOPs.")
    parser.add_argument("--save_qualitative", action="store_true",
                        help="Save two usable and two unusable test images with probability annotations.")
    parser.add_argument("--qualitative_size", type=int, default=512,
                        help="Image size for qualitative examples. Default: 512.")
    parser.add_argument("--qualitative_dpi", type=int, default=600,
                        help="DPI metadata for qualitative JPG examples. Default: 600.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    dl_tr, dl_va, dl_te, label2id, classes, train_counts = make_loaders(
        args.data_root, args.batch_size, args.num_workers, args.seed, args.use_weighted_sampler
    )
    num_classes = 2
    print("Classes:", classes)

    channels = [256, 512, 1024, 2048]
    alpha_priors = [0.85, 0.75, 0.55, 0.30]

    model = AdaptiveInterpDynamic(num_classes=num_classes, alpha_priors=alpha_priors, channels=channels)
    load_source_weights(model, args.ckpt_path)
    model.to(device)

    class_weights = make_class_weights(train_counts, device) if args.use_class_weights else None
    if class_weights is not None:
        print("Class weights [unusable, usable]:", class_weights.detach().cpu().numpy().tolist())

    # FLOPs for plotting FLOPs vs loss.
    flops_per_sample, profiled_params = try_compute_flops_and_params(model, device, input_size=args.input_size)
    flops_info = {
        "forward_flops_per_sample": flops_per_sample,
        "profiled_params": profiled_params,
        "input_size": args.input_size,
        "flops_train_multiplier": args.flops_train_multiplier,
        "note": "Forward FLOPs are estimated with thop if available; training FLOPs are approximated by flops_train_multiplier."
    }
    with open(os.path.join(args.out_dir, "flops_summary.json"), "w", encoding="utf-8") as f:
        json.dump(flops_info, f, indent=2)
    print(f"[FLOPs] forward/sample={flops_per_sample/1e9 if np.isfinite(flops_per_sample) else float('nan'):.4f} GFLOPs")

    optimizer = torch.optim.AdamW([
        {"params": model.tgt_bb.parameters(), "lr": args.lr},
        {"params": model.fc.parameters(), "lr": args.lr},
        {"params": model.alpha_generators.parameters(), "lr": args.lr * args.alpha_lr_mult},
    ], weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_path = os.path.join(args.out_dir, "best_model.pth")
    history = []

    for epoch in range(1, args.epochs + 1):
        warmup = epoch <= args.warmup_epochs
        set_trainable_params(model, warmup=warmup, ft_mode=args.ft_mode)
        trainable, total = count_trainable_params(model)
        print(f"\nEpoch {epoch}/{args.epochs} | warmup={warmup} | trainable={trainable/1e6:.2f}M/{total/1e6:.2f}M")

        loss = train_one_epoch(
            model, dl_tr, optimizer, device, alpha_priors, class_weights,
            args.lambda_prior, args.lambda_smooth, args.lambda_ent
        )
        val_metrics = evaluate(
            model, dl_va, device, num_classes,
            alpha_priors, class_weights,
            args.lambda_prior, args.lambda_smooth, args.lambda_ent
        )

        print(
            f"Loss={loss:.4f} | "
            f"Val Loss={val_metrics['loss']:.4f} | "
            f"Val F1={val_metrics['macro_f1']:.4f} | "
            f"Val BAcc={val_metrics['balanced_acc']:.4f} | "
            f"Val Acc={val_metrics['accuracy']:.4f} | "
            f"Val AUC={val_metrics['auc']:.4f}"
        )

        train_forward_flops_epoch = flops_per_sample * len(dl_tr.dataset) if np.isfinite(flops_per_sample) else float("nan")
        train_estimated_flops_epoch = train_forward_flops_epoch * args.flops_train_multiplier if np.isfinite(train_forward_flops_epoch) else float("nan")
        row = {
            "epoch": epoch,
            "loss": loss,
            "train_forward_flops_epoch": train_forward_flops_epoch,
            "train_estimated_flops_epoch": train_estimated_flops_epoch,
            "cumulative_train_estimated_flops": train_estimated_flops_epoch * epoch if np.isfinite(train_estimated_flops_epoch) else float("nan"),
            "cumulative_train_estimated_tflops": train_estimated_flops_epoch * epoch / 1e12 if np.isfinite(train_estimated_flops_epoch) else float("nan"),
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "cm"},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "history.csv"), index=False)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "label2id": label2id,
                "classes": classes,
                "args": vars(args),
                "best_val_macro_f1": best_f1,
            }, best_path)
            print(f"  -> Saved best model (Val F1={best_f1:.4f})")

    # Save history
    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Test
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    test_metrics = evaluate(
        model, dl_te, device, num_classes,
        alpha_priors, class_weights,
        args.lambda_prior, args.lambda_smooth, args.lambda_ent
    )

    print("\n===== TEST =====")
    print(
        f"Test Loss={test_metrics['loss']:.4f} | "
        f"Test F1={test_metrics['macro_f1']:.4f} | "
        f"BAcc={test_metrics['balanced_acc']:.4f} | "
        f"Acc={test_metrics['accuracy']:.4f} | "
        f"AUC={test_metrics['auc']:.4f} | "
        f"Sens={test_metrics['sensitivity']:.4f} | "
        f"Spec={test_metrics['specificity']:.4f}"
    )
    print("Confusion Matrix [rows=true, cols=pred; class order: unusable, usable]:\n", test_metrics["cm"])

    # Figure data 1 and 2:
    #   - batchsize vs time
    #   - VRAM (MB) vs latency
    profile_rows = []
    batch_sizes_for_profile = parse_int_list(args.profile_batch_sizes)
    if len(batch_sizes_for_profile) > 0:
        profile_rows = profile_batchsize_latency_vram(
            model=model,
            device=device,
            batch_sizes=batch_sizes_for_profile,
            input_size=args.input_size,
            warmup_iters=args.profile_warmup,
            profile_iters=args.profile_iters,
        )
        pd.DataFrame(profile_rows).to_csv(
            os.path.join(args.out_dir, "profile_batchsize_latency_vram.csv"), index=False
        )

    # Qualitative examples: two usable and two unusable images with probability annotations.
    qualitative_paths = []
    if args.save_qualitative:
        qualitative_paths = save_qualitative_probability_examples(
            model=model,
            data_root=args.data_root,
            out_dir=args.out_dir,
            device=device,
            n_per_class=2,
            input_size=args.qualitative_size,
            dpi=args.qualitative_dpi,
        )

    results = {
        "method": "DHTL-Net w/o KMD",
        "ablation": "All MatmulFreeDense layers are replaced with nn.Linear; DyT is replaced with nn.LayerNorm; FourierKAN is replaced with nn.GELU. Other settings are unchanged.",
        "classes": classes,
        "label2id": label2id,
        "test_loss": test_metrics["loss"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_balanced_acc": test_metrics["balanced_acc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "test_sensitivity_usable": test_metrics["sensitivity"],
        "test_specificity_unusable": test_metrics["specificity"],
        "test_confusion_matrix": test_metrics["cm"].tolist(),
        "flops_summary": flops_info,
        "profile_csv": os.path.join(args.out_dir, "profile_batchsize_latency_vram.csv") if profile_rows else None,
        "qualitative_examples": qualitative_paths,
        "args": vars(args),
    }

    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(args.out_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write("Method: DHTL-Net w/o KMD\n")
        f.write("Ablation: All MatmulFreeDense layers are replaced with nn.Linear; DyT is replaced with nn.LayerNorm; FourierKAN is replaced with nn.GELU. Other settings are unchanged.\n")
        f.write("Classes: " + str(classes) + "\n")
        f.write(f"Test Loss: {test_metrics['loss']:.6f}\n")
        f.write(f"Test Macro-F1: {test_metrics['macro_f1']:.6f}\n")
        f.write(f"Test Balanced Acc: {test_metrics['balanced_acc']:.6f}\n")
        f.write(f"Test Accuracy: {test_metrics['accuracy']:.6f}\n")
        f.write(f"Test AUC: {test_metrics['auc']:.6f}\n")
        f.write(f"Sensitivity usable: {test_metrics['sensitivity']:.6f}\n")
        f.write(f"Specificity unusable: {test_metrics['specificity']:.6f}\n")
        f.write("Confusion Matrix [rows=true, cols=pred; order: unusable, usable]:\n")
        f.write(np.array2string(test_metrics["cm"]))
        f.write("\n")
        f.write("\nArtifacts:\n")
        f.write("history_csv: " + os.path.join(args.out_dir, "history.csv") + "\n")
        f.write("flops_summary: " + os.path.join(args.out_dir, "flops_summary.json") + "\n")
        if profile_rows:
            f.write("profile_csv: " + os.path.join(args.out_dir, "profile_batchsize_latency_vram.csv") + "\n")
        if qualitative_paths:
            f.write("qualitative_examples_dir: " + os.path.join(args.out_dir, "qualitative_examples") + "\n")


if __name__ == "__main__":
    main()
