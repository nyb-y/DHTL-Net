import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from common import set_seed, parse_int_list
from data import make_loaders, make_class_weights
from model import load_encoder_from_ckpt, set_trainable_layers
from training import count_trainable_params, evaluate, train_one_epoch
from artifacts import try_compute_flops_and_params, profile_batchsize_latency_vram, save_qualitative_probability_examples
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["frozen", "all", "l4", "l34", "l234"],
    )

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--train_aug", type=str, default="default", choices=["default", "none"])
    parser.add_argument("--extra_train_root", type=str, default=None)
    parser.add_argument("--extra_data_root", type=str, default=None)
    parser.add_argument("--extra_split_train", type=float, default=0.7)
    parser.add_argument("--extra_split_val", type=float, default=0.15)
    parser.add_argument("--extra_split_test", type=float, default=0.15)
    parser.add_argument("--test_root_override", type=str, default=None)

    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--profile_batch_sizes", type=str, default="8,16,32,64")
    parser.add_argument("--profile_warmup", type=int, default=10)
    parser.add_argument("--profile_iters", type=int, default=30)
    parser.add_argument("--flops_train_multiplier", type=float, default=3.0)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])

    parser.add_argument("--save_qualitative", action="store_true")
    parser.add_argument("--qualitative_size", type=int, default=512)
    parser.add_argument("--qualitative_dpi", type=int, default=600)

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dl_tr, dl_va, dl_te, train_counts = make_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_weighted_sampler=args.use_weighted_sampler,
        train_aug=args.train_aug,
        extra_train_root=args.extra_train_root,
        extra_data_root=args.extra_data_root,
        extra_split_train=args.extra_split_train,
        extra_split_val=args.extra_split_val,
        extra_split_test=args.extra_split_test,
        test_root_override=args.test_root_override,
    )

    num_classes = 2
    classes = ["unusable", "usable"]
    label2id = {"unusable": 0, "usable": 1}

    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    load_encoder_from_ckpt(model, args.ckpt_path)
    model.to(device)

    class_weights = make_class_weights(train_counts, device) if args.use_class_weights else None

    if class_weights is not None:
        print("Class weights [unusable, usable]:", class_weights.detach().cpu().numpy().tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )

    flops_per_sample, profiled_params = try_compute_flops_and_params(
        model=model,
        device=device,
        input_size=args.input_size,
    )

    flops_info = {
        "forward_flops_per_sample": flops_per_sample,
        "profiled_params": profiled_params,
        "input_size": args.input_size,
        "flops_train_multiplier": args.flops_train_multiplier,
        "note": "Forward FLOPs are estimated with thop if available; training FLOPs are approximated by flops_train_multiplier."
    }

    with open(os.path.join(args.out_dir, "flops_summary.json"), "w", encoding="utf-8") as f:
        json.dump(flops_info, f, indent=2)

    print(f"[FLOPs] forward/sample={flops_per_sample / 1e9 if np.isfinite(flops_per_sample) else float('nan'):.4f} GFLOPs")

    with open(os.path.join(args.out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    best_f1 = -1.0
    best_path = os.path.join(args.out_dir, "best_model.pth")
    history = []

    for epoch in range(1, args.epochs + 1):
        warmup = epoch <= args.warmup_epochs

        set_trainable_layers(
            model=model,
            mode=args.mode,
            warmup=warmup,
        )

        trainable, total = count_trainable_params(model)

        print(
            f"\nEpoch {epoch}/{args.epochs} | "
            f"mode={args.mode} | "
            f"warmup={warmup} | "
            f"trainable={trainable / 1e6:.2f}M/{total / 1e6:.2f}M"
        )

        loss = train_one_epoch(
            model=model,
            loader=dl_tr,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        val_metrics = evaluate(
            model=model,
            loader=dl_va,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        print(
            f"Loss={loss:.4f} | "
            f"Val Loss={val_metrics['loss']:.4f} | "
            f"Val F1={val_metrics['macro_f1']:.4f} | "
            f"Val BAcc={val_metrics['balanced_acc']:.4f} | "
            f"Val Acc={val_metrics['accuracy']:.4f} | "
            f"Val AUC={val_metrics['auc']:.4f}"
        )

        train_forward_flops_epoch = (
            flops_per_sample * len(dl_tr.dataset)
            if np.isfinite(flops_per_sample)
            else float("nan")
        )

        train_estimated_flops_epoch = (
            train_forward_flops_epoch * args.flops_train_multiplier
            if np.isfinite(train_forward_flops_epoch)
            else float("nan")
        )

        row = {
            "epoch": epoch,
            "mode": args.mode,
            "warmup": warmup,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_ratio": trainable / total,
            "loss": loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_forward_flops_epoch": train_forward_flops_epoch,
            "train_estimated_flops_epoch": train_estimated_flops_epoch,
            "cumulative_train_estimated_flops": (
                train_estimated_flops_epoch * epoch
                if np.isfinite(train_estimated_flops_epoch)
                else float("nan")
            ),
            "cumulative_train_estimated_tflops": (
                train_estimated_flops_epoch * epoch / 1e12
                if np.isfinite(train_estimated_flops_epoch)
                else float("nan")
            ),
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "cm"},
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            os.path.join(args.out_dir, "history.csv"),
            index=False,
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "classes": classes,
                    "mode": args.mode,
                    "trainable_params": trainable,
                    "total_params": total,
                    "best_val_macro_f1": best_f1,
                    "args": vars(args),
                },
                best_path,
            )

            print(f"  -> Saved best model (Val F1={best_f1:.4f})")

    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    test_metrics = evaluate(
        model=model,
        loader=dl_te,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
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

    print(
        "Confusion Matrix [rows=true, cols=pred; class order: unusable, usable]:\n",
        test_metrics["cm"],
    )

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
            os.path.join(args.out_dir, "profile_batchsize_latency_vram.csv"),
            index=False,
        )

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
        "method": f"transfer_{args.mode}",
        "classes": classes,
        "label2id": label2id,
        "best_epoch": ckpt.get("epoch", None),
        "best_val_macro_f1": ckpt.get("best_val_macro_f1", best_f1),
        "trainable_params_at_best": ckpt.get("trainable_params", None),
        "total_params": ckpt.get("total_params", None),
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
        f.write(f"Method: transfer_{args.mode}\n")
        f.write(f"Classes: {classes}\n")
        f.write(f"Best epoch: {results['best_epoch']}\n")
        f.write(f"Best Val Macro-F1: {results['best_val_macro_f1']:.6f}\n")
        f.write(f"Trainable params at best: {results['trainable_params_at_best']}\n")
        f.write(f"Total params: {results['total_params']}\n")
        f.write(f"Test Loss: {test_metrics['loss']:.6f}\n")
        f.write(f"Test Macro-F1: {test_metrics['macro_f1']:.6f}\n")
        f.write(f"Test Balanced Acc: {test_metrics['balanced_acc']:.6f}\n")
        f.write(f"Test Accuracy: {test_metrics['accuracy']:.6f}\n")
        f.write(f"Test AUC: {test_metrics['auc']:.6f}\n")
        f.write(f"Sensitivity usable: {test_metrics['sensitivity']:.6f}\n")
        f.write(f"Specificity unusable: {test_metrics['specificity']:.6f}\n")
        f.write("Confusion Matrix [rows=true, cols=pred; order: unusable, usable]:\n")
        f.write(np.array2string(test_metrics["cm"]))
        f.write("\n\nArtifacts:\n")
        f.write("history_csv: " + os.path.join(args.out_dir, "history.csv") + "\n")
        f.write("history_json: " + os.path.join(args.out_dir, "history.json") + "\n")
        f.write("flops_summary: " + os.path.join(args.out_dir, "flops_summary.json") + "\n")
        if profile_rows:
            f.write("profile_csv: " + os.path.join(args.out_dir, "profile_batchsize_latency_vram.csv") + "\n")
        if qualitative_paths:
            f.write("qualitative_examples_dir: " + os.path.join(args.out_dir, "qualitative_examples") + "\n")


if __name__ == "__main__":
    main()
