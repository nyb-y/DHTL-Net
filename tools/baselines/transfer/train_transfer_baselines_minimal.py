#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import tradition


def main():
    args = sys.argv[1:]

    forbidden_flags = {
        "--use_weighted_sampler": "WeightedRandomSampler",
        "--use_class_weights": "class weights",
    }

    for flag, name in forbidden_flags.items():
        if flag in args:
            raise SystemExit(f"tradition_minimal.py disables {name}; remove {flag}.")

    if "--train_aug" not in args:
        args.extend(["--train_aug", "none"])

    if "--scheduler" not in args:
        args.extend(["--scheduler", "none"])

    sys.argv = [sys.argv[0]] + args
    tradition.main()


if __name__ == "__main__":
    main()
