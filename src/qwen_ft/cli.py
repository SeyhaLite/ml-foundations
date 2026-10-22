"""``qwen-ft`` command line interface.

Subcommands
-----------

``qwen-ft env``
    Print installed versions of the ML stack (torch/transformers/peft/trl/…)
    and whether CUDA is visible.  No downloads, no GPU required.

``qwen-ft validate CONFIG.yaml``
    Validate a training config (+ its dataset) without touching any model.

``qwen-ft validate-data DATASET.jsonl``
    Validate a chat-messages JSONL dataset.

``qwen-ft models [--family qwen3.5]``
    List the verified model registry.

``qwen-ft train CONFIG.yaml``
    Run LoRA SFT training (requires ``pip install -e '.[train]'`` + GPU).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from qwen_ft import __version__
from qwen_ft.config import ConfigError, load_config
from qwen_ft.data import DataValidationError, validate_jsonl_file
from qwen_ft.models import list_models


def _cmd_env(_args: argparse.Namespace) -> int:
    from qwen_ft.training import check_environment

    env = check_environment()
    print(f"qwen-ft {__version__}")
    for pkg, ver in env.items():
        print(f"  {pkg:14s} {ver if ver is not None else 'NOT INSTALLED'}")
    missing = [k for k in ("torch", "transformers", "peft", "trl") if env.get(k) is None]
    if missing:
        print("\nTraining stack incomplete. Install with:  pip install -e '.[train]'")
        return 0  # informational, not an error
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"OK  {args.config}")
    print(f"    model      : {cfg.model.model_id} (dtype={cfg.model.dtype}, max_length={cfg.model.max_length})")
    print(f"    lora       : r={cfg.lora.r} alpha={cfg.lora.alpha} dropout={cfg.lora.dropout} targets={cfg.lora.target_modules}")
    print(f"    training   : epochs={cfg.training.num_epochs} bs={cfg.training.batch_size}x{cfg.training.grad_accum_steps} "
          f"lr={cfg.training.learning_rate:g} out={cfg.training.output_dir}")

    warnings = cfg.warnings()
    if Path(cfg.training.dataset_path).is_file():
        try:
            report = validate_jsonl_file(cfg.training.dataset_path)
            print(f"    dataset    : {report['records']} records OK "
                  f"(turns {report['min_turns']}–{report['max_turns']}, avg {report['avg_chars']} chars)")
            warnings += report.get("warnings", [])
        except DataValidationError as exc:
            print(f"DATASET ERROR: {exc}", file=sys.stderr)
            return 1
    else:
        warnings.append(f"dataset file not found: {cfg.training.dataset_path}")

    for w in warnings:
        print(f"    [warn] {w}")
    return 0


def _cmd_validate_data(args: argparse.Namespace) -> int:
    try:
        report = validate_jsonl_file(args.dataset)
    except DataValidationError as exc:
        print(f"DATASET ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    for m in list_models(family=args.family):
        vram = f"~{m.lora_bf16_vram_gb:g}GB LoRA" if m.lora_bf16_vram_gb else "LoRA VRAM n/a"
        print(f"{m.model_id:32s} {m.family:8s} {m.kind:5s} ctx={m.context_tokens:>7,} "
              f"transformers>={m.min_transformers:5s} {vram}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    try:
        from qwen_ft.training import HeavyDependencyError, train_from_yaml
    except ImportError as exc:  # pragma: no cover
        print(f"IMPORT ERROR: {exc}", file=sys.stderr)
        return 1
    try:
        out = train_from_yaml(args.config, resume_from=args.resume_from)
    except HeavyDependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (ConfigError, DataValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Adapter saved to: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qwen-ft", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"qwen-ft {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("env", help="print installed ML stack versions").set_defaults(func=_cmd_env)

    pv = sub.add_parser("validate", help="validate a YAML training config (+dataset)")
    pv.add_argument("config")
    pv.set_defaults(func=_cmd_validate)

    pd = sub.add_parser("validate-data", help="validate a messages JSONL dataset")
    pd.add_argument("dataset")
    pd.set_defaults(func=_cmd_validate_data)

    pm = sub.add_parser("models", help="list the verified Qwen model registry")
    pm.add_argument("--family", default=None,
                    help="filter: qwen3 | qwen3.5 | qwen3.6 | qwen3.8")
    pm.set_defaults(func=_cmd_models)

    pt = sub.add_parser("train", help="run LoRA SFT training (needs the 'train' extra)")
    pt.add_argument("config")
    pt.add_argument("--resume-from", default=None)
    pt.set_defaults(func=_cmd_train)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
