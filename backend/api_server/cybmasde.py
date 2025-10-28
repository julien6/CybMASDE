#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CybMASDE CLI — Orchestrates the MAMAD methodology for multi-agent system design.
Author: Julien Soulé
"""
import argparse
import sys
import json
import os
import logging

from pathlib import Path
from world_model.project import Project

VERSION = "1.0.0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Global Argument Parser
# ============================================================
def build_parser():
    parser = argparse.ArgumentParser(
        prog="cybmasde",
        description="CybMASDE – Modular CLI for MAMAD methodology and MOISE+MARL systems.",
        epilog="Documentation: https://github.com/julien6/CybMASDE",
    )
    parser.add_argument("-v", "--version", action="version",
                        version=f"CybMASDE {VERSION}")
    parser.add_argument("-p", "--project", type=str,
                        help="Path to the CybMASDE project folder", default=os.getcwd())
    parser.add_argument("-c", "--config", type=str,
                        help="Alternative configuration file", default="project_configuration.json")

    subparsers = parser.add_subparsers(dest="command", title="Subcommands")

    # ========== INIT ==========
    p_init = subparsers.add_parser(
        "init", help="Create a new CybMASDE project.")
    p_init.add_argument("-n", "--name", required=True, help="Project name")
    p_init.add_argument("-d", "--description",
                        help="Project description", default="")
    p_init.add_argument(
        "-o", "--output", help="Output directory", default=None)
    p_init.add_argument(
        "--template", choices=["handcrafted", "worldmodel", "minimal"], default="handcrafted")

    # ========== VALIDATE ==========
    p_val = subparsers.add_parser(
        "validate", help="Validate project configuration.")
    p_val.add_argument("-q", "--quiet", action="store_true")
    p_val.add_argument("--strict", action="store_true")

    # ========== RUN ==========
    p_run = subparsers.add_parser(
        "run", help="Execute full or partial pipeline.")
    group_mode = p_run.add_mutually_exclusive_group()
    group_mode.add_argument("--full-auto", action="store_true")
    group_mode.add_argument("--semi-auto", action="store_true")
    group_mode.add_argument("--manual", action="store_true")
    p_run.add_argument("--skip-model", action="store_true")
    p_run.add_argument("--skip-analyze", action="store_true")
    p_run.add_argument("--max-refine", type=int, default=3)
    p_run.add_argument("--reward-threshold", type=float, default=None)
    p_run.add_argument("--std-threshold", type=float, default=None)
    p_run.add_argument("--accept-inferred", action="store_true")
    p_run.add_argument("--interactive-infer",
                       action="store_true", default=True)

    # ========== MODEL ==========
    p_mod = subparsers.add_parser("model", help="Run the modeling phase.")
    group_mod = p_mod.add_mutually_exclusive_group(required=True)
    group_mod.add_argument("--auto", action="store_true")
    group_mod.add_argument("--manual", action="store_true")
    p_mod.add_argument("--traces", type=str, help="Path to preexisting traces")
    p_mod.add_argument("--vae-dim", type=int, default=32)
    p_mod.add_argument("--lstm-hidden", type=int, default=64)

    # ========== TRAIN ==========
    p_train = subparsers.add_parser("train", help="Run the training phase.")
    p_train.add_argument(
        "--algo", choices=["MAPPO", "MADDPG", "QMIX", "IQL", "VDN", "ROMA"], default="MAPPO")
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--gamma", type=float, default=0.99)
    p_train.add_argument("--clip", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=None)
    p_train.add_argument("--epochs", type=int, default=100)

    # ========== ANALYZE ==========
    p_ana = subparsers.add_parser(
        "analyze", help="Run the analyzing phase (Auto-TEMM).")
    p_ana.add_argument("--auto-temm", action="store_true")
    p_ana.add_argument("--metrics", nargs="+",
                       choices=["reward", "stability", "org_fit"])
    p_ana.add_argument("--representativity", type=float, default=0.8)

    # ========== REFINE ==========
    p_ref = subparsers.add_parser("refine", help="Run refinement cycles.")
    p_ref.add_argument("--max", type=int, default=3)
    p_ref.add_argument("--accept-inferred", action="store_true")
    p_ref.add_argument("--interactive", action="store_true", default=True)

    # ========== DEPLOY ==========
    p_dep = subparsers.add_parser("deploy", help="Deploy learned policy.")
    group_dep = p_dep.add_mutually_exclusive_group(required=True)
    group_dep.add_argument("--direct", action="store_true")
    group_dep.add_argument("--remote", action="store_true")
    p_dep.add_argument("--checkpoint", type=str)
    p_dep.add_argument("--api", type=str, help="Target environment API URL")

    # ========== STATUS ==========
    subparsers.add_parser("status", help="Display project status.")

    # ========== CLEAN ==========
    p_clean = subparsers.add_parser("clean", help="Clean temporary files.")
    p_clean.add_argument("--traces", action="store_true")
    p_clean.add_argument("--checkpoints", action="store_true")
    p_clean.add_argument("--all", action="store_true")

    # ========== EXPORT ==========
    p_exp = subparsers.add_parser("export", help="Export results and metrics.")
    p_exp.add_argument(
        "--format", choices=["json", "csv", "yaml"], default="json")
    p_exp.add_argument("--output", type=str, default="export/")

    return parser


# ============================================================
# Command Dispatch
# ============================================================
def main():
    parser = build_parser()
    args = parser.parse_args()

    project: Project = Project()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Load project configuration if needed
    if args.command not in ["init", "help"]:
        config = project.load(args.project, args.config)
    else:
        config = None

    # Dispatch commands
    try:
        if args.command == "init":
            project.create_project(args)
        elif args.command == "validate":

            # p_val.add_argument("-q", "--quiet", action="store_true")
            # p_val.add_argument("--strict", action="store_true")

            project.validate(config, args)
        elif args.command == "run":
            project.run(config, args)
        elif args.command == "model":
            project.run_modeling(config, args)
        elif args.command == "train":
            project.run_training(config, args)
        elif args.command == "analyze":
            project.run_analysis(config, args)
        elif args.command == "refine":
            project.run_refinement(config, args)
        elif args.command == "deploy":
            project.deploy(config, args)
        elif args.command == "status":
            project.display(config)
        elif args.command == "clean":
            project.cleanup(config, args)
        elif args.command == "export":
            project.export(config, args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        logger.error("Execution interrupted by user.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
