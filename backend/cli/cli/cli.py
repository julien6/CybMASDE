#!/usr/bin/env python3
"""
CybMASDE Command-Line Interface (CLI)
====================================

This CLI provides user-friendly access to the CybMASDE platform.
It mirrors the functionality of the Project class defined in project.py,
offering subcommands for creation, validation, execution, training,
analysis, refinement, exporting, and cleaning.

Usage examples:
---------------
  cybmasde create my_project --desc "Demo" --output ~/Documents
  cybmasde run --project ~/Documents/my_project
  cybmasde train --project ~/Documents/my_project --algo mappo --epochs 5
  cybmasde analyze --project ~/Documents/my_project --auto-temm
  cybmasde refine --project ~/Documents/my_project --max 2
  cybmasde export --project ~/Documents/my_project --format json --output export/
  cybmasde cleanup --project ~/Documents/my_project --all
"""

import argparse
import sys
from world_model.project import Project


def main():
    parser = argparse.ArgumentParser(
        prog="cybmasde",
        description="CybMASDE CLI — Multi-Agent System Design Environment",
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available subcommands")

    # -----------------------------
    # create
    # -----------------------------
    p_create = subparsers.add_parser(
        "create", help="Create a new CybMASDE project")
    p_create.add_argument("name", type=str, help="Project name")
    p_create.add_argument(
        "--desc", type=str, default="New CybMASDE project", help="Project description")
    p_create.add_argument("--output", type=str,
                          default=None, help="Output directory")
    p_create.add_argument("--template", type=str, default="handcrafted",
                          help="Project template (default: handcrafted)")

    # -----------------------------
    # load / validate / status
    # -----------------------------
    p_validate = subparsers.add_parser(
        "validate", help="Validate a project configuration")
    p_validate.add_argument("--project", required=True,
                            help="Path to project directory")
    p_validate.add_argument("--strict", action="store_true",
                            help="Enable strict validation mode")

    p_status = subparsers.add_parser("status", help="Display project summary")
    p_status.add_argument("--project", required=True,
                          help="Path to project directory")

    # -----------------------------
    # run
    # -----------------------------
    p_run = subparsers.add_parser("run", help="Run a full CybMASDE workflow")
    p_run.add_argument("--project", required=True,
                       help="Path to project directory")
    p_run.add_argument("--max-refine", type=int, help="Max refinement cycles")
    p_run.add_argument("--reward-threshold", type=float,
                       help="Reward threshold")
    p_run.add_argument("--std-threshold", type=float,
                       help="Std deviation threshold")
    p_run.add_argument("--semi-auto", action="store_true",
                       help="Run in semi-auto mode")
    p_run.add_argument("--manual", action="store_true",
                       help="Run in manual mode")
    p_run.add_argument("--skip-model", action="store_true",
                       help="Skip modeling phase")
    p_run.add_argument("--skip-analyze", action="store_true",
                       help="Skip analyzing phase")

    # -----------------------------
    # modelling
    # -----------------------------
    p_model = subparsers.add_parser("model", help="Run the modelling phase")
    p_model.add_argument("--project", required=True,
                         help="Path to project directory")
    p_model.add_argument("--auto", action="store_true",
                         help="Run in automatic mode")
    p_model.add_argument("--manual", action="store_true",
                         help="Run in manual mode")
    p_model.add_argument("--traces", type=str, help="Path to input traces")
    p_model.add_argument("--vae-dim", type=int,
                         help="Hint for VAE latent dimension")
    p_model.add_argument("--lstm-hidden", type=int,
                         help="Hint for LSTM hidden dimension")

    # -----------------------------
    # training
    # -----------------------------
    p_train = subparsers.add_parser("train", help="Run the training phase")
    p_train.add_argument("--project", required=True,
                         help="Path to project directory")
    p_train.add_argument("--algo", type=str,
                         help="Algorithm name (e.g., mappo)")
    p_train.add_argument("--batch-size", type=int, help="Batch size")
    p_train.add_argument("--lr", type=float, help="Learning rate")
    p_train.add_argument("--gamma", type=float, help="Discount factor")
    p_train.add_argument("--clip", type=float, help="Clipping parameter")
    p_train.add_argument("--seed", type=int, help="Random seed")
    p_train.add_argument("--epochs", type=int, help="Training epochs")

    # -----------------------------
    # analyzing
    # -----------------------------
    p_analyze = subparsers.add_parser(
        "analyze", help="Run the analyzing phase (Auto-TEMM)")
    p_analyze.add_argument("--project", required=True,
                           help="Path to project directory")
    p_analyze.add_argument("--auto-temm", action="store_true",
                           help="Enable Auto-TEMM analysis")
    p_analyze.add_argument("--metrics", nargs="+",
                           help="List of metrics to compute")
    p_analyze.add_argument("--representativity", type=float,
                           help="Representativity threshold")

    # -----------------------------
    # refining
    # -----------------------------
    p_refine = subparsers.add_parser(
        "refine", help="Run the refinement cycles")
    p_refine.add_argument("--project", required=True,
                          help="Path to project directory")
    p_refine.add_argument("--max", type=int, help="Max refinement cycles")
    p_refine.add_argument("--accept-inferred", action="store_true",
                          help="Accept inferred org specs automatically")
    p_refine.add_argument("--interactive", action="store_true",
                          help="Enable interactive refinement")

    # -----------------------------
    # deploy
    # -----------------------------
    p_deploy = subparsers.add_parser(
        "deploy", help="Deploy the trained policy")
    p_deploy.add_argument("--project", required=True,
                          help="Path to project directory")
    mode = p_deploy.add_mutually_exclusive_group(required=True)
    mode.add_argument("--direct", action="store_true",
                      help="Deploy locally (direct mode)")
    mode.add_argument("--remote", action="store_true",
                      help="Deploy remotely (via API)")
    p_deploy.add_argument("--checkpoint", type=str,
                          help="Path to policy checkpoint")
    p_deploy.add_argument(
        "--api", type=str, help="URL of target Environment API")

    # -----------------------------
    # export
    # -----------------------------
    p_export = subparsers.add_parser(
        "export", help="Export results and metrics")
    p_export.add_argument("--project", required=True,
                          help="Path to project directory")
    p_export.add_argument(
        "--format", choices=["json", "csv", "yaml"], default="json", help="Export format")
    p_export.add_argument("--output", type=str,
                          default="export/", help="Output directory")

    # -----------------------------
    # cleanup
    # -----------------------------
    p_clean = subparsers.add_parser(
        "cleanup", help="Clean temporary project data")
    p_clean.add_argument("--project", required=True,
                         help="Path to project directory")
    p_clean.add_argument("--traces", action="store_true", help="Remove traces")
    p_clean.add_argument(
        "--checkpoints", action="store_true", help="Remove checkpoints")
    p_clean.add_argument("--all", action="store_true",
                         help="Remove all temporary data")

    # Parse args and dispatch
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    project = Project()

    # Command dispatch
    try:
        if args.command == "create":
            project.create_project(args.name, args.desc,
                                   args.output, args.template)
        elif args.command == "validate":
            project.load(args.project)
            project.validate(args)
        elif args.command == "status":
            project.display(args)
        elif args.command == "run":
            project.run(args)
        elif args.command == "model":
            project.run_modeling(args)
        elif args.command == "train":
            project.run_training(args)
        elif args.command == "analyze":
            project.run_analyzing(args)
        elif args.command == "refine":
            project.run_refining(args)
        elif args.command == "deploy":
            project.run_transferring(args)
        elif args.command == "export":
            project.export(args)
        elif args.command == "cleanup":
            project.cleanup(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting gracefully.")
        project.stop()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
