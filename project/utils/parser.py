from argparse import ArgumentParser
from typing import Tuple, Dict, List


def add_train_sac_gym_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--gym-env",
        help="--no-documentation-exists--",
        dest="gym_env",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--force",
        help="--no-documentation-exists--",
        dest="force",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--max-steps",
        help="--no-documentation-exists--",
        dest="max_steps",
        type=int,
        default=200,
        required=False,
    )
    return parser


from pyargwriter.api.hydra_plugin import add_hydra_parser


def add_train_sac_hockey_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--force",
        help="--no-documentation-exists--",
        dest="force",
        action="store_true",
        required=False,
    )
    return parser


def add_train_sb3_sac_args(parser: ArgumentParser) -> ArgumentParser:
    return parser


def add_train_dreamer_args(parser: ArgumentParser) -> ArgumentParser:
    return parser


def setup_entrypoint_parser(
    parser: ArgumentParser,
) -> Tuple[ArgumentParser, Dict[str, ArgumentParser]]:
    subparser = {}
    command_subparser = parser.add_subparsers(dest="command", title="command")
    train_dreamer = command_subparser.add_parser("train-dreamer", help="_summary_")
    train_dreamer = add_train_dreamer_args(train_dreamer)
    subparser["train_dreamer"] = train_dreamer
    train_sb3_sac = command_subparser.add_parser("train-sb3-sac", help="_summary_")
    train_sb3_sac = add_train_sb3_sac_args(train_sb3_sac)
    subparser["train_sb3_sac"] = train_sb3_sac
    train_sac_hockey = command_subparser.add_parser(
        "train-sac-hockey", help="--no-documentation-exists--"
    )
    train_sac_hockey = add_train_sac_hockey_args(train_sac_hockey)
    train_sac_hockey = add_hydra_parser(train_sac_hockey)
    subparser["train_sac_hockey"] = train_sac_hockey
    train_sac_gym = command_subparser.add_parser(
        "train-sac-gym", help="--no-documentation-exists--"
    )
    train_sac_gym = add_train_sac_gym_args(train_sac_gym)
    train_sac_gym = add_hydra_parser(train_sac_gym)
    subparser["train_sac_gym"] = train_sac_gym
    return parser, subparser


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser, _ = setup_entrypoint_parser(parser)
    return parser
