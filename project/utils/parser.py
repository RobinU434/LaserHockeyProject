from argparse import ArgumentParser
from typing import Tuple, Dict, List


def add_upload_dyna_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--checkpoint",
        help="path to SB3 sac checkpoint",
        dest="checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--server-url",
        help="URL of the server.",
        dest="server_url",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--server-port",
        help="Port of the server.",
        dest="server_port",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--token",
        help="Your access token.",
        dest="token",
        type=str,
        required=True,
    )
    return parser


def add_train_er_dyna_hockey_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--force",
        help="--no-documentation-exists--",
        dest="force",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--quiet",
        help="--no-documentation-exists--",
        dest="quiet",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--device",
        help="--no-documentation-exists--",
        dest="device",
        type=str,
        default="cpu",
        required=False,
    )
    return parser


def add_train_er_dyna_gym_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--gym-env",
        help="--no-documentation-exists--",
        dest="gym_env",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--n-actions",
        help="--no-documentation-exists--",
        dest="n_actions",
        type=int,
        default=10,
        required=False,
    )
    parser.add_argument(
        "--force",
        help="--no-documentation-exists--",
        dest="force",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--quiet",
        help="--no-documentation-exists--",
        dest="quiet",
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
    parser.add_argument(
        "--device",
        help="--no-documentation-exists--",
        dest="device",
        type=str,
        default="cpu",
        required=False,
    )
    return parser


def add_train_md_dyna_gym_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--gym-env",
        help="--no-documentation-exists--",
        dest="gym_env",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--n-actions",
        help="--no-documentation-exists--",
        dest="n_actions",
        type=int,
        default=10,
        required=False,
    )
    parser.add_argument(
        "--force",
        help="--no-documentation-exists--",
        dest="force",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--quiet",
        help="--no-documentation-exists--",
        dest="quiet",
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
    parser.add_argument(
        "--device",
        help="--no-documentation-exists--",
        dest="device",
        type=str,
        default="cpu",
        required=False,
    )
    return parser


def add_render_dyna_gym_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--checkpoint",
        help="--no-documentation-exists--",
        dest="checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--gym-env",
        help="--no-documentation-exists--",
        dest="gym_env",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--deterministic",
        help="--no-documentation-exists--",
        dest="deterministic",
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


def add_train_dyna_gym_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--gym-env",
        help="--no-documentation-exists--",
        dest="gym_env",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--n-actions",
        help="--no-documentation-exists--",
        dest="n_actions",
        type=int,
        default=10,
        required=False,
    )
    parser.add_argument(
        "--force",
        help="--no-documentation-exists--",
        dest="force",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--quiet",
        help="--no-documentation-exists--",
        dest="quiet",
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
    parser.add_argument(
        "--device",
        help="--no-documentation-exists--",
        dest="device",
        type=str,
        default="cpu",
        required=False,
    )
    return parser


def add_train_dyna_hockey_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--force",
        help="--no-documentation-exists--",
        dest="force",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--device",
        help="--no-documentation-exists--",
        dest="device",
        type=str,
        default="cpu",
        required=False,
    )
    parser.add_argument(
        "--quiet",
        help="--no-documentation-exists--",
        dest="quiet",
        action="store_true",
        required=False,
    )
    return parser


def add_render_sac_gym_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--checkpoint",
        help="--no-documentation-exists--",
        dest="checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--gym-env",
        help="--no-documentation-exists--",
        dest="gym_env",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--deterministic",
        help="--no-documentation-exists--",
        dest="deterministic",
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


def add_eval_sac_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--checkpoint",
        help="--no-documentation-exists--",
        dest="checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--n-games",
        help="--no-documentation-exists--",
        dest="n_games",
        type=int,
        default=10,
        required=False,
    )
    parser.add_argument(
        "--deterministic",
        help="--no-documentation-exists--",
        dest="deterministic",
        action="store_true",
        required=False,
    )
    return parser


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
    parser.add_argument(
        "--device",
        help="--no-documentation-exists--",
        dest="device",
        type=str,
        default="cpu",
        required=False,
    )
    parser.add_argument(
        "--quiet",
        help="--no-documentation-exists--",
        dest="quiet",
        action="store_true",
        required=False,
    )
    return parser


def add_render_sac_hockey_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--deterministic",
        help="--no-documentation-exists--",
        dest="deterministic",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--strong-opponent",
        help="--no-documentation-exists--",
        dest="strong_opponent",
        action="store_true",
        required=False,
    )
    return parser


def add_train_sac_hockey_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--force",
        help="--no-documentation-exists--",
        dest="force",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--device",
        help="--no-documentation-exists--",
        dest="device",
        type=str,
        default="cpu",
        required=False,
    )
    return parser


def add_train_sb3_er_sac_hockey_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--force",
        help="--no-documentation-exists--",
        dest="force",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--device",
        help="--no-documentation-exists--",
        dest="device",
        type=str,
        default="cpu",
        required=False,
    )
    return parser


def add_upload_sb3_sac_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--checkpoint",
        help="path to SB3 sac checkpoint",
        dest="checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--server-url",
        help="URL of the server.",
        dest="server_url",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--server-port",
        help="Port of the server.",
        dest="server_port",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--token",
        help="Your access token.",
        dest="token",
        type=str,
        required=True,
    )
    return parser


def add_train_sb3_er_sac_gym_args(parser: ArgumentParser) -> ArgumentParser:
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
        "--device",
        help="--no-documentation-exists--",
        dest="device",
        type=str,
        default="cpu",
        required=False,
    )
    return parser


def add_train_sb3_sac_gym_args(parser: ArgumentParser) -> ArgumentParser:
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
        "--device",
        help="--no-documentation-exists--",
        dest="device",
        type=str,
        default="cpu",
        required=False,
    )
    return parser


def add_train_sb3_sac_sp_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--force",
        help="--no-documentation-exists--",
        dest="force",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--device",
        help="--no-documentation-exists--",
        dest="device",
        type=str,
        default="cpu",
        required=False,
    )
    return parser


from pyargwriter.api.hydra_plugin import add_hydra_parser


def add_train_sb3_sac_hockey_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--force",
        help="--no-documentation-exists--",
        dest="force",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--device",
        help="--no-documentation-exists--",
        dest="device",
        type=str,
        default="cpu",
        required=False,
    )
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
    train_sb3_sac_hockey = command_subparser.add_parser(
        "train-sb3-sac-hockey", help="--no-documentation-exists--"
    )
    train_sb3_sac_hockey = add_train_sb3_sac_hockey_args(train_sb3_sac_hockey)
    train_sb3_sac_hockey = add_hydra_parser(train_sb3_sac_hockey)
    subparser["train_sb3_sac_hockey"] = train_sb3_sac_hockey
    train_sb3_sac_sp = command_subparser.add_parser(
        "train-sb3-sac-sp", help="--no-documentation-exists--"
    )
    train_sb3_sac_sp = add_train_sb3_sac_sp_args(train_sb3_sac_sp)
    train_sb3_sac_sp = add_hydra_parser(train_sb3_sac_sp)
    subparser["train_sb3_sac_sp"] = train_sb3_sac_sp
    train_sb3_sac_gym = command_subparser.add_parser(
        "train-sb3-sac-gym", help="--no-documentation-exists--"
    )
    train_sb3_sac_gym = add_train_sb3_sac_gym_args(train_sb3_sac_gym)
    train_sb3_sac_gym = add_hydra_parser(train_sb3_sac_gym)
    subparser["train_sb3_sac_gym"] = train_sb3_sac_gym
    train_sb3_er_sac_gym = command_subparser.add_parser(
        "train-sb3-er-sac-gym", help="--no-documentation-exists--"
    )
    train_sb3_er_sac_gym = add_train_sb3_er_sac_gym_args(train_sb3_er_sac_gym)
    train_sb3_er_sac_gym = add_hydra_parser(train_sb3_er_sac_gym)
    subparser["train_sb3_er_sac_gym"] = train_sb3_er_sac_gym
    upload_sb3_sac = command_subparser.add_parser(
        "upload-sb3-sac", help="upload sb3 sac agent to competition server"
    )
    upload_sb3_sac = add_upload_sb3_sac_args(upload_sb3_sac)
    subparser["upload_sb3_sac"] = upload_sb3_sac
    train_sb3_er_sac_hockey = command_subparser.add_parser(
        "train-sb3-er-sac-hockey", help="--no-documentation-exists--"
    )
    train_sb3_er_sac_hockey = add_train_sb3_er_sac_hockey_args(train_sb3_er_sac_hockey)
    train_sb3_er_sac_hockey = add_hydra_parser(train_sb3_er_sac_hockey)
    subparser["train_sb3_er_sac_hockey"] = train_sb3_er_sac_hockey
    train_sac_hockey = command_subparser.add_parser(
        "train-sac-hockey", help="--no-documentation-exists--"
    )
    train_sac_hockey = add_train_sac_hockey_args(train_sac_hockey)
    train_sac_hockey = add_hydra_parser(train_sac_hockey)
    subparser["train_sac_hockey"] = train_sac_hockey
    render_sac_hockey = command_subparser.add_parser(
        "render-sac-hockey", help="--no-documentation-exists--"
    )
    render_sac_hockey = add_render_sac_hockey_args(render_sac_hockey)
    subparser["render_sac_hockey"] = render_sac_hockey
    train_sac_gym = command_subparser.add_parser(
        "train-sac-gym", help="--no-documentation-exists--"
    )
    train_sac_gym = add_train_sac_gym_args(train_sac_gym)
    train_sac_gym = add_hydra_parser(train_sac_gym)
    subparser["train_sac_gym"] = train_sac_gym
    eval_sac = command_subparser.add_parser(
        "eval-sac", help="--no-documentation-exists--"
    )
    eval_sac = add_eval_sac_args(eval_sac)
    subparser["eval_sac"] = eval_sac
    render_sac_gym = command_subparser.add_parser(
        "render-sac-gym", help="--no-documentation-exists--"
    )
    render_sac_gym = add_render_sac_gym_args(render_sac_gym)
    subparser["render_sac_gym"] = render_sac_gym
    train_dyna_hockey = command_subparser.add_parser(
        "train-dyna-hockey", help="--no-documentation-exists--"
    )
    train_dyna_hockey = add_train_dyna_hockey_args(train_dyna_hockey)
    train_dyna_hockey = add_hydra_parser(train_dyna_hockey)
    subparser["train_dyna_hockey"] = train_dyna_hockey
    train_dyna_gym = command_subparser.add_parser(
        "train-dyna-gym", help="--no-documentation-exists--"
    )
    train_dyna_gym = add_train_dyna_gym_args(train_dyna_gym)
    train_dyna_gym = add_hydra_parser(train_dyna_gym)
    subparser["train_dyna_gym"] = train_dyna_gym
    render_dyna_gym = command_subparser.add_parser(
        "render-dyna-gym", help="--no-documentation-exists--"
    )
    render_dyna_gym = add_render_dyna_gym_args(render_dyna_gym)
    subparser["render_dyna_gym"] = render_dyna_gym
    train_md_dyna_gym = command_subparser.add_parser(
        "train-md-dyna-gym", help="--no-documentation-exists--"
    )
    train_md_dyna_gym = add_train_md_dyna_gym_args(train_md_dyna_gym)
    train_md_dyna_gym = add_hydra_parser(train_md_dyna_gym)
    subparser["train_md_dyna_gym"] = train_md_dyna_gym
    train_er_dyna_gym = command_subparser.add_parser(
        "train-er-dyna-gym", help="--no-documentation-exists--"
    )
    train_er_dyna_gym = add_train_er_dyna_gym_args(train_er_dyna_gym)
    train_er_dyna_gym = add_hydra_parser(train_er_dyna_gym)
    subparser["train_er_dyna_gym"] = train_er_dyna_gym
    train_er_dyna_hockey = command_subparser.add_parser(
        "train-er-dyna-hockey", help="--no-documentation-exists--"
    )
    train_er_dyna_hockey = add_train_er_dyna_hockey_args(train_er_dyna_hockey)
    train_er_dyna_hockey = add_hydra_parser(train_er_dyna_hockey)
    subparser["train_er_dyna_hockey"] = train_er_dyna_hockey
    upload_dyna = command_subparser.add_parser(
        "upload-dyna", help="upload sb3 sac agent to competition server"
    )
    upload_dyna = add_upload_dyna_args(upload_dyna)
    subparser["upload_dyna"] = upload_dyna
    return parser, subparser


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser, _ = setup_entrypoint_parser(parser)
    return parser
