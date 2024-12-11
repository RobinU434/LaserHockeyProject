from argparse import ArgumentParser


def add_train_sb3_sac_args(parser: ArgumentParser) -> ArgumentParser:
    return parser


def add_train_dreamer_args(parser: ArgumentParser) -> ArgumentParser:
    return parser


def setup_entrypoint_parser(parser: ArgumentParser) -> ArgumentParser:
    command_subparser = parser.add_subparsers(dest="command", title="command")
    train_dreamer = command_subparser.add_parser("train-dreamer", help="_summary_")
    train_dreamer = add_train_dreamer_args(train_dreamer)
    train_sb3_sac = command_subparser.add_parser("train-sb3-sac", help="_summary_")
    train_sb3_sac = add_train_sb3_sac_args(train_sb3_sac)
    return parser


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser = setup_entrypoint_parser(parser)
    return parser
