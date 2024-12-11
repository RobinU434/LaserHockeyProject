from argparse import ArgumentParser
from project.entrypoint import Entrypoint
from project.utils.parser import setup_parser


def execute(args: dict) -> bool:
    module = Entrypoint()
    match args["command"]:
        case "train-dreamer":
            module.train_dreamer()

        case "train-sb3-sac":
            module.train_sb3_sac()

        case _:
            return False

    return True


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Entrypoint to experimentation landscape")

    parser = setup_parser(parser)

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    if not execute(args_dict):
        parser.print_usage()


if __name__ == "__main__":
    main()
