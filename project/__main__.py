from argparse import ArgumentParser
from pathlib import Path
from pyargwriter import api
from project.utils.parser import setup_entrypoint_parser
from project.entrypoint import Entrypoint
from project.utils.parser import setup_parser


def execute(args: dict) -> bool:
    module = Entrypoint()
    _, command_parser = setup_entrypoint_parser(ArgumentParser())
    match args["command"]:
        case "train-dreamer":
            module.train_dreamer()

        case "train-sb3-sac":
            module.train_sb3_sac()

        case "train-sac":
            api.hydra_plugin.hydra_wrapper(
                module.train_sac,
                args,
                command_parser["train_sac"],
                config_var_name="config",
                version_base=None,
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_sac.yaml",
            )

        case "train-sac-pendulum":
            api.hydra_plugin.hydra_wrapper(
                module.train_sac_pendulum,
                args,
                command_parser["train_sac_pendulum"],
                config_var_name="config",
                version_base=None,
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_sac.yaml",
            )

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
