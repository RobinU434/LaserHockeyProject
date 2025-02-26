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

        case "train-sb3-sac-hockey":
            api.hydra_plugin.hydra_wrapper(
                module.train_sb3_sac_hockey,
                args,
                command_parser["train_sb3_sac_hockey"],
                config_var_name="config",
                version_base=None,
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_sb_sac.yaml",
            )

        case "train-sb3-sac-sp":
            api.hydra_plugin.hydra_wrapper(
                module.train_sb3_sac_sp,
                args,
                command_parser["train_sb3_sac_sp"],
                config_var_name="config",
                version_base=None,
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_sb_sac.yaml",
            )

        case "train-sb3-sac-gym":
            api.hydra_plugin.hydra_wrapper(
                module.train_sb3_sac_gym,
                args,
                command_parser["train_sb3_sac_gym"],
                config_var_name="config",
                version_base=None,
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_sb_sac.yaml",
            )

        case "train-sb3-er-sac-gym":
            api.hydra_plugin.hydra_wrapper(
                module.train_sb3_er_sac_gym,
                args,
                command_parser["train_sb3_er_sac_gym"],
                config_var_name="config",
                version_base=None,
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_sb_sac.yaml",
            )

        case "upload-sb3-sac":
            module.upload_sb3_sac(
                checkpoint=args["checkpoint"],
                server_url=args["server_url"],
                server_port=args["server_port"],
                token=args["token"],
            )

        case "train-sb3-er-sac-hockey":
            api.hydra_plugin.hydra_wrapper(
                module.train_sb3_er_sac_hockey,
                args,
                command_parser["train_sb3_er_sac_hockey"],
                config_var_name="config",
                version_base=None,
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_sb_sac.yaml",
            )

        case "train-sac-hockey":
            api.hydra_plugin.hydra_wrapper(
                module.train_sac_hockey,
                args,
                command_parser["train_sac_hockey"],
                config_var_name="config",
                version_base=None,
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_sac.yaml",
            )

        case "render-sac-hockey":
            module.render_sac_hockey(
                deterministic=args["deterministic"],
                strong_opponent=args["strong_opponent"],
            )

        case "train-sac-gym":
            api.hydra_plugin.hydra_wrapper(
                module.train_sac_gym,
                args,
                command_parser["train_sac_gym"],
                config_var_name="config",
                version_base=None,
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_sac.yaml",
            )

        case "eval-sac":
            module.eval_sac(
                checkpoint=args["checkpoint"],
                n_games=args["n_games"],
                deterministic=args["deterministic"],
            )

        case "render-sac-gym":
            module.render_sac_gym(
                checkpoint=args["checkpoint"],
                gym_env=args["gym_env"],
                deterministic=args["deterministic"],
                max_steps=args["max_steps"],
            )

        case "train-dyna-hockey":
            api.hydra_plugin.hydra_wrapper(
                module.train_dyna_hockey,
                args,
                command_parser["train_dyna_hockey"],
                config_var_name="config",
                version_base=None,
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_dyna.yaml",
            )

        case "train-dyna-gym":
            api.hydra_plugin.hydra_wrapper(
                module.train_dyna_gym,
                args,
                command_parser["train_dyna_gym"],
                config_var_name="config",
                version_base=None,
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_dyna.yaml",
            )

        case "render-dyna-gym":
            module.render_dyna_gym(
                checkpoint=args["checkpoint"],
                gym_env=args["gym_env"],
                deterministic=args["deterministic"],
                max_steps=args["max_steps"],
            )

        case "train-md-dyna-gym":
            api.hydra_plugin.hydra_wrapper(
                module.train_md_dyna_gym,
                args,
                command_parser["train_md_dyna_gym"],
                config_var_name="config",
                version_base=None,
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_md_dyna.yaml",
            )

        case "train-er-dyna-gym":
            api.hydra_plugin.hydra_wrapper(
                module.train_er_dyna_gym,
                args,
                command_parser["train_er_dyna_gym"],
                config_var_name="config",
                version_base=None,
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_er_dyna.yaml",
            )

        case "train-er-dyna-hockey":
            api.hydra_plugin.hydra_wrapper(
                module.train_er_dyna_hockey,
                args,
                command_parser["train_er_dyna_hockey"],
                config_var_name="config",
                version_base=None,
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_er_dyna.yaml",
            )

        case "upload-dyna":
            module.upload_dyna(
                checkpoint=args["checkpoint"],
                server_url=args["server_url"],
                server_port=args["server_port"],
                token=args["token"],
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
