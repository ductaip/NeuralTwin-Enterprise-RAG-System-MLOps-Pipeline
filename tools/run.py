from datetime import datetime as dt
from pathlib import Path

import click
from loguru import logger

from codeatlas import settings
from pipelines import (
    export_artifact_to_json,
    feature_engineering,
)


@click.command(
    help="""
CodeAtlas project CLI.

Main entry point for the pipeline execution.
This entrypoint is where everything comes together.

Run the ZenML CodeAtlas pipelines with various options.

Run a pipeline with the required parameters. This executes
all steps in the pipeline in the correct order using the orchestrator
stack component that is configured in your active ZenML stack.

Examples:

  \b
  # Run the feature engineering pipeline
  python run.py --run-feature-engineering

  \b
  # Run the pipeline without cache
  python run.py --run-feature-engineering --no-cache

"""
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--run-export-artifact-to-json",
    is_flag=True,
    default=False,
    help="Whether to run the Artifact -> JSON pipeline",
)
@click.option(
    "--run-feature-engineering",
    is_flag=True,
    default=False,
    help="Whether to run the FE pipeline.",
)
@click.option(
    "--export-settings",
    is_flag=True,
    default=False,
    help="Whether to export your settings to ZenML or not.",
)
def main(
    no_cache: bool = False,
    run_export_artifact_to_json: bool = False,
    run_feature_engineering: bool = False,
    export_settings: bool = False,
) -> None:
    assert (
        run_export_artifact_to_json or run_feature_engineering or export_settings
    ), "Please specify an action to run."

    if export_settings:
        logger.info("Exporting settings to ZenML secrets.")
        settings.export()

    pipeline_args = {
        "enable_cache": not no_cache,
    }
    root_dir = Path(__file__).resolve().parent.parent

    if run_export_artifact_to_json:
        run_args_etl = {}
        pipeline_args["config_path"] = root_dir / "configs" / "export_artifact_to_json.yaml"
        assert pipeline_args["config_path"].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = f"export_artifact_to_json_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        export_artifact_to_json.with_options(**pipeline_args)(**run_args_etl)

    if run_feature_engineering:
        run_args_fe = {}
        pipeline_args["config_path"] = root_dir / "configs" / "feature_engineering.yaml"
        pipeline_args["run_name"] = f"feature_engineering_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        feature_engineering.with_options(**pipeline_args)(**run_args_fe)


if __name__ == "__main__":
    main()
