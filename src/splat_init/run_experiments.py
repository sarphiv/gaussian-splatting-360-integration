from __future__ import annotations

import subprocess
from pathlib import Path

import tyro
from tqdm import tqdm

from configs.run_experiments_args import Args, ExperimentArgs


def _experiment_results_dir(output_dir: Path, experiment: ExperimentArgs) -> Path:
    return output_dir / f"{experiment.model}-{experiment.dataset_stride}"

def main() -> None:
    args = tyro.cli(Args)

    if args.evaluate_poses:
        for experiment in (pbar := tqdm(args.experiments, desc="Evaluating poses")):
            pbar.set_postfix({"model": experiment.model, "stride": experiment.dataset_stride})

            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "splat_init.evaluate_poses",
                "--results-dir",
                str(_experiment_results_dir(args.output_dir, experiment)),
                "--data.dataset-stride",
                str(experiment.dataset_stride),
                "--data.dataset-offset",
                str(experiment.dataset_stride // 2),
                "--data.dataset-fps",
                str(2 / experiment.dataset_stride),
                "--model.model",
                experiment.model,
                "--model.chunker",
                *(
                    ["None"]
                    if experiment.chunker is None else
                    [str(experiment.chunker[0]), str(experiment.chunker[1])]
                ),
            ]
            subprocess.run(cmd, check=True)

    if args.train_splats:
        for experiment in (pbar := tqdm(args.experiments, desc="Training splats")):
            pbar.set_postfix({"model": experiment.model, "stride": experiment.dataset_stride})

            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "splat_init.train_splats",
                "--results-dir",
                str(_experiment_results_dir(args.output_dir, experiment)),
            ]
            subprocess.run(cmd, check=True)

    if args.evaluate_splats:
        for experiment in (pbar := tqdm(args.experiments, desc="Evaluating splats")):
            pbar.set_postfix({"model": experiment.model, "stride": experiment.dataset_stride})

            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "splat_init.evaluate_splats",
                "--results-dir",
                str(_experiment_results_dir(args.output_dir, experiment)),
            ]
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
