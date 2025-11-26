from __future__ import annotations

import tyro
import lightning as L
import torch as th
from tqdm import tqdm

from splat_init.data.datamodule_360 import SceneSampleLazy
from splat_init.data.threesixty_loc import ThreeSixtyLocDataset
from splat_init.data.stanford_2d_3d import Stanford2d3dDataset
from splat_init.models.sequence_chunker import SequenceChunker
from splat_init.models.vggt_perspective_transform import VggtPerspectiveTransform
from splat_init.models.vggt_naive_equirectangular import VggtNaiveEquirectangular
from configs.evaluation_args import Args



# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _build_dataset(args: Args) -> ThreeSixtyLocDataset[SceneSampleLazy] | Stanford2d3dDataset[SceneSampleLazy]:
    if args.data.dataset_name == "stanford_2d_3d":
        return Stanford2d3dDataset(
            SceneSampleLazy,
            args.data.dataset_dir,
            perspective_loader_threads=args.data.dataloader_workers
        )
    elif args.data.dataset_name == "360_loc":
        return ThreeSixtyLocDataset(
            SceneSampleLazy,
            args.data.dataset_dir,
            stride=args.data.dataset_stride,
            depth_required=False,
            worker_count=args.data.dataloader_workers
        )
    else:
        raise ValueError(f"Unknown dataset: {args.data.dataset_name}")

def _build_model(args: Args) -> L.LightningModule:
    models = {
        "vggt_perspective_transform": VggtPerspectiveTransform,
        "vggt_naive_equirectangular": VggtNaiveEquirectangular,
    }

    return SequenceChunker(
        model=models[args.model.model](),
        chunk_size=args.model.chunker_chunk_size,
        chunk_overlap=args.model.chunker_chunk_overlap
    )

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    args = tyro.cli(Args)
    th.set_float32_matmul_precision("medium")
    L.seed_everything(args.seed)

    dataset = _build_dataset(args)
    model = _build_model(args)

    for scene in tqdm(dataset, desc="Evaluating scenes"):
        # Inference
        poses, _, _ = model([scene])

        # Metrics
        # TODO: Calculate metrics

        # Store
        output_dir = args.output_dir / scene.id
        output_dir.mkdir(parents=True, exist_ok=True)

        th.save({"poses": poses.cpu()}, output_dir / "model_output.pt")
        # TODO: Output metrics to output 
        th.save({ }, output_dir / "metrics.pt")
        


if __name__ == "__main__":
    main()
