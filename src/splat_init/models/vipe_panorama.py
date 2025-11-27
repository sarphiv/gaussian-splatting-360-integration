"""LightningModule wrapper around ViPE's panorama pipeline.

The module mirrors the behaviour of running ``vipe infer -p panorama`` but
operates directly on in-memory panorama tensors shaped ``[B, S, C, H, W]``.
It builds a minimal VideoStream backed by those tensors, runs the ViPE
panorama SLAM pipeline, and returns world-to-camera pose matrices alongside an
optional depth projection derived from the reconstructed SLAM map.
"""

from __future__ import annotations

from typing import Iterator

import hydra
import numpy as np
import torch as th
from lightning.pytorch import LightningModule
from omegaconf import DictConfig

from vipe import get_config_path
from vipe.ext import lietorch as lt
from vipe.pipeline.processors import EquirectProjectionProcessor, TrackAnythingProcessor
from vipe.slam.interface import SLAMMap, SLAMOutput
from vipe.slam.system import SLAMSystem
from vipe.streams.base import CachedVideoStream, FrameAttribute, ProcessedVideoStream, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType
from vipe.utils.geometry import so3_to_se3, se3_to_so3


class _TensorVideoStream(VideoStream):
    """VideoStream backed by an in-memory tensor of panorama frames."""

    def __init__(self, frames: th.Tensor, name: str, fps: float) -> None:
        super().__init__()
        assert frames.dim() == 4, "Expected tensor shaped [S, C, H, W]"
        self._frames = frames
        self._len = frames.shape[0]
        self._size = (frames.shape[-2], frames.shape[-1])
        self._name = name
        self._fps = fps

    def frame_size(self) -> tuple[int, int]:
        return self._size

    def attributes(self) -> set[FrameAttribute]:
        return {FrameAttribute.CAMERA_TYPE}

    def fps(self) -> float:
        return self._fps

    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[VideoFrame]:
        for idx in range(self._len):
            yield self._make_frame(idx)

    def _make_frame(self, idx: int) -> VideoFrame:
        frame = self._frames[idx]
        assert frame.dim() == 3, "Frame tensor must be [C, H, W]"

        rgb = frame[:3]
        if frame.shape[0] == 4:
            rgb = rgb * frame[3:4]

        rgb = rgb.permute(1, 2, 0).contiguous()
        rgb = rgb.clamp(0.0, 1.0)

        return VideoFrame(
            raw_frame_idx=idx,
            rgb=rgb,
            camera_type=CameraType.PANORAMA,
        )


class VipePanorama(LightningModule):
    """Run ViPE's panorama pipeline on pre-loaded panorama tensors."""

    def __init__(
        self,
        fps: float = 30.0,
        return_depth: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.fps = fps
        self.return_depth = return_depth
        self.pipeline_cfg = self._compose_config(
            "panorama",
            virtual_num_views=4,
            use_top=True,
            use_bottom=False,
        )
        self.init_cfg = self.pipeline_cfg.init
        self.virtual_cfg = self.pipeline_cfg.virtual
        self.slam_cfg = self.pipeline_cfg.slam

        self.virtual_intrinsics, self.virtual_size = self._compute_virtual_camera(self.virtual_cfg)
        self.rig_transforms = self._build_rig(self.virtual_cfg)

    @staticmethod
    def _compose_config(
        pipeline_name: str,
        *,
        virtual_num_views: int | None,
        use_top: bool | None,
        use_bottom: bool | None,
    ) -> DictConfig:
        """Load the Hydra config used by the ViPE CLI for the requested pipeline."""

        overrides = [
            f"pipeline={pipeline_name}",
            "pipeline.output.save_artifacts=false",
            "pipeline.output.save_viz=false",
        ]

        if virtual_num_views is not None:
            overrides.append(f"pipeline.virtual.num_views={virtual_num_views}")
        if use_top is not None:
            overrides.append(f"pipeline.virtual.top={str(use_top).lower()}")
        if use_bottom is not None:
            overrides.append(f"pipeline.virtual.bottom={str(use_bottom).lower()}")

        with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
            args = hydra.compose("default", overrides=overrides)

        return args.pipeline

    @staticmethod
    def _compute_virtual_camera(virtual_cfg: DictConfig) -> tuple[th.Tensor, tuple[int, int]]:
        """Return intrinsics and (height, width) for the virtual pinhole rig."""

        virtual_height = int(virtual_cfg.height)
        focal = virtual_height / (2 * np.tan(np.deg2rad(float(virtual_cfg.fovx)) / 2))
        virtual_width = int(focal * np.tan(np.deg2rad(float(virtual_cfg.fovx)) / 2) * 2)
        virtual_width += virtual_width % 2

        intrinsics = th.tensor((focal, focal, virtual_width // 2, virtual_height // 2), dtype=th.float32)
        return intrinsics, (virtual_height, virtual_width)

    @staticmethod
    def _build_rig(virtual_cfg: DictConfig) -> list[lt.SE3]:
        """Construct the SE3 rig used to sample perspective views from a panorama."""

        rig_transforms = [
            so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(yaw, 0.0))
            for yaw in np.linspace(0, 2 * np.pi, virtual_cfg.num_views, endpoint=False)
        ]
        if virtual_cfg.top:
            rig_transforms.append(so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(0.0, np.pi / 2)))
        if virtual_cfg.bottom:
            rig_transforms.append(so3_to_se3(EquirectProjectionProcessor.yaw_pitch_to_rotation(0.0, -np.pi / 2)))
        return rig_transforms

    def _build_stream(self, images: th.Tensor) -> VideoStream:
        """Create a ViPE VideoStream from a batch of panorama tensors."""

        assert images.dim() == 4, "Expected tensor shaped [S, C, H, W]"
        frames = images.to(device=self.device, dtype=th.float32)
        return _TensorVideoStream(frames, name="tensor_sequence", fps=self.fps)

    def _apply_init_processors(self, video_stream: VideoStream) -> VideoStream:
        """Attach TrackAnything preprocessing when enabled in the config."""

        if getattr(self.init_cfg, "instance", None) is None:
            raise ValueError("TrackAnything config is missing; ensure pipeline.init.instance is set")

        processor = TrackAnythingProcessor(
            self.init_cfg.instance.phrases,
            add_sky=self.init_cfg.instance.add_sky,
            sam_run_gap=int(max(video_stream.fps() * self.init_cfg.instance.kf_gap_sec, 1)),
        )
        return ProcessedVideoStream(video_stream, [processor])

    def _build_slam_streams(self, video_stream: VideoStream) -> tuple[list[VideoStream], lt.SE3]:
        """Project the panorama stream into a rig of perspective streams."""

        cached = CachedVideoStream(video_stream)
        slam_streams: list[VideoStream] = []
        for rig_transform in self.rig_transforms:
            projector = EquirectProjectionProcessor(
                se3_to_so3(rig_transform),
                self.virtual_size,
                self.virtual_intrinsics,
            )
            slam_streams.append(ProcessedVideoStream(cached, [projector]).cache(online=True))

        rig_se3 = lt.stack(self.rig_transforms, dim=0)
        return slam_streams, rig_se3

    def _run_slam(self, slam_streams: list[VideoStream], rig: lt.SE3) -> SLAMOutput:
        """Run the ViPE SLAM backend."""

        slam_pipeline = SLAMSystem(device=th.device("cuda"), config=self.slam_cfg)
        return slam_pipeline.run(slam_streams, rig=rig)

    def _project_depth(self, trajectory: lt.SE3, slam_map: SLAMMap | None, size: tuple[int, int]) -> th.Tensor | None:
        """Project the reconstructed SLAM map back onto the panorama grid."""

        if slam_map is None:
            return None

        depth_maps = []
        intrinsics = th.zeros(4, device=self.device, dtype=th.float32)
        for frame_idx in range(trajectory.shape[0]):
            depth = slam_map.project_map(
                frame_idx,
                view_idx=-1,
                target_size=size,
                target_intrinsics=intrinsics,
                target_pose=trajectory[frame_idx],
                target_camera_type=CameraType.PANORAMA,
                infill=True,
            )
            depth_maps.append(depth)

        depth_stack = th.stack(depth_maps, dim=0).unsqueeze(1)
        return depth_stack

    def forward(self, images: th.Tensor) -> tuple[th.Tensor, th.Tensor | None, dict[str, th.Tensor]]:
        """Estimate camera poses and optional depth from panorama tensors."""

        if images.dim() != 5:
            raise ValueError("Expected input shaped [B, S, C, H, W]")
        batch, seq_len, _, height, width = images.shape
        if batch != 1:
            raise ValueError("ViPE only supports batch size 1")
        if self.device.type != "cuda":
            raise RuntimeError("ViPE requires CUDA; move the module to a CUDA device first")

        video_stream = self._build_stream(images[0])
        video_stream = self._apply_init_processors(video_stream)
        slam_streams, rig = self._build_slam_streams(video_stream)

        with th.inference_mode():
            slam_output = self._run_slam(slam_streams, rig)
            trajectory = slam_output.trajectory
            if trajectory.shape[0] != seq_len:
                raise RuntimeError(
                    f"SLAM returned {trajectory.shape[0]} poses for a sequence of length {seq_len}"
                )

            pose_c2w = trajectory.matrix()
            pose_w2c = th.linalg.inv(pose_c2w)

            depth = self._project_depth(trajectory, slam_output.slam_map, (height, width)) if self.return_depth else None

        return pose_w2c.unsqueeze(0), depth.unsqueeze(0) if depth is not None else None, {}

    def configure_optimizers(self):
        """Lightning hook for compatibility; ViPE is inference-only."""

        return []
