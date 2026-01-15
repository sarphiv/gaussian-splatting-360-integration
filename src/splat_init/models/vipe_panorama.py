"""LightningModule wrapper around ViPE's panorama pipeline.

The module mirrors the behaviour of running ``vipe infer -p panorama`` but
operates directly on in-memory panorama tensors shaped ``[B, S, C, H, W]``.
It builds a minimal VideoStream backed by those tensors, runs the ViPE
panorama SLAM pipeline, and returns world-to-camera pose matrices alongside
keypoints projected from the reconstructed SLAM map.
"""

from __future__ import annotations

import math
from typing import Iterator, cast

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
from vipe.utils.geometry import project_points_to_panorama, se3_to_so3, so3_to_se3



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
        return_depth: bool = True
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.fps = fps
        self.return_depth = return_depth
        self.pipeline_cfg = self._compose_config(
            "panorama",
            virtual_num_views=4,
            use_top=False,
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
        frames = images.to(device=self.device, dtype=cast(th.dtype, self.dtype))
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

    def _empty_cuda_cache(self) -> None:
        """Attempt to empty the CUDA cache to free up memory"""
        if self.device.type == "cuda" and th.cuda.is_available():
            th.cuda.empty_cache()

    def _run_slam(self, slam_streams: list[VideoStream], rig: lt.SE3) -> SLAMOutput:
        """Run the ViPE SLAM backend."""

        slam_pipeline = SLAMSystem(device=self.device, config=self.slam_cfg)
        return slam_pipeline.run(slam_streams, rig=rig)

    def _sample_keypoints_from_slam_map(
        self,
        pose_w2c: th.Tensor,
        slam_map: SLAMMap,
        size: tuple[int, int],
        sample_ratio: float = 0.001,
        tstamp_nn: int = 3,
    ) -> list[tuple[th.Tensor, th.Tensor]]:
        """Project SLAM map points into panorama frames and sample keypoints."""

        assert pose_w2c.shape[-2:] == (4, 4), "Expected pose matrices shaped [S,4,4]"
        assert 0.0 < sample_ratio <= 1.0, "Sample ratio must be in (0, 1]"

        height, width = size
        frame_inds = np.asarray(slam_map.dense_disp_frame_inds, dtype=np.int64)
        dtype = slam_map.dense_disp_xyz.dtype

        empty_xy = th.empty((2, 0), device=th.device("cpu"), dtype=dtype)
        empty_xyz = th.empty((3, 0), device=th.device("cpu"), dtype=dtype)
        if frame_inds.size == 0:
            return [(empty_xy, empty_xyz) for _ in range(pose_w2c.shape[0])]

        keypoints: list[tuple[th.Tensor, th.Tensor]] = []
        for frame_idx in range(pose_w2c.shape[0]):
            right_keyframe_idx = int(np.searchsorted(frame_inds, frame_idx))
            right_keyframe_idx = min(right_keyframe_idx + tstamp_nn, len(frame_inds) - 1)
            left_keyframe_idx = max(right_keyframe_idx - 2 * tstamp_nn, 0)

            xyz_list: list[th.Tensor] = []
            for keyframe_idx in range(left_keyframe_idx, right_keyframe_idx + 1):
                xyz_kf, _ = slam_map.get_dense_disp_pcd(keyframe_idx, view_idx=-1)
                if xyz_kf.numel() > 0:
                    xyz_list.append(xyz_kf)

            if not xyz_list:
                keypoints.append((empty_xy, empty_xyz))
                continue

            xyz_world = th.cat(xyz_list, dim=0)
            # SLAM map points are in world coordinates; move into the panorama camera frame.
            w2c = pose_w2c[frame_idx].to(device=xyz_world.device, dtype=xyz_world.dtype)
            xyz_cam = xyz_world @ w2c[:3, :3].T + w2c[:3, 3]

            uvd = project_points_to_panorama(xyz_cam, return_depth=True)
            # Convert normalized panorama UV to pixel centers.
            x = uvd[:, 0] * width - 0.5
            y = uvd[:, 1] * height - 0.5
            depth = uvd[:, 2]

            valid = (
                (depth > 0.0)
                & th.isfinite(x)
                & th.isfinite(y)
                & (x >= 0.0)
                & (x < width)
                & (y >= 0.0)
                & (y < height)
            )
            if not valid.any():
                keypoints.append((empty_xy, empty_xyz))
                continue

            xyz_valid = xyz_world[valid]
            x = x[valid]
            y = y[valid]

            num_valid = xyz_valid.shape[0]
            num_samples = min(num_valid, math.ceil(sample_ratio * num_valid))
            if num_samples < num_valid:
                perm = th.randperm(num_valid, device=xyz_valid.device)[:num_samples]
                xyz_valid = xyz_valid[perm]
                x = x[perm]
                y = y[perm]

            xy = th.stack((x, y), dim=0)
            keypoints.append((xy.cpu(), xyz_valid.transpose(0, 1).cpu()))

        return keypoints

    def forward(
        self, images: th.Tensor
    ) -> tuple[th.Tensor, list[tuple[th.Tensor, th.Tensor]] | None, dict[str, th.Tensor]]:
        """Estimate camera poses and SLAM-map-derived keypoints from panorama tensors."""

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
            self._empty_cuda_cache()
            slam_output = self._run_slam(slam_streams, rig)
            trajectory = slam_output.trajectory
            assert trajectory.shape[0] == seq_len, "SLAM trajectory length mismatch"
            self._empty_cuda_cache()

            pose_c2w = trajectory.matrix()
            pose_w2c = th.linalg.inv(pose_c2w)

        if not self.return_depth or slam_output.slam_map is None:
            keypoints = [
                (
                    th.empty((2, 0), device=th.device("cpu"), dtype=images.dtype),
                    th.empty((3, 0), device=th.device("cpu"), dtype=images.dtype),
                )
                for _ in range(seq_len)
            ]
        else:
            keypoints = self._sample_keypoints_from_slam_map(
                pose_w2c.detach(),
                slam_output.slam_map,
                (height, width),
                sample_ratio=0.001,
            )

        return pose_w2c.unsqueeze(0).to(images), keypoints, {}

    def configure_optimizers(self):
        """Lightning hook for compatibility; ViPE is inference-only."""

        return []
