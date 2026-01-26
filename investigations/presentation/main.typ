#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

#import "@preview/numbly:0.1.0": numbly
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.institution,
  config-info(
    title: [Integrating 360° Video Capture into Modern 3D Reconstruction Methods],
    subtitle: [Investigation of initialization methods for Gaussian splatting],
    author: [sarphiv (Anonymized)],
    date: datetime.today(),
    institution: [Technical University of Denmark],
    logo: [],
  ),
  config-common(
    slide-level: 3,
    // show-notes-on-second-screen: right
  ),
  config-colors(
    neutral-lightest: rgb("#ffffff")
  )
)

#set heading(numbering: numbly("{1}.", default: "1.1"))
#set text(size: 24pt)


#title-slide()

#outline(depth: 2, indent: 1em)


= 360° captures for Gaussian splatting

== Perspective images
#speaker-note[
  + Gaussian splatting research has a lot of focus on perspective images.
  + Commonly used datasets are based on perspective images.
  + Using the common datasets, easier to argue your method is better.
  + Mip-NeRF 360 is not about 360 images, but rather about 360 orbits.
  + Of course some research focuses on 360 degree data.
]

#slide(align: horizon + center)[
  #image("media/mip-nerf360.jpg", height: 60%)
  Mip-NeRF 360
][
  #image("media/tanks-and-temples-truck.jpg", height: 60%)
  Tanks and temples
][
  #image("media/zip-nerf.jpg", height: 60%)
  Zip-NeRF
]

== 360° captures
#speaker-note[
  + In my experience, the actual users (not researchers) of Gaussian splatting mostly use 360 captures.
    + Construction industry
    + Hardware scanner companies (writes software for their camera systems)
    + Enthusiasts
    + Cultural heritage digitization
  + Notable exception are drone captures.
  + Why? At least in the construction industry:
    + Significantly faster to capture everything.
    + Less chance of user error not capturing something.
    + Pose estimation is easier, keypoints stay in "frame".
  + Companies are catching on to this
    + XGRIDS
    + Cupix
    + OpenSpace
]

#slide(align: horizon + center)[
  #image("media/openspace.jpg", height: 60%)
  OpenSpace
][
  #image("media/xgrids.jpg", height: 60%)
  XGRIDS
][
  #image("media/olli-huttunen.png", height: 60%)
  Olli Huttunen
]

#slide(align: horizon + center)[
  #image("media/equirectangular-sides.png", width: 100%)
  Sides of a square room in equirectangular format
][
  #image("media/360loc-atrium-frame.jpg", width: 100%)
  Equirectangular images enable arbitrary reframing in post
]


=== Typical hardware
#slide(align: horizon+center, composer: (2fr, 1fr))[
  #align(left)[
    - Matterport is prohibitively expensive
    #pause

    - Two very distorted fish eye cameras stitched
    - Captures almost everything (e.g. no stick)
    - "Good enough" spatial + temporal resolution
    - Used by photographers, construction industry, social media content creators, and enthusiasts
  ]
][
  #meanwhile
  #alternatives[
    #image("media/matterport-camera.jpg", height: 100%)
  ][
    #stack(
      spacing: 0.5em,
      image("media/insta360-camera.jpeg", height: 40%),
      [Insta360 X5 (2025)],
      image("media/insta360-stick.webp", height: 40%)
    )
  ]
]


=== Goal of investigation
#speaker-note[















  so since a lot of the research is focussed on perspective images
  this means the workflows are based around perspective images too.
  unfortunately this means users have to do extra work to make their 360 captures compatible
  with the existing 360 workflows.


  Gaussian initialization

  poses and initial gaussian positions

  are some of the new models better than the usual colmap?
  enthusiasts also use proprietary software nowadays like realitycapture, but we are not considering proprietary software today. 











  
]

#[
  #set align(center)
  #set text(size: 0.7em)
  #diagram(
    edge-stroke: 0.1em,
    node-corner-radius: 0.1em,
    mark-scale: 60%,
    spacing: 2em,
    node-inset: 0.6em,
    node-stroke: 0.1em,

    node((0, 0), [Insta360\ 360° capture], name: <n0>),
    node((1, 0), [Perspective\ projection], name: <n1>),
    node((2, 0), align(center)[RealityScan\ (COLMAP)], name: <n2>),
    node((3, 0), [Postshot\ (Paper)], name: <n3>),
    node((4, 0), [Gaussian splat], name: <n4>),

    edge(<n0>, <n1>, "->"),
    edge(<n1>, <n2>, "->"),
    edge(<n2>, <n3>, "->"),
    edge(<n3>, <n4>, "->")
  )
]
#v(1.4em)

+ Walk around with an Insta360 X5 on a selfie stick
+ Project 360° images to many perspective images
+ Use proprietary software to estimate poses and point cloud
+ Train Gaussian splat with proprietary easy to use software
+ Show model to others or export for use in other software
#v(1em)


#focus-slide[
  What is the reconstruction quality of various
  Gaussian splatting initialization methods for 360° captures?
]


== Datasets
#speaker-note[
  + Need 360 degree datasets to investigate this.
  + Skimmed many papers for interesting datasets with poses and depth.
  + Synthetic, design and aesthetic gap, perfect ground truth, any poses.
    + InteriorNet, Unreal hyperrealism look, persp. depth, access request.
    + Replica, looks real, small apartment scale, pinhole/orthographic only.
  + Real world, layout and look, inaccurate poses+depth, evaluation hard.
    + OmniScenes, Ricoh Theta + Matterport, SfM methods, COLMAP depth, manual postprocessing (alignment + filtering).
    + 360Wild, internet videos, OpenSfM poses, COLMAP depth cube maps.
    + Matterport3D, LiDAR, access request, found different Matterport-based dataset.
]

#align(center + horizon)[
  #grid(
    align: center,
    columns: 3,
    rows: 2,
    gutter: 1em,
    grid.cell(
      colspan: 3,
      stack(
        dir: ltr,
        spacing: 1em,
        [#image("media/interior-net.png", height: 34%) InteriorNet],
        [#image("media/replica.png", height: 34%) Replica],
      )
    ),
    [#image("media/omni-scenes.png", height: 34%) OmniScenes],
    [#image("media/360-wild.png", height: 34%) 360Wild],
    [#image("media/matterport-dataset.png", height: 34%) Matterport3D],
  )
]


=== Stanford 2D-3D
#slide()[
  #align(center, image("media/stanford2d3d-areas.png", width: 85%))

  - 1413 equirectangular frames from 272 capture sessions over 6 areas
  - Matterport camera: rotating RGB camera + structured light
  - Stitched perspective captures + semantic labels
]

#speaker-note[
  + Great RGB camera and depth quality
  + Large scale and great coverage of rooms
  + The black areas on top are due to stitching perspective images together
  + Depth are from 3D reconstructed meshes, which explains artifacts
]
#slide()[
  #stack(
    dir: ltr,
    spacing: 0.5em,
    [#image("media/stanford2d3d-rgba.png", width: 50%)],
    [#image("media/stanford2d3d-depth.png", width: 50%)]
  )
]

#speaker-note[
  + Initially tried this dataset.
  + Random looking predictions from VGGT as if it has no clue.
  + Capture sessions usually have too few frames.
  + Trajectories have information that can be exploited.
  + Such as relatively smooth transition between pose translation and rotation.
  + Models may be trained with a bias towards some specific distance interval frames.
]
#slide()[
  #image("media/stanford2d3d-distribution.png")
  - Capture sessions too sparse, 6 frames or below for 75% 
  - No order to frames $->$ No trajectory information to exploit
  - Arbitrary rotation and translation magnitude between frames
]


=== 360Loc
#speaker-note[
  + 360 camera + LiDAR scanner rig.
  + Point cloud: ICP and BA of scans.
  + RGB: Ransac (LiDAR to camera) and manual correction.
]
#image("media/360loc-pipeline.png")


#slide(align: center)[
  #grid(
    align: center,
    columns: 3,
    rows: 2,
    gutter: 1em,
    image("media/360loc-atrium-overview-1.png", height: 40%),
    image("media/360loc-atrium-overview-2.png", height: 40%),
    image("media/360loc-atrium-overview-3.png", height: 40%),
    grid.cell(
      colspan: 3,
      stack(
        dir: ltr,
        spacing: 0.5em,
        align(left, [
          - High resolution and pretty images
          - 4 scenes, multiple trajectories
          - Cited 27 times (Google Scholar)
          - Approx. 2 FPS and large areas
          - Subsampled above visualizations
          - Person in RGB images, not depth
        ]),
        align(right, image("media/360loc-atrium-overview-4.png", height: 50%))
      )
    )
    
  )
]

#slide[
  - What RGB camera was used? Was not specified.
  - RGB camera did not have locked settings.
  - Depth images only exist for one trajectory per scene.
  - Dataset is hosted on OneDrive, slow download, failing downloads.
  - Depth and pose use different units, and no units are provided.
  - No dataset loading code, only official download and extracting code.

  #stack(
    dir: ltr,
    spacing: 1.0em,
    align(center, image("media/360loc-atrium-underexposed.png", height: 40%)),
    align(center, image("media/360loc-atrium-scale.png", height: 40%))
  )
  
]

#speaker-note[
  + Easy to debug, when the issue is known.
  + Really hard when it could have been somewhere in your own code.
  + Especially after the datasheet numbers for the scanner do not work.
  + Found red container in one of the pictures.
  + Calibrated to real life size, since containers are standardized.
]
#slide()[
  #align(center)[
    #stack(
      dir: ltr,
      spacing: 0.5em,
      image("media/360loc-hall-red-container-rgb.png", height: 60%),
      image("media/360loc-hall-red-container-point.png", height: 60%),
    )
    Pose unit is likely meters - depth is likely centimeters
  ]
]

#speaker-note[
  + Done my open source duty of reporting findings.
  + Contributers seem to have abandoned the repository.
]
#slide(align: center)[
  #image("media/360loc-github.png")
]



= Pipeline
Some inspirational quote \ - Someone

#focus-slide[
  What is the reconstruction quality of various
  Gaussian splatting initialization methods for 360° captures?
]

#speaker-note[
  + Dataset has been found.
  + Next step is not the models.
  + If cannot see what doing, then not doing anything.
  + Need good visualization and metric tooling.
    + 3D visualization of dataset reconstruction, poses, keypoints.
    + Quantitative metrics for pose estimation quality.
  + Helps greatly with guiding development.
]
#[
  #set align(center)
  #set text(size: 0.7em)
  #diagram(
    edge-stroke: 0.1em,
    node-corner-radius: 0.1em,
    mark-scale: 60%,
    spacing: 2em,
    node-inset: 0.6em,
    node-stroke: 0.1em,

    node((-1, 0), text(white)[360Loc\ dataset], name: <n0>, fill: color.orange),
    node((0.5, 0), [Pose+keypoint\ estimation], name: <n1>),
    node((1.5, -1.5), text(white)[Visualization tools], name: <n4>, fill: color.blue),
    node((2.5, 0), [Gaussian splat\ training], name: <n2>),
    node((4, 0), [Gaussian splat], name: <n3>),

    edge(<n0>, <n1>, "->"),
    edge(<n1>, <n2>, "->"),
    edge(<n2>, <n3>, "->"),
    edge(<n4>, <n1>, "->", corner: left),
    edge(<n4>, <n2>, "->", corner: right),
  )
]

== Pose evaluation

#speaker-note[
  + Incorporation of poses into the alignment process may be useful.
  + But at the same time, the model is supposed to do a good job, so should not be necessary.
  + Cannot continue with bad poses, because unable to align for metrics.
]
#slide(align: horizon, composer: (1fr, 1.4fr))[
  - Equirectangular sequence
  - Predict poses and keypoints
  - Procrustes analysis only finds parameters $R, t$, and $s$
  - Better alignment somehow by orientation of poses?
  - Train, okay, wonk, and fails
][
  #image("media/360loc-atrium-pose-alignment.png")
]


=== Metrics
#speaker-note[
  + Storing mean and std. of all pose metrics.
  + All metrics are computed after alignment.
  + People seem to usually use ATE (RMSE after alignment) and RPE (relative pose error).
  + ATE similar to translation error here, but no squaring and roots here.
  + RTE measures drift, but this was unfortunately not implemented.
  + Not big brain enough to intuitively understand geodesic distance on SO(3).
  + Decompose into two more intuitive metrics: pointing and roll.
]
#table(
  columns: (auto, 1fr),
  inset: 0.5em,
  align: horizon,
  table.header([*Metric*], [*Description*]),
  "Translation [m]", "Pose position error in meters.",
  "Rotation geodesic [rad]", "Shortest distance error on SO(3) manifold",
  "Rotation pointing [rad]", "Angle between forward pointing vectors",
  "Rotation roll [rad]", "Roll error after alignment of forward vectors",
  "Prediction time [s]", "Time taken to predict all poses",
)


=== Useful design decisions
- Set up project inside a container - reproducibility and easier debugging.
- Choose an old but popular CUDA toolkit version for compatibility.
- Use `uv` for Python package management, and maybe `mamba` for others.
- Modifications of git submodules requires you to host your own fork.
#pause
- Model implementation chooses when and what subsets of scenes to load.
- Dataset implementation should return lazy partial scene loader.
- Measure data loading performance of threads to maximize SSD usage.
- Break training data leak - look at subset of scenes during development.
#pause
- Procrustes analysis forces sequence length to be much greater than 3.
- VGGT defines first frame as reference pose. This convention works well.
- Rerun works for quick visualizations, but it is memory hungry.


== Gaussian splat evaluation
#[
  #set align(center)
  #set text(size: 0.7em)
  #diagram(
    edge-stroke: 0.1em,
    node-corner-radius: 0.1em,
    mark-scale: 60%,
    spacing: 2em,
    node-inset: 0.6em,
    node-stroke: 0.1em,

    node((-1, 0), text(white)[360Loc\ dataset], name: <n0>, fill: color.orange),
    node((0.5, 0), text(white)[Pose+keypoint\ estimation], name: <n1>, fill: color.orange),
    node((1.5, -1.5), text(white)[Visualization tools], name: <n4>, fill: color.orange),
    node((2.5, 0), text(white)[Gaussian splat\ training], name: <n2>, fill: color.blue),
    node((4, 0), [Gaussian splat], name: <n3>),

    edge(<n0>, <n1>, "->"),
    edge(<n1>, <n2>, "->"),
    edge(<n2>, <n3>, "->"),
    edge(<n4>, <n1>, "->", corner: left),
    edge(<n4>, <n2>, "->", corner: right),
  )
]


=== 3DGS vs. gsplat
#speaker-note[
  + 3DGS is almost 3 years old, ancient technique, and too slow.
  + gsplat is an old and stable community project with support for multiple techniques.
  + Unfortunatly, dependency issues with one of the models and no equirectangular support.
]
#slide(align: center + horizon, composer: (4fr, 1fr, 4fr))[
  #rotate(-5deg, box(image("media/3dgs-paper.png", height: 70%), stroke: color.black))
][
  vs.
][
  #image("media/gsplat-logo.png")
]


=== On-the-fly NVS
#grid(
  columns: (1fr, 1.1fr),
  rows: 2,
  gutter: 1em,
  [
    - 2/5 authors from 3DGS paper
    - SIGGRAPH 2025 (July)
    - Order of magnitude faster
    - Competitive results
    - Large scale scenes (anchors)
    - Optimizes poses (init + finetune)
    
  ],
  align(right, image("media/otf-nvs-atrium.png")),
  grid.cell(
    colspan: 2,
    align: center + horizon,
    [
      #v(1.6em)
      #text(size: 1.9em)[Too few Gaussians, 360Loc is too large]
    ]
  )
)


=== LichtField Studio
#slide(composer: (2fr, 1.1fr))[
  - Supports NVIDIA 3DGUT
  - CLI access to training parameters
  - Supports equirectangular images natively

  - Modifications necessary for integration
  - Can only evaluate every n'th image
  - Buggy rendering engine
][
  #image("media/lichtfeld-ui.png")
]

#slide(align: center)[
  #image("media/lichtfeld-github-flip.png", height: 80%)
  1 day after wasting hours debugging
]


=== Metrics
#speaker-note[
  + Doing experiments at different dataset strides.
    + Higher means images are further apart.
    + More challenging for reconstruction.
    + But much less compute intensive.
  + Using cube maps because the metrics are designed for perspective images.
    + E.g. LPIPS uses a neural network trained on perspective images.
  + Validation images are from ground truth poses inbetween training poses.
    + If a predicted pose is wildly off, then not interpolating to find "middle" pose.
    + Walking speed is non-constant, and could result in 20cm pose error.
    + We want good reconstruction quality, the metrics are merely proxy's of this.
]
#[
  #set align(center)
  #set text(size: 1.0em)
  #diagram(
    edge-stroke: 0.1em,
    node-corner-radius: 0.1em,
    mark-scale: 60%,
    spacing: 2em,
    node-inset: 0.6em,
    node-stroke: 0.1em,

    node((0, 0), text(white)[1], name: <n0>, fill: blue),
    node((1, 0), [2], name: <n1>),
    node((2, 0), text(white)[3], name: <n2>, fill: orange),
    node((3, 0), [4], name: <n3>),
    node((4, 0), text(white)[5], name: <n4>, fill: blue),
    node((5, 0), [6], name: <n5>),
    node((6, 0), text(white)[7], name: <n6>, fill: orange),
    node((7, 0), [8], name: <n7>),
    node((8, 0), text(white)[9], name: <n8>, fill: blue),

    edge(<n0>, <n1>, "->"),
    edge(<n1>, <n2>, "->"),
    edge(<n2>, <n3>, "->"),
    edge(<n3>, <n4>, "->"),
    edge(<n4>, <n5>, "->"),
    edge(<n5>, <n6>, "->"),
    edge(<n6>, <n7>, "->"),
    edge(<n7>, <n8>, "->")
  )
]
#v(2em)
#[
  #set align(center)
  *PSNR*, *SSIM*, *LPIPS* of cube map of frame inbetween training frames\
  +\
  *qualitative accessment* of renders from arbitrary poses
]


---

#slide(align: center + horizon)[
  Model $times$ Dataset stride $times$ Scene $= 96$ experiments

  #table(
    columns: (1fr, 1fr, 1fr),
    inset: 0.5em,
    align: horizon + center,
    table.header([*Model*], [*Dataset stride*], [*Scene*]),
    "GT, COLMAP, VGGT(N), VGGT(P), DA3(S), ViPE",
    "2, 4, 8, 16",
    "Atrium, Concourse, Hall, Piatrium"
  )
]


== Hardware setup
#speaker-note[
  + Pretty beefy personal computer, but still quite compute constrained.
  + Recommend to use someone else's computer.
  + Miserable noise and heat level. Case is even bent now.
  + GPU is glitching out now, last time saw something like this my GPU died a few weeks later.
]
#stack(
  dir: ltr,
  [
    - #text("CPU", font: "DejaVu Sans Mono") Ryzen 9 5950X, 16x 4.6GHz

    - #text("GPU", font: "DejaVu Sans Mono") RTX 3090, 24GB GDDR6X
    - #text("RAM", font: "DejaVu Sans Mono") DDR4, 64GB, 3600MT/s CL16
    - #text("SSD", font: "DejaVu Sans Mono") FireCuda 530, NVMe 1.4, 2TB
  ],
  align(right)[#image("media/gpu-glitching.jpg", width: 50%)]
)



= Models
---
+ Ground truth

+ COLMAP
+ Meta Visual Geometry Grounded Transformer
+ ByteDance Depth Anything 3 Streaming
+ NVIDIA Video Pose Engine


== Ground truth
#speaker-note[
  + Alternative baseline.
    + Ground truth poses and projected depth maps.
    + COLMAP or proprietary software usually used.
    + But they may fail, so using ground truth as baseline.
  + Uniform sampling of equirectangular images.
    + Oversamples bottom and top.
    + Visible as concentrated spots around the poses.
  + Sampling cube map without bottom face probably better.
    + Capture hardware and hand at bottom.
    + Too late because training too slow.

]
#slide(composer: (1fr, 1.4fr))[
  - Poses and keypoints

  - Alternative baseline
  - 0.1% projected depth maps
  - Uniform sampling issues
  - Cubemap without bottom?
][
  #image("media/360loc-atrium-ground-truth.png")
]


== COLMAP
#slide(composer: (1fr, 1fr))[
  - Widely known and used
  - Poses and sparse keypoints
  - Classical: feature extraction, matching, registration, BA
  - Not GPU memory hungry

  - No use of trajectory information
  - Requires perspective images
][ 
  #set align(right)
  #image("media/360loc-atrium-face-frame.jpg")
  #stack(
    dir: ltr,
    spacing: 0.1em,
    image("media/360loc-atrium-face-3.png", width: 24.55%),
    image("media/360loc-atrium-face-1.png", width: 24.55%),
    image("media/360loc-atrium-face-2.png", width: 24.55%),
    image("media/360loc-atrium-face-0.png", width: 24.55%),
  )
]

=== Two problems
#slide(align: center + horizon, composer: (1fr, 1fr))[
  #text("1", size: 3em)\
  How does one convert equirectangular to perspective?
][
  #text("2", size: 3em)\
  How does one merge\ predicted poses and keypoints?
]

=== Equirectangular to perspective
#slide(align: center)[
  #image("media/google-eac-equirectangular.webp")
  Equirectangular projection disproportionately allocates pixels near poles
]

#slide(align: center + horizon)[
  #image("media/google-eac-cubemap.webp")
  *All sphere to plane projections are distorted*
]

=== Optimized Tangens Cubemap
#speaker-note[
  + A lot of litterature focuses on making all pixels tied to an equal area on the sphere.
  + Therefore each pixel has closer to an equal amount of information.
  + The data storage efficiency is also higher.
  + The hope was that this was a better format than equirectangular images.
  + However, in practice leads to worse prediction performance.
  + We only cared about prediction performance, not data storage efficiency.
  + Straight lines are not straight - not perspective image.
  + Wasted a lot of time investigating this.
]
#image("media/stanford2d3d-otc.png")

=== Cubemap
#speaker-note[
  + Bottom face, capture hardware and hand is in the way.
  + Top face, outside clouds.
    + Pose translation, litle cloud movement, parallax.
    + Wind, so clouds still move.
  + Cuts 33% of computation cost.
  + Keypoints can basically only be placed in half the equirectangular image.
]
#slide(align: center + horizon)[
  #alternatives()[
    #image("media/360loc-atrium-face-frame.jpg")
  ][
    #stack(
      dir: ltr,
      spacing: 0.1em,
      image("media/360loc-atrium-face-3.png", width: 24.3%),
      image("media/360loc-atrium-face-1.png", width: 24.3%),
      image("media/360loc-atrium-face-2.png", width: 24.3%),
      image("media/360loc-atrium-face-0.png", width: 24.3%),
    )
  ][
    #image("media/360loc-atrium-depth.png")
  ]
]

=== Cubemap, a mistake?
#slide(align: center + horizon)[
  4x the batch size and no overlap between faces\ \
  #pause
  *Perhaps fewer faces with higher FoV?*
]


=== Merging predictions
- Multiple poses and keypoints for each frame

- COLMAP treats each face independently
- Rotate faces to face forwards, then
#pause
- Mean translation, #alternatives[mean rotation][*mean rotation*], SE(3) transform keypoints

#pause
#pause
- #alternatives[NASA mathemagics?][Quarternion Averaging (Markley et al. 2007)]


== Meta Visual Geometry Grounded Transformer (VGGT)
#align(center)[#image("media/vggt-pipeline.png", height: 60%)]
- Visual Geometry Grounded Transformer, released March 2025
- Trained on perspective images
- Camera tokens are identical for non-first images (no trajectory)
- Very memory hungry model

=== Sequence chunking
#speaker-note[

]
#[
  #set align(center)
  #set text(size: 1.0em)
  #diagram(
    edge-stroke: 0.1em,
    node-corner-radius: 0.1em,
    mark-scale: 60%,
    spacing: 2em,
    node-inset: 0.6em,
    node-stroke: 0.1em,

    node((0, 0), text(white)[1], name: <n0>, fill: blue),
    node((1, 0), text(white)[2], name: <n1>, fill: blue),
    node((2, 0), text(white)[3], name: <n2>, fill: blue),
    node((3, 0), text(white)[4], name: <n3>, fill: blue),
    node((4, 0), text(white)[5], name: <n4>, fill: blue),
    node((5, 0), [6], name: <n5>),
    node((6, 0), [7], name: <n6>),
    node((7, 0), [8], name: <n7>),

    edge(<n0>, <n1>, "->"),
    edge(<n1>, <n2>, "->"),
    edge(<n2>, <n3>, "->"),
    edge(<n3>, <n4>, "->"),
    edge(<n4>, <n5>, "->"),
    edge(<n5>, <n6>, "->"),
    edge(<n6>, <n7>, "->")
  )
]
#[
  #set align(center)
  #set text(size: 1.0em)
  #diagram(
    edge-stroke: 0.1em,
    node-corner-radius: 0.1em,
    mark-scale: 60%,
    spacing: 2em,
    node-inset: 0.6em,
    node-stroke: 0.1em,

    node((0, 0), [1], name: <n0>),
    node((1, 0), [2], name: <n1>),
    node((2, 0), text(white)[3], name: <n2>, fill: blue),
    node((3, 0), text(white)[4], name: <n3>, fill: blue),
    node((4, 0), text(white)[5], name: <n4>, fill: blue),
    node((5, 0), text(white)[6], name: <n5>, fill: blue),
    node((6, 0), text(white)[7], name: <n6>, fill: blue),
    node((7, 0), [8], name: <n7>),

    edge(<n0>, <n1>, "->"),
    edge(<n1>, <n2>, "->"),
    edge(<n2>, <n3>, "->"),
    edge(<n3>, <n4>, "->"),
    edge(<n4>, <n5>, "->"),
    edge(<n5>, <n6>, "->"),
    edge(<n6>, <n7>, "->")
  )
]
#[
  #set align(center)
  #set text(size: 1.0em)
  #diagram(
    edge-stroke: 0.1em,
    node-corner-radius: 0.1em,
    mark-scale: 60%,
    spacing: 2em,
    node-inset: 0.6em,
    node-stroke: 0.1em,

    node((0, 0), [1], name: <n0>),
    node((1, 0), [2], name: <n1>),
    node((2, 0), [3], name: <n2>),
    node((3, 0), [4], name: <n3>),
    node((4, 0), text(white)[5], name: <n4>, fill: blue),
    node((5, 0), text(white)[6], name: <n5>, fill: blue),
    node((6, 0), text(white)[7], name: <n6>, fill: blue),
    node((7, 0), text(white)[8], name: <n7>, fill: blue),

    edge(<n0>, <n1>, "->"),
    edge(<n1>, <n2>, "->"),
    edge(<n2>, <n3>, "->"),
    edge(<n3>, <n4>, "->"),
    edge(<n4>, <n5>, "->"),
    edge(<n5>, <n6>, "->"),
    edge(<n6>, <n7>, "->")
  )
]
- Arbitrary sequence length with controllable memory usage
- Cannot attend to entire sequence
- Chunks must be merged via overlap regions
- Pose merging can be done as before
- Keypoint merging: different pixels $->$ append; same pixel $->$ 3D average

=== Naive vs. perspective-based implementation
#slide(align: horizon, composer: (1fr, 1fr))[
  *Naive*

  - Neural network $->$ Miracle $->$ Can ingest equirectangular images

  - No merging of faces necessary
  - Depth point cloud bent outwards
][
  *Perspective transform*
  - Cubemap transform to stay similar to training data

  - Requires merging of faces
  - Much more compute intensive
]


== ByteDance Depth Anything Streaming (DA3)
#speaker-note[
  + Claims DA3 has better performance than VGGT
  + Chunk size 60 is where they claim their performance becomes good.
  + Diminishing returns for higher chunk sizes.
  + Their model seems to use more memory than they report.
  + Subsampling depth maps, because streaming variant had implementation issues in the way it exposed its keypoints.
]
#v(-1.0em)
#align(center)[#image("media/da3-pipeline.png", height: 50%)]
#v(-1.3em)
- Streaming variant of Depth Anything 3, released December 11th 2025
- Competent chunking: loop closure, and keypoint/depth alignment
#pause
- Performance is very sensitive to image resolution and chunk size
- Unable to replicate their memory usage results
#pause
- Also requires perspective images
- However, additionally uses trajectory information

=== Order of perspective images
#speaker-note[
  + Each frame is tied to multiple faces.
  + Which face comes first in the sequence?
  + Trying to avoid rotation as it usually recommended against in SfM pipelines.
  + 4 separate runs also reduces the sequence length fed to the model.
]
#[
  #set align(center)
  #set text(size: 1.0em)
  #diagram(
    edge-stroke: 0.1em,
    node-corner-radius: 0.1em,
    mark-scale: 60%,
    spacing: 2em,
    node-inset: 0.6em,
    node-stroke: 0.1em,

    node((0, 0), [1], name: <n0>),
    node((1, 0), [2], name: <n1>),
    node((2, 0), [3], name: <n2>),
    node((3, 0), [4], name: <n3>),
    node((4, 0), [5], name: <n4>),

    edge(<n0>, <n1>, "->"),
    edge(<n1>, <n2>, "->"),
    edge(<n2>, <n3>, "->"),
    edge(<n3>, <n4>, "->")
  )
]
#align(center)[$arrow.b$]
#[
  #set align(center)
  #set text(size: 0.7em)
  #diagram(
    edge-stroke: 0.1em,
    node-corner-radius: 0.1em,
    mark-scale: 60%,
    spacing: 2em,
    node-inset: 0.6em,
    node-stroke: 0.1em,

    node((0, 0), text(white)[1.front], name: <f0>, width: 4em, fill: color.blue),
    node((1, 0), [2.front], name: <f1>, width: 4em),
    node((2, 0), [3.front], name: <f2>, width: 4em),
    node((3, 0), [4.front], name: <f3>, width: 4em),
    node((4, 0), text(white)[5.front], name: <f4>, width: 4em, fill: color.orange),
    edge(<f0>, <f1>, "->"),
    edge(<f1>, <f2>, "->"),
    edge(<f2>, <f3>, "->"),
    edge(<f3>, <f4>, "->"),

    edge(<f4>, <l4>, "->"),

    node((0, 0.8), text(white)[1.left], name: <l0>, width: 4em, fill: color.orange),
    node((1, 0.8), [2.left], name: <l1>, width: 4em),
    node((2, 0.8), [3.left], name: <l2>, width: 4em),
    node((3, 0.8), [4.left], name: <l3>, width: 4em),
    node((4, 0.8), text(white)[5.left], name: <l4>, width: 4em, fill: color.blue),
    edge(<l1>, <l0>, "->"),
    edge(<l2>, <l1>, "->"),
    edge(<l3>, <l2>, "->"),
    edge(<l4>, <l3>, "->"),

    edge(<l0>, <b0>, "->"),

    node((0, 1.65), text(white)[1.back], name: <b0>, width: 4em, fill: color.blue),
    node((1, 1.65), [2.back], name: <b1>, width: 4em),
    node((2, 1.65), [3.back], name: <b2>, width: 4em),
    node((3, 1.65), [4.back], name: <b3>, width: 4em),
    node((4, 1.65), text(white)[5.back], name: <b4>, width: 4em, fill: color.orange),
    edge(<b0>, <b1>, "->"),
    edge(<b1>, <b2>, "->"),
    edge(<b2>, <b3>, "->"),
    edge(<b3>, <b4>, "->"),

    edge(<b4>, <r4>, "->"),

    node((0, 2.55), text(white)[1.right], name: <r0>, width: 4em, fill: color.orange),
    node((1, 2.55), [2.right], name: <r1>, width: 4em),
    node((2, 2.55), [3.right], name: <r2>, width: 4em),
    node((3, 2.55), [4.right], name: <r3>, width: 4em),
    node((4, 2.55), text(white)[5.right], name: <r4>, width: 4em, fill: color.blue),
    edge(<r1>, <r0>, "->"),
    edge(<r2>, <r1>, "->"),
    edge(<r3>, <r2>, "->"),
    edge(<r4>, <r3>, "->"),
  )
]

---

#[
  #set align(center)
  #set text(size: 1.0em)
  #diagram(
    edge-stroke: 0.1em,
    node-corner-radius: 0.1em,
    mark-scale: 60%,
    spacing: 2em,
    node-inset: 0.6em,
    node-stroke: 0.1em,

    node((0, 0), [1], name: <n0>),
    node((1, 0), [2], name: <n1>),
    node((2, 0), [3], name: <n2>),
    node((3, 0), [4], name: <n3>),
    node((4, 0), [5], name: <n4>),

    edge(<n0>, <n1>, "->"),
    edge(<n1>, <n2>, "->"),
    edge(<n2>, <n3>, "->"),
    edge(<n3>, <n4>, "->")
  )
]
#align(center)[$arrow.b$]
#[
  #set align(center)
  #set text(size: 0.7em)
  #diagram(
    edge-stroke: 0.1em,
    node-corner-radius: 0.1em,
    mark-scale: 60%,
    spacing: 2em,
    node-inset: 0.6em,
    node-stroke: 0.1em,

    node((0, 0), text(white)[1.front], name: <f0>, width: 4em, fill: color.blue),
    node((1, 0), [2.front], name: <f1>, width: 4em),
    node((2, 0), [3.front], name: <f2>, width: 4em),
    node((3, 0), [4.front], name: <f3>, width: 4em),
    node((4, 0), text(white)[5.front], name: <f4>, width: 4em, fill: color.orange),
    edge(<f0>, <f1>, "->"),
    edge(<f1>, <f2>, "->"),
    edge(<f2>, <f3>, "->"),
    edge(<f3>, <f4>, "->"),

    node((0, 0.8), text(white)[1.left], name: <l0>, width: 4em, fill: color.blue),
    node((1, 0.8), [2.left], name: <l1>, width: 4em),
    node((2, 0.8), [3.left], name: <l2>, width: 4em),
    node((3, 0.8), [4.left], name: <l3>, width: 4em),
    node((4, 0.8), text(white)[5.left], name: <l4>, width: 4em, fill: color.orange),
    edge(<l0>, <l1>, "->"),
    edge(<l1>, <l2>, "->"),
    edge(<l2>, <l3>, "->"),
    edge(<l3>, <l4>, "->"),

    node((0, 1.65), text(white)[1.back], name: <b0>, width: 4em, fill: color.blue),
    node((1, 1.65), [2.back], name: <b1>, width: 4em),
    node((2, 1.65), [3.back], name: <b2>, width: 4em),
    node((3, 1.65), [4.back], name: <b3>, width: 4em),
    node((4, 1.65), text(white)[5.back], name: <b4>, width: 4em, fill: color.orange),
    edge(<b0>, <b1>, "->"),
    edge(<b1>, <b2>, "->"),
    edge(<b2>, <b3>, "->"),
    edge(<b3>, <b4>, "->"),

    node((0, 2.55), text(white)[1.right], name: <r0>, width: 4em, fill: color.blue),
    node((1, 2.55), [2.right], name: <r1>, width: 4em),
    node((2, 2.55), [3.right], name: <r2>, width: 4em),
    node((3, 2.55), [4.right], name: <r3>, width: 4em),
    node((4, 2.55), text(white)[5.right], name: <r4>, width: 4em, fill: color.orange),
    edge(<r0>, <r1>, "->"),
    edge(<r1>, <r2>, "->"),
    edge(<r2>, <r3>, "->"),
    edge(<r3>, <r4>, "->"),
  )
]

== NVIDIA Video Pose Engine (ViPE)
#align(center)[#image("media/vipe-pipeline.png", height: 60%)]
- Released August 11th 2025
- Creates camera rig based off of equirectangular image
- Identifies moving objects via predicted segmentation mask
- Streaming, but has memory usage spike - needs sequence chunking



== Common themes
- They test their models on large GPUs and claim low memory usage
- Small memory spikes kill their models on small GPUs
- OpenAI Codex very useful to quickly dig into their code to find issues
#v(0.4em)
- I did not perform an exhaustive parameter search
- Spent hours finding decent settings within hardware limits.



= Results

== Overview
#[
  #let abox(color) = box(fill: color, width: 0.8em, height: 0.8em, radius: 0.15em, stroke: black + 0.05em)
  #let tbox = abox(green)
  #let obox = abox(yellow)
  #let wbox = abox(red)
  #let fbox = abox(black)

  #table(
    columns: (1.4fr, 1fr, 1fr, 1fr, 1fr),
    inset: 0.4em,
    align: center + horizon,
    stroke: (x, y) => if y == 1 {
      (bottom: 0.05em + black)
    } else if x == 0 and y > 1 {
      (right: 0.05em + black)
    },
    align(right, text(size: 0.7em)[Scene name $->$]),
    table.cell(
      colspan: 4,
      "Atrium, Concourse, Hall, Piatrium"
    ),
    align(right, text(size: 0.7em)[Dataset stride $->$]), "16", "8", "4", "2",

    "COLMAP", [#fbox #fbox #fbox #fbox], [#tbox #fbox #fbox #fbox], [#tbox #wbox #fbox #fbox], [#tbox #fbox #tbox #fbox],
    "VGGT naive", [#fbox #fbox #fbox #fbox], [#fbox #fbox #fbox #fbox], [#fbox #fbox #fbox #fbox], [#fbox #fbox #fbox #fbox],
    "VGGT cube map", [#obox #fbox #fbox #wbox], [#obox #fbox #fbox #wbox], [#wbox #fbox #wbox #wbox], [#wbox #fbox #wbox #wbox],
    "DA3 streaming", [#wbox #fbox #fbox #wbox], [#fbox #fbox #fbox #obox], [#obox #fbox #wbox #fbox], [#fbox #fbox #obox #fbox],
    "ViPE", [#tbox #fbox #fbox #obox], [#tbox #tbox #tbox #tbox], [#tbox #tbox #tbox #tbox], [#tbox #tbox #tbox #tbox],
  )

]

---
#align(center)[#image("media/results-pose-class.png")]
---
#align(center)[#image("media/results-fps.png")]

=== Fail
#align(center)[
  VGGT naive, 15.29m $plus.minus$ 7.04; 178.52$degree$ $plus.minus$ 1.31
  #image("media/results-fail-0-naive-8.png", height: 80%)
]
---
#align(center)[
  COLMAP, 17.51m $plus.minus$ 5.80; 29.95$degree$ $plus.minus$ 20.19
  #image("media/results-fail-1-colmap-8.png", height: 80%)
]
---
#align(center)[
  DA3 streaming, 22.91m $plus.minus$ 10.32; 98.34$degree$ $plus.minus$ 36.28
  #image("media/results-fail-2-da3-8.png", height: 80%)
]
---
#align(center)[
  Point cloud
  #image("media/results-fail-2-da3-8-point-cloud.png", height: 80%)
]
#align(center)[
  Overview
  #image("media/results-fail-2-da3-8-training-1.png", height: 80%)
]
#align(center)[
  Near multiple poses
  #image("media/results-fail-2-da3-8-training-2.png", height: 80%)
]
#align(center)[
  Overview
  #image("media/results-fail-2-da3-8-training-3.png", height: 80%)
]
#align(center)[
  Near a pose
  #image("media/results-fail-2-da3-8-training-4.png", height: 80%)
]
#align(center)[
  Overview
  #image("media/results-fail-2-da3-8-training-5.png", height: 80%)
]
---
- Poses are all messed up
- Keypoint cloud is all messed up

- Geometry of reconstruction will be all messed up
- Soup of Gaussians, each pose surrounds itself with Gaussians


=== Wonk
#align(center)[
  VGGT cube map, 16.82 $plus.minus$ 10.06; 70.29$degree$ $plus.minus$ 75.27
  #image("media/results-wonk-3-cube-8.png", height: 80%)
]
---
#align(center)[
  VGGT cube map, 16.82 $plus.minus$ 10.06; 70.29$degree$ $plus.minus$ 75.27
  #image("media/results-wonk-3-cube-8-point-cloud.png", height: 80%)
]
---
#align(center)[
  Overview
  #image("media/results-wonk-3-cube-8-training-1.png", height: 80%)
]
---
#align(center)[
  Near multiple poses
  #image("media/results-wonk-3-cube-8-training-2.png", height: 80%)
]
---
#align(center)[
  Near multiple poses
  #image("media/results-wonk-3-cube-8-training-3.png", height: 80%)
]
---
#align(center)[
  Near multiple poses
  #image("media/results-wonk-3-cube-8-training-4.png", height: 80%)
]
---
#align(center)[
  Near some other poses
  #image("media/results-wonk-3-cube-8-training-5.png", height: 80%)
]
---
#align(center)[
  Near some other poses
  #image("media/results-wonk-3-cube-8-training-6.png", height: 80%)
]
---
#align(center)[
  Overview
  #image("media/results-wonk-3-cube-8-training-7.png", height: 80%)
]
---
- Poses clearly captured something but overall did badly
- Keypoint cloud kind of locally makes sense

- Reconstruction has some 3D structure initially
- Diverges later in training to a Gaussian soup

=== Okay
#align(center)[
  ViPE, 8.01m $plus.minus$ 7.00; 12.49$degree$ $plus.minus$ 15.13
  #image("media/results-wonk-3-vipe-16.png", height: 80%)
]
---
#align(center)[
  ViPE, 8.01m $plus.minus$ 7.00; 12.49$degree$ $plus.minus$ 15.13
  #image("media/results-wonk-3-vipe-16-point-cloud.png", height: 80%)
]
---
#align(center)[
  VGGT cube map, 6.22m $plus.minus$ 4.85; 33.64$degree$ $plus.minus$ 55.73
  #image("media/results-okay-0-cube-8.png", height: 80%)
]
---
#align(center)[
  VGGT cube map, 6.22m $plus.minus$ 4.85; 33.64$degree$ $plus.minus$ 55.73
  #image("media/results-okay-0-cube-8-point-cloud.png", height: 80%)
]

---

#align(center)[
  Validation pose
  #image("media/results-okay-0-cube-8-validation-1.png", height: 50%)
]
---
#align(center)[
  Validation pose
  #image("media/results-okay-0-cube-8-validation-2.png", height: 50%)
]
---
#align(center)[
  Validation pose
  #image("media/results-okay-0-cube-8-validation-3.png", height: 50%)
]
---
#align(center)[
  Validation pose
  #image("media/results-okay-0-cube-8-validation-4.png", height: 50%)
]

---

#align(center)[
  Arbitrary pose
  #image("media/results-okay-0-cube-8-arbitrary-1.png", height: 80%)
]
---
#align(center)[
  Arbitrary pose
  #image("media/results-okay-0-cube-8-arbitrary-2.png", height: 80%)
]

---
- Poses are inaccurate but generally captured the trajectory
- Keypoint cloud appears to overall have structure

- Validation frames are in between training frames
- Interpolation vs. reconstruction


=== Train
#align(center)[
  COLMAP, 1.18m $plus.minus$ 1.81; 4.27$degree$ $plus.minus$ 9.48
  #image("media/results-train-0-colmap-8.png", height: 80%)
]
---
#align(center)[
  ViPE, 0.77m $plus.minus$ 0.61; 0.93$degree$ $plus.minus$ 0.50
  #image("media/results-train-3-vipe-8.png", height: 80%)
]
---
#align(center)[
  Validation pose
  #image("media/results-train-3-vipe-8-validation-1.png", height: 50%)
]
---
#align(center)[
  Validation pose
  #image("media/results-train-3-vipe-8-validation-2.png", height: 50%)
]
---
#align(center)[
  Validation pose
  #image("media/results-train-3-vipe-8-validation-3.png", height: 50%)
]
---
#align(center)[
  Arbitrary pose
  #image("media/results-train-3-vipe-8-arbitrary-1.png", height: 80%)
]
---
#align(center)[
  Arbitrary pose
  #image("media/results-train-3-vipe-8-arbitrary-2.png", height: 80%)
]
---
- Poses are starting to be accurate
- Keypoint cloud captures structure

- Same reconstruction issues as okay, but less severe
- Potentially undertrained and undertuned settings


=== Metrics

- Metrics for all pose success categories

- Metrics for okay and train
- Metrics for only train

---
#align(center)[
  #image("media/results-translation-all.png")
]
---
#align(center)[
  #image("media/results-geodesic-all.png")
]

---

#align(center)[
  #image("media/results-translation-okay-train.png")
]
---
#align(center)[
  #image("media/results-geodesic-okay-train.png")
]

---

#align(center)[
  #image("media/results-translation-train.png")
]
---
#align(center)[
  #image("media/results-pointing.png")
]
---
#align(center)[
  #image("media/results-roll.png")
]
---
#align(center)[
  #image("media/results-psnr.png")
]
---
#align(center)[
  #image("media/results-ssim.png")
]
---
#align(center)[
  #image("media/results-lpips.png")
]


== Interesting points
- Completely out of time, here's some keywords
- COLMAP
  - Sometimes capable of auto-detecting that it has failed
  - When failure reported for one frame, usually many frames have failed
  - Sometimes a small sequence of frames get completely misplaced, but then COLMAP recovers the correct trajectory again.
- VGGT
  - Cube map version sometimes veers off in the opposite direction of turning
  - Naive version is very uncertain about its depth estimates

== Key learnings
- Gaussian initialization still takes a long time and is hardware intensive

- The reconstruction quality is very sensitive to a good initialization

- Gaussian splatting methods can still not trivially handle large scenes


= Links
- #link("https://github.com/sarphiv/gaussian-splatting-360-integration")[Repository]
- #link("https://www.notion.so/2761409ead7180aeb63bd787d04bef54?
  v=2761409ead7180e18e1e000cd114fabb")[References]