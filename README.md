---
title: ML UV Unwrap
emoji: 🧊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# ML UV Unwrap

ML-trained UV unwrapping tool for 3D meshes. Upload a mesh (OBJ/GLB/PLY/STL/FBX)
and unwrap it with 23 methods:

- **Classical**: xatlas, LSCM, ABF, ARAP, harmonic, conformal, graph cuts, hilbert
- **Research**: voronoi disks, instant meshes, libuvula
- **ML**: FlexPara (unsupervised per-mesh optimization), PartUV, multi-chart, seam crafter, MeshTailor, UVSegNet, Flatten Anything, ArtUV, FFHQ-UV, hybrid, quality select, auto-detect

Built with FastAPI + Three.js + PyTorch (CPU).

## Notes

- Runs on CPU only; larger meshes may take a while.
- The PartField checkpoint (`model_objaverse.ckpt`, ~1.2 GB) is not bundled.
  PartUV / multi-chart modes fall back to geometric features automatically.
