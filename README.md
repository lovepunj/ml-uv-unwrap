---
title: ML UV Unwrap
emoji: 🧊
colorFrom: blue
colorTo: green
pinned: false
---

# ML UV Unwrap

ML-trained UV unwrapping tool for 3D meshes. Upload a mesh (OBJ/GLB/PLY/STL/FBX)
and unwrap it with 23 methods:

- **Classical**: xatlas, LSCM, ABF, ARAP, harmonic, conformal, graph cuts, hilbert
- **Research**: voronoi disks, instant meshes, libuvula
- **ML**: FlexPara (unsupervised per-mesh optimization), PartUV, multi-chart, seam crafter, MeshTailor, UVSegNet, Flatten Anything, ArtUV, FFHQ-UV, hybrid, quality select, auto-detect

Built with FastAPI + Three.js + PyTorch (CPU).

## Hosting

- **Frontend**: GitHub Pages (`https://lovepunj.github.io/ml-uv-unwrap/`).
  The backend URL is injected into `web/static/config.js` at build time via the
  `UVUNWRAP_BACKEND_URL` repository variable (GitHub > Settings > Secrets and
  variables > Actions > Variables).
- **Backend**: Render free web service (512MB RAM, 0.1 vCPU, spins down after
  15 min idle, ephemeral disk). Deploy via `render.yaml` (Blueprint).

### Deploy backend to Render

1. Sign up at https://render.com (no credit card needed; sign in with GitHub).
2. Dashboard > **New** > **Blueprint** and select the
   `lovepunj/ml-uv-unwrap` repository. Render reads `render.yaml`
   (Docker build, free plan, health check on `/api/health`) and deploys.
   You get an HTTPS URL like `https://ml-uv-unwrap.onrender.com`.
3. Set `UVUNWRAP_BACKEND_URL` to that URL as a GitHub Actions **variable** and
   re-run the `Deploy frontend to GitHub Pages` workflow.

`autoDeploy: true` in `render.yaml` redeploys on every push to `main`.

### Alternative: deploy manually on Render

Dashboard > **New** > **Web Service** > GitHub > `lovepunj/ml-uv-unwrap`,
branch `main`, **Docker** runtime, free plan. Port is auto-detected from
`$PORT`; health check path `/api/health`.

## Notes

- Runs on CPU only; larger meshes may take a while. The free instance is
  slowest (0.1 vCPU) — expect several minutes per job.
- The PartField checkpoint (`model_objaverse.ckpt`, ~1.2 GB) is not bundled.
  PartUV / multi-chart modes fall back to geometric features automatically.
- The free Render instance has no persistent storage, restarts without notice,
  and sleeps after 15 min idle (cold start ~30-60s), so in-progress jobs are
  lost. Uploads are capped at 50MB.
