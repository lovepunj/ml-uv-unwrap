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
- **Backend**: Koyeb free instance (512MB RAM, 0.1 vCPU, scale-to-zero after
  1h of inactivity, no persistent disk). Deploy via `koyeb.yml` or the
  auto-deploy workflow in `.github/workflows/deploy-koyeb.yml`.

### Deploy backend to Koyeb

1. Create a Koyeb account at https://app.koyeb.com (no card needed for the
   free instance).
2. Generate an API token: Account Settings > API > New token.
3. Add it as a GitHub secret named `KOYEB_API_TOKEN`
   (Settings > Secrets and variables > Actions).
4. Push to `main` — the `Deploy backend to Koyeb` workflow builds the
   Dockerfile and deploys to a `ml-uv-unwrap.koyeb.app` URL.
5. Set `UVUNWRAP_BACKEND_URL` to that HTTPS URL and re-run the
   `Deploy frontend to GitHub Pages` workflow.

### Deploy backend manually (no token)

In the Koyeb dashboard: Create App > GitHub > `lovepunj/ml-uv-unwrap`,
branch `main`, builder **Dockerfile**, port **8080**, instance type **free**.

## Notes

- Runs on CPU only; larger meshes may take a while. The free instance is
  slowest (0.1 vCPU) — expect several minutes per job.
- The PartField checkpoint (`model_objaverse.ckpt`, ~1.2 GB) is not bundled.
  PartUV / multi-chart modes fall back to geometric features automatically.
- The free Koyeb instance has no persistent storage and sleeps after 1 hour
  without traffic, so in-progress jobs are lost. Uploads are capped at 50MB.
