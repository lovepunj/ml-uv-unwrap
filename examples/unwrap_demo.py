"""Quick demo of the ML UV unwrapping pipeline.

Usage:
    python examples/unwrap_demo.py              # Standard mode
    python examples/unwrap_demo.py --partuv     # PartUV mode
"""

import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_demo_mesh():
    """Create a simple demo mesh (icosphere) for testing."""
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    return mesh


def main():
    parser = argparse.ArgumentParser(description="ML UV Unwrap Demo")
    parser.add_argument("--partuv", action="store_true", help="Use PartUV mode")
    parser.add_argument("--classical", type=str, choices=["xatlas", "lscm", "abf"],
                        help="Use classical method (xatlas, lscm, abf)")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid mode (classical + ML)")
    parser.add_argument("--mesh", type=str, help="Input mesh path (default: icosphere)")
    parser.add_argument("--iters", type=int, default=800, help="Optimization iterations")
    parser.add_argument("--points", type=int, default=2000, help="Sample points")
    parser.add_argument("--output", type=str, default="demo_output.obj", help="Output path")
    args = parser.parse_args()

    import torch

    print("=== ML UV Unwrap Demo ===")
    if args.partuv:
        print("    Mode: PartUV (semantic part-aware)")
    elif args.classical:
        print(f"    Mode: Classical ({args.classical})")
    elif args.hybrid:
        print("    Mode: Hybrid (classical + ML)")
    else:
        print("    Mode: Standard ML")
    print()

    # 1. Create/load mesh
    if args.mesh:
        import trimesh
        print(f"1. Loading mesh: {args.mesh}...")
        mesh = trimesh.load(args.mesh, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
    else:
        print("1. Creating demo mesh (icosphere)...")
        mesh = create_demo_mesh()
    print(f"   Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    if args.partuv:
        from src.pipeline.unwrapper import UVUnwrapPipeline

        print(f"\n2. Running PartUV pipeline (PartField + FlexPara)...")
        pipeline = UVUnwrapPipeline(
            use_partuv=True,
            num_points=args.points,
            num_iterations=args.iters,
            device="cpu",
        )

        result = pipeline.unwrap(mesh, num_iterations=args.iters, log_every=max(1, args.iters // 10))
        print(f"\n   UV shape: {result['uv_coords'].shape}")
        print(f"   Parts detected: {result.get('num_parts', 'N/A')}")

        pipeline.export(result, args.output)
    elif args.classical:
        from src.pipeline.classical_unwrapper import ClassicalUnwrapper

        print(f"\n2. Running classical unwrap ({args.classical})...")
        unwrapper = ClassicalUnwrapper(method=args.classical)
        result = unwrapper.unwrap(mesh)

        print(f"\n   UV shape: {result['uv_coords'].shape}")

        # Export
        from src.pipeline.postprocessor import export_uv_mesh, pack_uv_charts, add_uv_margins
        uv_packed = pack_uv_charts(result["uv_coords"])
        uv_packed = add_uv_margins(uv_packed, result["faces"])
        export_uv_mesh(
            args.output,
            vertices=result["vertices"],
            uv_coords=uv_packed,
            faces=result["faces"],
        )
    elif args.hybrid:
        from src.pipeline.unwrapper import UVUnwrapPipeline

        print(f"\n2. Running hybrid pipeline (xatlas + ML refinement)...")
        pipeline = UVUnwrapPipeline(
            mode="hybrid",
            num_points=args.points,
            num_iterations=args.iters,
            device="cpu",
            classical_method="xatlas",
        )

        result = pipeline.unwrap(mesh, num_iterations=args.iters, log_every=max(1, args.iters // 10))
        print(f"\n   UV shape: {result['uv_coords'].shape}")

        pipeline.export(result, args.output)
    else:
        from src.models import FlexParaUnwrapper
        from src.pipeline.preprocessor import MeshPreprocessor
        from src.pipeline.postprocessor import export_uv_mesh, pack_uv_charts
        from src.training.trainer import train_unsupervised

        # 2. Preprocess
        print(f"\n2. Preprocessing ({args.points} points)...")
        preprocessor = MeshPreprocessor(num_points=args.points, device="cpu")
        data = preprocessor.process(mesh)
        points = data["points"]
        print(f"   Sampled points: {points.shape[1]}")

        # 3. Initialize model
        print("\n3. Initializing model...")
        model = FlexParaUnwrapper(
            num_charts=1,
            hidden_dim=128,
            num_layers=6,
        )
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {num_params:,}")

        # 4. Train
        print(f"\n4. Training ({args.iters} iterations)...")
        history = train_unsupervised(
            model=model,
            points=points,
            num_iterations=args.iters,
            lr=1e-3,
            log_every=max(1, args.iters // 10),
        )

        # 5. Extract UVs
        print("\n5. Extracting UV coordinates...")
        with torch.no_grad():
            outputs = model(points)
        uv_coords = outputs["uv_coords"][0].numpy()
        print(f"   UV shape: {uv_coords.shape}")

        # 6. Export
        print("\n6. Exporting...")
        output_path = Path(args.output)
        uv_packed = pack_uv_charts(uv_coords)
        export_uv_mesh(
            output_path,
            vertices=data["vertices"][0].numpy(),
            uv_coords=uv_packed,
            faces=data["faces"].numpy(),
        )

    print(f"\nDone! Output: {args.output}")
    print("Open in Blender or any 3D viewer to see the UV map.")


if __name__ == "__main__":
    main()
