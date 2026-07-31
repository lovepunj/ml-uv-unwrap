"""CLI entry point for ml-uv-unwrap."""

import argparse
import sys
from pathlib import Path


def cmd_unwrap(args):
    """Unwrap a single mesh."""
    from .pipeline.unwrapper import UVUnwrapPipeline

    pipeline = UVUnwrapPipeline(
        model_path=args.model,
        num_points=args.num_points,
        num_iterations=args.iterations,
        mode=args.mode,
    )

    result = pipeline.unwrap(args.input, log_every=args.log_every)
    pipeline.export(result, args.output)

    print(f"\nDone! UV map saved to {args.output}")


def cmd_train(args):
    """Train the model on a dataset (unsupervised per-mesh optimization)."""
    from torch.utils.data import DataLoader

    from .data.mesh_dataset import MeshDataset
    from .models import FlexParaUnwrapper
    from .training.trainer import train_batch

    model = FlexParaUnwrapper(
        num_charts=args.num_charts,
        hidden_dim=256,
        num_layers=8,
    )

    dataset = MeshDataset(
        root_dir=args.data_dir,
        num_points=args.num_points,
        augment=True,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"Training on {len(dataset)} meshes")
    print(f"  Num charts: {args.num_charts}")
    print(f"  Iterations per mesh: {args.iterations}")
    print(f"  Points per mesh: {args.num_points}")

    history = train_batch(
        model=model,
        dataloader=dataloader,
        num_iterations=args.iterations,
        lr=args.lr,
        save_dir=args.save_dir,
        device="cuda" if args.cuda else "cpu",
    )

    # Save final model
    save_path = Path(args.save_dir) / "model_final.pt"
    import torch
    torch.save(model.state_dict(), save_path)
    print(f"\nFinal model saved to {save_path}")


def cmd_train_supervised(args):
    """Train with ground-truth UV supervision."""
    import torch
    from torch.utils.data import DataLoader

    from .data.uv_dataset import UVDataset
    from .models.networks.artuv import ArtUVModel
    from .training.trainer import train_supervised

    model = ArtUVModel(
        hidden_dim=args.hidden_dim,
        num_graph_layers=args.num_layers,
    )

    dataset = UVDataset(
        root_dir=args.data_dir,
        num_points=args.num_points,
        max_verts=args.max_verts,
        augment=True,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"Supervised training on {len(dataset)} UV-mapped meshes")
    print(f"  Epochs: {args.epochs}")
    print(f"  Points per mesh: {args.num_points}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Layers: {args.num_layers}")

    history = train_supervised(
        model=model,
        dataloader=dataloader,
        num_epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
        device="cuda" if args.cuda else "cpu",
        log_every=max(1, args.epochs // 20),
        save_every=max(1, args.epochs // 10),
    )

    print(f"\nTraining complete. Model saved to {args.save_dir}/")


def cmd_prepare_data(args):
    """Download and prepare UV training data."""
    from .data.prepare_dataset import process_directory, download_objaverse
    from pathlib import Path

    output_dir = Path(args.output)
    input_dir = Path(args.input_dir) if args.input_dir else None

    if args.source == "objaverse":
        temp_dir = output_dir / "_raw_downloads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        downloaded = download_objaverse(temp_dir, args.max_samples, args.download_procs)
        if not downloaded:
            print("No files downloaded. Check your network connection.")
            return
        input_dir = temp_dir
    elif input_dir is None:
        print("ERROR: --input-dir required for manual source")
        return

    process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        max_samples=args.max_samples,
        max_verts=args.max_verts,
        max_faces=args.max_faces,
        quality_threshold=args.quality_threshold,
    )


def cmd_eval(args):
    """Evaluate unwrapping quality."""
    from .data.mesh_io import load_mesh, mesh_to_tensors, normalize_mesh
    from .evaluation.metrics import (
        angular_distortion,
        area_distortion,
        chart_count,
        seam_length,
    )
    from .pipeline.unwrapper import UVUnwrapPipeline

    pipeline = UVUnwrapPipeline(
        model_path=args.model,
        num_points=args.num_points,
    )

    input_path = Path(args.input)
    if input_path.is_dir():
        # Evaluate directory of meshes
        from .data.mesh_dataset import MeshDataset
        dataset = MeshDataset(input_path, num_points=args.num_points)
        paths = dataset.mesh_paths
    else:
        paths = [input_path]

    all_metrics = []
    for mesh_path in paths:
        print(f"\nEvaluating: {mesh_path}")
        result = pipeline.unwrap(mesh_path, num_iterations=args.iterations, log_every=500)

        metrics = {
            "path": str(mesh_path),
            "angular": angular_distortion(
                result["vertices"], result["uv_coords"], result["faces"]
            ),
            "area": area_distortion(
                result["vertices"], result["uv_coords"], result["faces"]
            ),
            "charts": chart_count(result["uv_coords"], result["faces"]),
        }
        all_metrics.append(metrics)

        print(f"  Angular distortion: {metrics['angular']['mean']:.2f}°")
        print(f"  Area distortion: {metrics['area']['mean_log']:.4f}")
        print(f"  Charts: {metrics['charts']}")

    # Summary
    if all_metrics:
        print("\n--- Summary ---")
        print(
            f"Mean angular distortion: "
            f"{np.mean([m['angular']['mean'] for m in all_metrics]):.2f}°"
        )
        print(
            f"Mean area distortion: "
            f"{np.mean([m['area']['mean_log'] for m in all_metrics]):.4f}"
        )
        print(
            f"Mean charts: "
            f"{np.mean([m['charts'] for m in all_metrics]):.1f}"
        )


def main():
    parser = argparse.ArgumentParser(
        prog="mluvunwrap",
        description="ML-trained UV unwrapping tool for 3D meshes",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # unwrap
    p_unwrap = subparsers.add_parser("unwrap", help="Unwrap a single mesh")
    p_unwrap.add_argument("input", help="Input mesh file (OBJ, PLY, etc.)")
    p_unwrap.add_argument("output", help="Output OBJ file with UVs")
    p_unwrap.add_argument("--model", "-m", help="Pretrained model checkpoint")
    p_unwrap.add_argument("--mode", default="classical",
                          choices=["ml", "classical", "hybrid", "multi_chart", "detect",
                                   "flatten_anything", "mesh_tailor", "seam_crafter",
                                   "uv_segnet", "quality_select", "artuv", "partuv",
                                   "arap", "harmonic", "conformal", "graph_cuts", "hilbert",
                                   "voronoi_disks", "instant_meshes", "libuvula", "ffhq_uv"],
                          help="Unwrapping algorithm to use")
    p_unwrap.add_argument("--num-points", "-n", type=int, default=10000)
    p_unwrap.add_argument("--iterations", "-i", type=int, default=1600)
    p_unwrap.add_argument("--log-every", type=int, default=100)
    p_unwrap.set_defaults(func=cmd_unwrap)

    # train
    p_train = subparsers.add_parser("train", help="Train on a mesh dataset")
    p_train.add_argument("data_dir", help="Directory of training meshes")
    p_train.add_argument("--save-dir", default="./checkpoints")
    p_train.add_argument("--num-charts", type=int, default=1)
    p_train.add_argument("--num-points", "-n", type=int, default=10000)
    p_train.add_argument("--iterations", "-i", type=int, default=1600)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--cuda", action="store_true")
    p_train.set_defaults(func=cmd_train)

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate unwrapping quality")
    p_eval.add_argument("input", help="Input mesh or directory of meshes")
    p_eval.add_argument("--model", "-m", help="Pretrained model checkpoint")
    p_eval.add_argument("--num-points", "-n", type=int, default=10000)
    p_eval.add_argument("--iterations", "-i", type=int, default=1600)
    p_eval.set_defaults(func=cmd_eval)

    # prepare-data
    p_data = subparsers.add_parser("prepare-data", help="Download and prepare UV training data")
    p_data.add_argument("--source", choices=["objaverse", "manual"], default="manual",
                        help="Data source: objaverse (auto-download) or manual (your files)")
    p_data.add_argument("--input-dir", type=str, default=None,
                        help="Input directory of mesh files (for manual mode)")
    p_data.add_argument("--output", type=str, required=True,
                        help="Output directory for preprocessed data")
    p_data.add_argument("--max-samples", type=int, default=5000)
    p_data.add_argument("--max-verts", type=int, default=10000)
    p_data.add_argument("--max-faces", type=int, default=20000)
    p_data.add_argument("--quality-threshold", type=float, default=0.3)
    p_data.add_argument("--download-procs", type=int, default=4)
    p_data.set_defaults(func=cmd_prepare_data)

    # train-supervised
    p_suv = subparsers.add_parser("train-supervised",
                                   help="Train with ground-truth UV supervision")
    p_suv.add_argument("data_dir", help="Directory of preprocessed UV data (.pt files)")
    p_suv.add_argument("--save-dir", default="./checkpoints_supervised")
    p_suv.add_argument("--epochs", type=int, default=100)
    p_suv.add_argument("--num-points", "-n", type=int, default=3000)
    p_suv.add_argument("--max-verts", type=int, default=10000)
    p_suv.add_argument("--lr", type=float, default=1e-3)
    p_suv.add_argument("--hidden-dim", type=int, default=128)
    p_suv.add_argument("--num-layers", type=int, default=5)
    p_suv.add_argument("--cuda", action="store_true")
    p_suv.set_defaults(func=cmd_train_supervised)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

# For rich compatibility
import numpy as np  # noqa: E402

app = main
