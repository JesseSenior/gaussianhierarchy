import torch

from gaussian_hierarchy import _C


def create_hier(
    means: torch.Tensor,
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    camera_positions: torch.Tensor,
    output_dir: str,
    limit: float = 0.0005,
) -> None:
    """Create Gaussian hierarchy from tensor data

    Args:
        means: (N,3) Tensor of Gaussian means
        features_dc: (N,3) Tensor of DC SH coefficients
        features_rest: (N,15,3) Tensor of rest SH coefficients
        opacities: (N,) Tensor of opacities
        scales: (N,3) Tensor of scales
        quats: (N,4) Tensor of rotations (quaternions)
        camera_positions: (M,3) Tensor of camera positions
        output_dir: Directory to write output files
        limit: Appearance filtering threshold (default: 0.0005)
    """
    _C.create_hier(
        means.contiguous(),
        features_dc.contiguous(),
        features_rest.contiguous(),
        opacities.contiguous(),
        scales.contiguous(),
        quats.contiguous(),
        camera_positions.contiguous(),
        output_dir,
        limit,
    )


def merge_hier(
    hier_files: list[str],
    chunk_centers: torch.Tensor,
    output_path: str,
) -> None:
    """Merge multiple hierarchies into one based on chunk centers

    Args:
        hier_files: List of paths to input hierarchy files
        chunk_centers: (N,3) Tensor of chunk centers coordinates
        output_path: Path to write merged hierarchy file
    """
    # Convert tensor to CPU if needed
    if not chunk_centers.is_cpu:
        chunk_centers = chunk_centers.cpu()

    # Validate input dimensions
    torch._assert(chunk_centers.dim() == 2, "chunk_centers must be 2D tensor")
    torch._assert(chunk_centers.size(1) == 3, "chunk_centers must be Nx3")
    torch._assert(len(hier_files) == chunk_centers.size(0), "Mismatch between hier_files count and chunk_centers rows")

    # Convert Python list to vector<string> for C++ API
    import os

    full_paths = [os.path.abspath(p) for p in hier_files]

    # Call C++ implementation
    _C.merge_hier(full_paths, chunk_centers.contiguous(), output_path)
