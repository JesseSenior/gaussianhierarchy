import torch

from gaussian_hierarchy._C import (
    create_hier,
    load_hierarchy,
    write_hierarchy,
    expand_to_target,
    expand_to_size,
    get_interpolation_weights,
)


def create_hier_from_tensors(
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
    create_hier(
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
