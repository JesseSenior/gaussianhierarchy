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
        opacities: (N,1) Tensor of opacities
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


def load_hierarchy(path: str, device: str = "cuda") -> tuple:
    """Load hierarchy data from file

    Args:
        path: Path to hierarchy file
        device: Target device for tensors (default: cuda)

    Returns:
        Tuple of tensors (pos, shs, alpha, scale, rot, nodes, boxes)
    """
    # Load all tensors and move to target device
    result = _C.load_hierarchy(path)
    return tuple(t.to(device).contiguous() for t in result)


def write_hierarchy(
    pos: torch.Tensor,
    shs: torch.Tensor,
    opacities: torch.Tensor,
    log_scales: torch.Tensor,
    rotations: torch.Tensor,
    nodes: torch.Tensor,
    boxes: torch.Tensor,
    path: str,
) -> None:
    """Write hierarchy data to file

    Args:
        pos: (N,3) Positions tensor
        shs: (N,16,3) SH coefficients tensor
        opacities: (N,1) Opacities tensor
        log_scales: (N,3) Log scales tensor
        rotations: (N,4) Rotations tensor
        nodes: (N,7) Nodes metadata tensor
        boxes: (N,2,4) Bounding boxes tensor
        path: Path to write output file
    """
    # Ensure all tensors are contiguous and on CPU before writing
    _C.write_hierarchy(
        path,
        pos.contiguous().cpu(),
        shs.contiguous().cpu(),
        opacities.contiguous().cpu(),
        log_scales.contiguous().cpu(),
        rotations.contiguous().cpu(),
        nodes.contiguous().cpu(),
        boxes.contiguous().cpu(),
    )


def expand_to_size(
    nodes: torch.Tensor,
    boxes: torch.Tensor,
    size: float,
    viewpoint: torch.Tensor,
    viewdir: torch.Tensor,
    render_indices: torch.Tensor,
    parent_indices: torch.Tensor,
    nodes_for_render_indices: torch.Tensor,
) -> int:
    """Expand hierarchy nodes to target size based on viewing parameters

    Args:
        nodes: (N,7) Nodes metadata tensor
        boxes: (N,2,4) Bounding boxes tensor
        size: Target size threshold
        viewpoint: (3,) Camera position
        viewdir: (3,) View direction vector
        render_indices: (M,) Indices of nodes to render
        parent_indices: (M,) Parent node indices
        nodes_for_render_indices: (M,) Node indices for rendering

    Returns:
        Number of nodes after expansion
    """
    return _C.expand_to_size(
        nodes.contiguous(),
        boxes.contiguous(),
        size,
        viewpoint.contiguous(),
        viewdir.contiguous(),
        render_indices.contiguous(),
        parent_indices.contiguous(),
        nodes_for_render_indices.contiguous(),
    )


def get_interpolation_weights(
    indices: torch.Tensor,
    size: float,
    nodes: torch.Tensor,
    boxes: torch.Tensor,
    viewpoint: torch.Tensor,
    viewdir: torch.Tensor,
    ts: torch.Tensor,
    num_kids: torch.Tensor,
) -> None:
    """Calculate interpolation weights for specified nodes

    Args:
        indices: (K,) Node indices to process
        size: Size threshold for weight calculation
        nodes: (N,7) Nodes metadata tensor
        boxes: (N,2,4) Bounding boxes tensor
        viewpoint: (3,) Camera position
        viewdir: (3,) View direction vector
        ts: (K,) Output tensor for interpolation weights
        num_kids: (K,) Output tensor for child counts
    """
    _C.get_interpolation_weights(
        indices.contiguous(),
        size,
        nodes.contiguous(),
        boxes.contiguous(),
        viewpoint.cpu().contiguous(),
        viewdir.cpu().contiguous(),
        ts.contiguous(),
        num_kids.contiguous(),
    )
