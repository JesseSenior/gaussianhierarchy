#include "torch_interface.h"
#include "writer.h"
#include "hierarchy_explicit_loader.h"
#include "FlatGenerator.h"
#include "PointbasedKdTreeGenerator.h"
#include "ClusterMerger.h"
#include "rotation_aligner.h"
#include "appearance_filter.h"
#include "common.h"
#include <torch/extension.h>
#include <vector>
#include <iostream>

using namespace torch;

void MergeHier(
    const std::vector<std::string>& hier_files,
    const Tensor& chunk_centers,
    const std::string& output_path)
{
    // Convert chunk centers tensor to Eigen vectors
    TORCH_CHECK(chunk_centers.dim() == 2, "chunk_centers must be 2D tensor");
    TORCH_CHECK(chunk_centers.size(1) == 3, "chunk_centers must be Nx3");
    
    const int num_chunks = chunk_centers.size(0);
    std::vector<Eigen::Vector3f> chunk_centers_vec(num_chunks);
    auto centers_a = chunk_centers.accessor<float, 2>();
    for (int i = 0; i < num_chunks; ++i) {
        chunk_centers_vec[i] = Eigen::Vector3f(
            centers_a[i][0], centers_a[i][1], centers_a[i][2]);
    }

    // Merge hierarchies
    std::vector<Gaussian> gaussians;
    ExplicitTreeNode* root = new ExplicitTreeNode;
    
    for (int chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
        ExplicitTreeNode* chunk_root = new ExplicitTreeNode;
        std::vector<Gaussian> chunk_gaussians;
        
        // Load explicit hierarchy for current chunk
        HierarchyExplicitLoader::loadExplicit(
            hier_files[chunk_id].c_str(),
            chunk_gaussians,
            chunk_root,
            chunk_id,
            chunk_centers_vec);

        // Merge bounds
        if (chunk_id == 0) {
            root->bounds = chunk_root->bounds;
        } else {
            for (int i = 0; i < 3; ++i) {
                root->bounds.minn[i] = std::min(root->bounds.minn[i], chunk_root->bounds.minn[i]);
                root->bounds.maxx[i] = std::max(root->bounds.maxx[i], chunk_root->bounds.maxx[i]);
            }
        }
        
        // Add as child node
        root->children.push_back(chunk_root);
        root->merged.push_back(chunk_root->merged[0]);
        root->depth = std::max(root->depth, chunk_root->depth + 1);
        
        // Merge gaussians
        gaussians.insert(gaussians.end(), chunk_gaussians.begin(), chunk_gaussians.end());
    }

    // Write merged hierarchy
    Writer::writeHierarchy(output_path.c_str(), gaussians, root, true);
}

void CreateHier(
    Tensor &means,
    Tensor &features_dc,
    Tensor &features_rest,
    Tensor &opacities,
    Tensor &scales,
    Tensor &quats,
    const Tensor &camera_positions,
    const std::string &output_dir,
    float limit = 0.0005f)
{
    // Validate input dimensions
    // Validate input dimensions
    TORCH_CHECK(means.size(0) == features_dc.size(0), "Means and features_dc must have same number of points");
    TORCH_CHECK(means.size(0) == features_rest.size(0), "Means and features_rest must have same number of points");
    TORCH_CHECK(means.size(0) == opacities.size(0), "Means and opacities must have same number of points");
    TORCH_CHECK(means.size(0) == scales.size(0), "Means and scales must have same number of points");
    TORCH_CHECK(means.size(0) == quats.size(0), "Means and quats must have same number of points");
    TORCH_CHECK(means.size(1) == 3, "Means must be Nx3");
    TORCH_CHECK(features_dc.size(1) == 3, "features_dc must be Nx3");
    TORCH_CHECK(features_rest.size(1) == 15 && features_rest.size(2) == 3, "features_rest must be Nx15x3");
    TORCH_CHECK(scales.size(1) == 3, "Scales must be Nx3");
    TORCH_CHECK(quats.size(1) == 4, "Rotations must be Nx4");

    // Convert tensors to CPU if needed
    if (!position.is_cpu())
        position = position.cpu();
    if (!color.is_cpu())
        color = color.cpu();
    if (!opacity.is_cpu())
        opacity = opacity.cpu();
    if (!scale.is_cpu())
        scale = scale.cpu();
    if (!rotation.is_cpu())
        rotation = rotation.cpu();

    // Create Gaussian structures
    const int count = position.size(0);
    std::vector<Gaussian> gaussians(count);

    auto means_a = means.accessor<float, 2>();
    auto features_dc_a = features_dc.accessor<float, 2>();
    auto features_rest_a = features_rest.accessor<float, 3>();
    auto opacities_a = opacities.accessor<float, 1>();
    auto scales_a = scales.accessor<float, 2>();
    auto quats_a = quats.accessor<float, 2>();

#pragma omp parallel for
    for (int i = 0; i < count; ++i)
    {
        gaussians[i].position = Eigen::Vector3f(position_a[i][0], position_a[i][1], position_a[i][2]);
        gaussians[i].opacity = opacity_a[i][0];
        // Convert scale to log scale and rotation to normalized quaternion
        gaussians[i].scale = Eigen::Vector3f(
            std::exp(scale_a[i][0]),
            std::exp(scale_a[i][1]),
            std::exp(scale_a[i][2]));
        Eigen::Vector4f rot(rotation_a[i][0], rotation_a[i][1], rotation_a[i][2], rotation_a[i][3]);
        gaussians[i].rotation = rot.normalized();

        // Copy SH coefficients with correct channel ordering (RGB)
        // features_dc: [N, 3] -> shs[0:3]
        for (int j = 0; j < 3; ++j)
            gaussians[i].shs[j] = features_dc_a[i][j]; // DC components

        // features_rest: [N, 15, 3] -> shs[3:48]
        for (int j = 1; j < 16; ++j)
            for (int k = 0; k < 3; ++k)
                gaussians[i].shs[j * 3 + k] = features_rest_a[i][j - 1][k];
        computeCovariance(gaussians[i].scale, gaussians[i].rotation, gaussians[i].covariance);
    }

    // Process hierarchy
    PointbasedKdTreeGenerator generator;
    ExplicitTreeNode *root = generator.generate(gaussians);

    ClusterMerger merger;
    merger.merge(root, gaussians);

    RotationAligner::align(root, gaussians);

    // Initialize appearance filter with camera positions
    AppearanceFilter filter;
    filter.init(camera_positions);
    filter.filter(root, gaussians, limit, 2.0f);

    // Write output files
    filter.writeAnchors((output_dir + "/anchors.bin").c_str(), root, gaussians, limit);
    Writer::writeHierarchy((output_dir + "/hierarchy.hier").c_str(), gaussians, root, true);
}
