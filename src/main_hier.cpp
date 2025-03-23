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
    const torch::Tensor& chunk_centers,
    const std::string& output_path)
{
    std::cout << "\n=== 开始合并层次结构 ===" << std::endl;
    std::cout << "待合并块数: " << hier_files.size() << std::endl;
    std::cout << "输出路径: " << output_path << std::endl;

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
        std::cout << "\n正在处理块 " << chunk_id + 1 << "/" << num_chunks << std::endl;
        std::cout << "块中心: " << chunk_centers_vec[chunk_id].transpose() << std::endl;
        
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
        std::cout << "已加载块 " << chunk_id << " 包含 " << chunk_gaussians.size() 
                  << " 个高斯" << std::endl;
        root->children.push_back(chunk_root);
        root->merged.push_back(chunk_root->merged[0]);
        root->depth = std::max(root->depth, chunk_root->depth + 1);
        
        // Merge gaussians
        gaussians.insert(gaussians.end(), chunk_gaussians.begin(), chunk_gaussians.end());
    }

    // Write merged hierarchy
    std::cout << "\n=== 合并完成 ===" << std::endl;
    std::cout << "合并后总高斯数: " << gaussians.size() << std::endl;
    std::cout << "最终根节点子节点数: " << root->children.size() << std::endl;
    Writer::writeHierarchy(output_path.c_str(), gaussians, root, true);
}

void CreateHier(
    torch::Tensor &means,
    torch::Tensor &features_dc,
    torch::Tensor &features_rest,
    torch::Tensor &opacities,
    torch::Tensor &scales,
    torch::Tensor &quats,
    const torch::Tensor &camera_positions,
    const std::string &output_dir,
    float limit = 0.0005f)
{
    // Validate input dimensions
    std::cout << "=== 开始创建层次结构 ===" << std::endl;
    std::cout << "输入参数验证通过" << std::endl;
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
        means = means.cpu();
    if (!features_dc.is_cpu())
        features_dc = features_dc.cpu();
    if (!opacities.is_cpu())
        opacities = opacities.cpu();
    if (!scales.is_cpu())
        scales = scales.cpu();
    if (!quats.is_cpu())
        quats = quats.cpu();

    // Create Gaussian structures
    const int count = means.size(0);
    std::vector<Gaussian> gaussians(count);

    // 输出基本信息
    std::cout << "总高斯数: " << count << std::endl;
    std::cout << "输出目录: " << output_dir << std::endl;

    auto means_a = means.accessor<float, 2>();
    auto features_dc_a = features_dc.accessor<float, 2>();
    auto features_rest_a = features_rest.accessor<float, 3>();
    auto opacities_a = opacities.accessor<float, 2>();
    auto scales_a = scales.accessor<float, 2>();
    auto quats_a = quats.accessor<float, 2>();

    std::cout << "\n开始转换高斯数据..." << std::endl;
#pragma omp parallel for
    for (int i = 0; i < count; ++i)
    {
        gaussians[i].position = Eigen::Vector3f(means_a[i][0], means_a[i][1], means_a[i][2]);
        gaussians[i].opacity = opacities_a[i][0];
        // Convert scale to log scale and rotation to normalized quaternion
        gaussians[i].scale = Eigen::Vector3f(
            std::exp(scales_a[i][0]),
            std::exp(scales_a[i][1]),
            std::exp(scales_a[i][2]));
        Eigen::Vector4f rot(quats_a[i][0], quats_a[i][1], quats_a[i][2], quats_a[i][3]);
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
        
        if (i % 10000 == 0) {
            std::cout << "已处理 " << i << "/" << count << " 个高斯点" << std::endl;
        }
    }
    std::cout << "高斯数据转换完成" << std::endl;

    // Process hierarchy
    std::cout << "\n生成初始KD树结构..." << std::endl;
    PointbasedKdTreeGenerator generator;
    ExplicitTreeNode *root = generator.generate(gaussians);
    std::cout << "根节点边界: [" << root->bounds.minn.transpose() << "] - [" 
              << root->bounds.maxx.transpose() << "]" << std::endl;

    std::cout << "\n执行聚类合并..." << std::endl;
    ClusterMerger merger;
    merger.merge(root, gaussians);
    std::cout << "聚类合并完成，当前层次深度: " << root->depth << std::endl;

    std::cout << "\n对齐旋转..." << std::endl;
    RotationAligner::align(root, gaussians);
    std::cout << "旋转对齐完成" << std::endl;

    // Initialize appearance filter with camera positions
    AppearanceFilter filter;
    filter.init(camera_positions);
    filter.filter(root, gaussians, limit, 2.0f);

    // Write output files
    std::cout << "\n写入输出文件..." << std::endl;
    std::cout << "生成文件: " << output_dir << "/hierarchy.hier" << std::endl;
    filter.writeAnchors((output_dir + "/anchors.bin").c_str(), root, gaussians, limit);
    Writer::writeHierarchy((output_dir + "/hierarchy.hier").c_str(), gaussians, root, true);
}
