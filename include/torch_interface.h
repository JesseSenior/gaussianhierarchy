#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LoadHierarchy(std::string filename);

void WriteHierarchy(
					std::string filename,
					torch::Tensor& pos,
					torch::Tensor& shs,
					torch::Tensor& opacities,
					torch::Tensor& log_scales,
					torch::Tensor& rotations,
					torch::Tensor& nodes,
					torch::Tensor& boxes);

torch::Tensor
ExpandToTarget(torch::Tensor& nodes, int target);

int ExpandToSize(
torch::Tensor& nodes, 
torch::Tensor& boxes, 
float size, 
torch::Tensor& viewpoint, 
torch::Tensor& viewdir, 
torch::Tensor& render_indices,
torch::Tensor& parent_indices,
torch::Tensor& nodes_for_render_indices);

void GetTsIndexed(
torch::Tensor& indices,
float size,
torch::Tensor& nodes,
torch::Tensor& boxes,
torch::Tensor& viewpoint, 
torch::Tensor& viewdir, 
torch::Tensor& ts,
torch::Tensor& num_kids);

void CreateHier(
torch::Tensor& means,
torch::Tensor& features_dc,
torch::Tensor& features_rest,
torch::Tensor& opacities,
torch::Tensor& scales,
torch::Tensor& quats,
const torch::Tensor& camera_positions,
const std::string& output_dir,
float limit);

void MergeHier(
const std::vector<std::string>& hier_files,
const torch::Tensor& chunk_centers,
const std::string& output_path);
