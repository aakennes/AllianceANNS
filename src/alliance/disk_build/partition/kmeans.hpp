// Adapted from PipeANN (Apache-2.0); see third_party/PipeANN.LICENSE.
#pragma once
#include <cstddef>

namespace alliance::disk_build::vamana_kmeans {
// Train row-major centers until max_reps or residual convergence. Membership
// and distance-bound workspaces are internal; only the final centers are output.
void run_elkan(float *data, size_t num_points, size_t dim, float *centers, size_t num_centers,
               size_t max_reps);

// Initialize the caller-owned num_centers * dim buffer using kmeans++.
void kmeanspp_selecting_pivots(float *data, size_t num_points, size_t dim, float *pivot_data,
                               size_t num_centers);
} // namespace alliance::disk_build::vamana_kmeans
