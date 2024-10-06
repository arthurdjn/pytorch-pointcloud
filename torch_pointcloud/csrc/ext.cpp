#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "avg_voxelize/avg_voxelize.h"
#include "ball_query/ball_query.h"
#include "fps/fps.h"
#include "k_interpolate/k_interpolate.h"
#include "knn/knn.h"
#include "scatter/scatter.h"
#include "sided_dist/sided_dist.h"
#include "three_interpolate/three_interpolate.h"
#include "three_nn/three_nn.h"
#include "trilinear_devoxelize/trilinear_devoxelize.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_query", &ball_query, "Ball Query (CPU/CUDA)");
  m.def("fps", &fps, "FPS (CPU/CUDA)");
  m.def("k_interpolate", &k_interpolate, "K Interpolate (CPU/CUDA)");
  m.def("k_interpolate_backward", &k_interpolate_backward);
  m.def("knn", &knn, "KNN (CPU/CUDA)");
  m.def("three_nn", &three_nn, "Three NN (CPU/CUDA)");
  m.def("three_interpolate", &three_interpolate, "Three Interpolate (CPU/CUDA)");
  m.def("three_interpolate_backward", &three_interpolate_backward);
  m.def("sided_distance", &sided_distance);
  m.def("sided_distance_backward", &sided_distance_backward);
  m.def("avg_voxelize", &avg_voxelize);
  m.def("avg_voxelize_backward", &avg_voxelize_backward);
  m.def("trilinear_devoxelize", &trilinear_devoxelize);
  m.def("trilinear_devoxelize_backward", &trilinear_devoxelize_backward);
  m.def("scatter_sum", &scatter_sum);
  m.def("scatter_mean", &scatter_mean);
  m.def("scatter_prod", &scatter_prod);
  m.def("scatter_min", &scatter_min);
  m.def("scatter_max", &scatter_max);
  m.def("scatter_sum_backward", &scatter_sum_backward);
  m.def("scatter_mean_backward", &scatter_mean_backward);
  m.def("scatter_prod_backward", &scatter_prod_backward);
  m.def("scatter_min_backward", &scatter_min_backward);
  m.def("scatter_max_backward", &scatter_max_backward);
}
