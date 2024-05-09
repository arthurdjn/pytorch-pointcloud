#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "ball_query/ball_query.h"
#include "fps/fps.h"
#include "k_interpolate/k_interpolate.h"
#include "knn/knn.h"
#include "sided_dist/sided_dist.h"
#include "three_interpolate/three_interpolate.h"
#include "three_nn/three_nn.h"

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
}
