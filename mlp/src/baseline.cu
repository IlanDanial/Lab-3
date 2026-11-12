// Read-only naive GPU baseline.
// Per layer: one thread per output element does a dot product straight from
// global memory, then a separate kernel adds the bias and applies ReLU.
#include <cstddef>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include "params.h"
#include "problem.h"

static int g_B, g_n[4];
static float *g_X = nullptr, *g_W[3] = {nullptr, nullptr, nullptr}, *g_b[3] = {nullptr, nullptr, nullptr};
static float *g_h1 = nullptr, *g_h2 = nullptr, *g_out = nullptr;

// Y[r][c] = sum_k X[r][k] * W[c][k]      (W is stored OUT x IN)
__global__ void matmul_naive(const float* X, const float* W, float* Y, int B, int n_in, int n_out) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < B && col < n_out) {
    float sum = 0.0f;
    for (int k = 0; k < n_in; k++) sum += X[row * n_in + k] * W[col * n_in + k];
    Y[row * n_out + col] = sum;
  }
}

__global__ void bias_relu(float* Y, const float* b, int B, int n_out, int relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B * n_out) {
    float v = Y[idx] + b[idx % n_out];
    if (relu) v = fmaxf(v, 0.0f);
    Y[idx] = v;
  }
}

static void upload(float** d, const float* h, size_t count) {
  CHECK(cudaMalloc((void**)d, count * sizeof(float)));
  CHECK(cudaMemcpy(*d, h, count * sizeof(float), cudaMemcpyHostToDevice));
}

void mlp_baseline_setup(const MlpProblem& p) {
  g_B = p.B;
  for (int i = 0; i < 4; i++) g_n[i] = p.n[i];
  upload(&g_X, p.X, (size_t)p.B * p.n[0]);
  for (int l = 0; l < MLP_LAYERS; l++) {
    upload(&g_W[l], p.W[l], (size_t)p.n[l + 1] * p.n[l]);
    upload(&g_b[l], p.b[l], (size_t)p.n[l + 1]);
  }
  CHECK(cudaMalloc((void**)&g_h1, (size_t)p.B * p.n[1] * sizeof(float)));
  CHECK(cudaMalloc((void**)&g_h2, (size_t)p.B * p.n[2] * sizeof(float)));
  CHECK(cudaMalloc((void**)&g_out, (size_t)p.B * p.n[3] * sizeof(float)));
}

void mlp_baseline_compute() {
  const float* in[3] = {g_X, g_h1, g_h2};
  float* outp[3] = {g_h1, g_h2, g_out};
  for (int l = 0; l < MLP_LAYERS; l++) {
    const int n_in = g_n[l], n_out = g_n[l + 1];
    dim3 block(16, 16);
    dim3 grid((n_out + 15) / 16, (g_B + 15) / 16);
    matmul_naive<<<grid, block>>>(in[l], g_W[l], outp[l], g_B, n_in, n_out);
    const int total = g_B * n_out;
    bias_relu<<<(total + 255) / 256, 256>>>(outp[l], g_b[l], g_B, n_out, l < MLP_LAYERS - 1 ? 1 : 0);
  }
  CHECK(cudaGetLastError());
}

void mlp_baseline_fetch(float* logits) {
  CHECK(cudaMemcpy(logits, g_out, (size_t)g_B * g_n[3] * sizeof(float), cudaMemcpyDeviceToHost));
}

void mlp_baseline_teardown() {
  cudaFree(g_X);
  for (int l = 0; l < MLP_LAYERS; l++) { cudaFree(g_W[l]); cudaFree(g_b[l]); g_W[l] = g_b[l] = nullptr; }
  cudaFree(g_h1); cudaFree(g_h2); cudaFree(g_out);
  g_X = g_h1 = g_h2 = g_out = nullptr;
}
