// Your implementation starts as a copy of the provided baseline.
// Edit this file and add .cu/.cuh files in src/ as needed.
// setup: allocate device memory and upload inputs (untimed).
//        You may change the input layout. Do not precompute layer outputs.
// compute: run one full forward pass over all B rows on every call
//          (timed, device work only).
// fetch: copy the B * n[3] logits back to the host (untimed).
// teardown: free your allocations (untimed).
// Use float32 arithmetic and keep the layer semantics in include/problem.h.
// See include/problem.h and README.md for the full requirements.

#include <cstddef>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include "params.h"
#include "problem.h"

static int s_B, s_n[4];
static float *s_X = nullptr, *s_W[3] = {nullptr, nullptr, nullptr}, *s_b[3] = {nullptr, nullptr, nullptr};
static float *s_h1 = nullptr, *s_h2 = nullptr, *s_out = nullptr;

constexpr int TILE = 16;      // threads per block side
constexpr int R = 8;          // each thread computes R x R outputs (best in sweep)
constexpr int BT = TILE * R;  // outputs per block side (128)
constexpr int BK = 32;        // k-depth of a shared tile (best in sweep)

// Y[r][c] = sum_k X[r][k] * W[k][c]      (W transposed in setup: IN x OUT)
// Thread owns rows ty + TILE*i, cols tx + TILE*j: conflict-free shared reads, coalesced writes
__global__ void matmul_student(const float* X, const float* W, float* Y, int B, int n_in, int n_out) {
  __shared__ float Xs[BT][BK];
  __shared__ float Ws[BK][BT];
  const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * TILE + tx;
  const int row0 = blockIdx.y * BT, col0 = blockIdx.x * BT;
  float acc[R][R] = {};
  for (int k0 = 0; k0 < n_in; k0 += BK) {
    // Both tiles hold BT*BK values; zero-fill past the edges
    for (int e = tid; e < BT * BK; e += TILE * TILE) {
      const int xr = row0 + e / BK, xk = k0 + e % BK;
      Xs[e / BK][e % BK] = (xr < B && xk < n_in) ? X[xr * n_in + xk] : 0.0f;
      const int wk = k0 + e / BT, wc = col0 + e % BT;
      Ws[e / BT][e % BT] = (wk < n_in && wc < n_out) ? W[wk * n_out + wc] : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; k++) {
      float x[R], w[R];
      for (int i = 0; i < R; i++) x[i] = Xs[ty + TILE * i][k];
      for (int j = 0; j < R; j++) w[j] = Ws[k][tx + TILE * j];
      for (int i = 0; i < R; i++)
        for (int j = 0; j < R; j++) acc[i][j] += x[i] * w[j];
    }
    __syncthreads();
  }
  for (int i = 0; i < R; i++)
    for (int j = 0; j < R; j++) {
      const int r = row0 + ty + TILE * i, c = col0 + tx + TILE * j;
      if (r < B && c < n_out) Y[r * n_out + c] = acc[i][j];
    }
}

__global__ void bias_relu_student(float* Y, const float* b, int B, int n_out, int relu) {
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

void mlp_student_setup(const MlpProblem& p) {
  s_B = p.B;
  for (int i = 0; i < 4; i++) s_n[i] = p.n[i];
  upload(&s_X, p.X, (size_t)p.B * p.n[0]);
  for (int l = 0; l < MLP_LAYERS; l++) {
    const int n_in = p.n[l], n_out = p.n[l + 1];
    std::vector<float> wt((size_t)n_in * n_out);   // IN x OUT, for coalesced reads
    for (int j = 0; j < n_out; j++)
      for (int k = 0; k < n_in; k++) wt[(size_t)k * n_out + j] = p.W[l][(size_t)j * n_in + k];
    upload(&s_W[l], wt.data(), wt.size());
    upload(&s_b[l], p.b[l], (size_t)n_out);
  }
  CHECK(cudaMalloc((void**)&s_h1, (size_t)p.B * p.n[1] * sizeof(float)));
  CHECK(cudaMalloc((void**)&s_h2, (size_t)p.B * p.n[2] * sizeof(float)));
  CHECK(cudaMalloc((void**)&s_out, (size_t)p.B * p.n[3] * sizeof(float)));
}

void mlp_student_compute() {
  const float* in[3] = {s_X, s_h1, s_h2};
  float* outp[3] = {s_h1, s_h2, s_out};
  for (int l = 0; l < MLP_LAYERS; l++) {
    const int n_in = s_n[l], n_out = s_n[l + 1];
    dim3 block(TILE, TILE);
    dim3 grid((n_out + BT - 1) / BT, (s_B + BT - 1) / BT);
    matmul_student<<<grid, block>>>(in[l], s_W[l], outp[l], s_B, n_in, n_out);
    const int total = s_B * n_out;
    bias_relu_student<<<(total + 255) / 256, 256>>>(outp[l], s_b[l], s_B, n_out, l < MLP_LAYERS - 1 ? 1 : 0);
  }
  CHECK(cudaGetLastError());
}

void mlp_student_fetch(float* logits) {
  CHECK(cudaMemcpy(logits, s_out, (size_t)s_B * s_n[3] * sizeof(float), cudaMemcpyDeviceToHost));
}

void mlp_student_teardown() {
  cudaFree(s_X);
  for (int l = 0; l < MLP_LAYERS; l++) { cudaFree(s_W[l]); cudaFree(s_b[l]); s_W[l] = s_b[l] = nullptr; }
  cudaFree(s_h1); cudaFree(s_h2); cudaFree(s_out);
  s_X = s_h1 = s_h2 = s_out = nullptr;
}
