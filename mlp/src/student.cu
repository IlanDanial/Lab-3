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
#include <cuda_pipeline.h>
#include "cuda_check.h"
#include "params.h"
#include "problem.h"

static int s_B, s_n[4], s_ld[4];  // s_ld: row stride of each activation, padded to a multiple of 4
static float *s_X = nullptr, *s_W[3] = {nullptr, nullptr, nullptr}, *s_b[3] = {nullptr, nullptr, nullptr};
static float *s_h1 = nullptr, *s_h2 = nullptr, *s_out = nullptr;

constexpr int TILE = 16;      // threads per block side
constexpr int R = 8;          // each thread computes R x R outputs (best in sweep)
constexpr int BT = TILE * R;  // outputs per block side (128)
constexpr int BK = 32;        // k-depth of a shared tile (best in sweep; NBUF=3 needs BK=16 to fit 48 KB)
constexpr int NBUF = 1;       // pipeline stages: 1 = load then compute; N = copy N-1 tiles ahead (cp.async, measured no faster)
constexpr int ORDER = 0;      // FFMA order per k step: 0 = row by row, 1 = column by column, 2 = snake
static_assert(BK % 4 == 0 && BT % 4 == 0, "tiles are copied as float4");

// Copy 4 floats; out of bounds writes zeros and never reads src
template <bool ASYNC>
__device__ __forceinline__ void copy4(float* dst, const float* src, bool in) {
  if constexpr (ASYNC) {
    __pipeline_memcpy_async(dst, src, 16, in ? 0 : 16);
  } else {
    *reinterpret_cast<float4*>(dst) = in ? *reinterpret_cast<const float4*>(src) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
}

// Fill one X tile (BT x BK) and one W tile (BK x BT), one float4 per copy.
// Row strides are multiples of 4 with zero padding, so a float4 is fully inside a row or fully past it.
template <bool ASYNC>
__device__ __forceinline__ void load_tile(float (*Xs)[BK], float (*Ws)[BT], const float* X, int ldx,
                                          const float* W, int ldw, int B, int n_in,
                                          int row0, int col0, int k0, int tid) {
  constexpr int XQ = BK / 4, WQ = BT / 4;  // float4s per tile row
  for (int e = tid; e < BT * XQ; e += TILE * TILE) {
    const int r = e / XQ, q = 4 * (e % XQ), xr = row0 + r, xk = k0 + q;
    const bool in = xr < B && xk < ldx;
    copy4<ASYNC>(&Xs[r][q], in ? X + xr * ldx + xk : X, in);
  }
  for (int e = tid; e < BK * WQ; e += TILE * TILE) {
    const int r = e / WQ, q = 4 * (e % WQ), wk = k0 + r, wc = col0 + q;
    const bool in = wk < n_in && wc < ldw;
    copy4<ASYNC>(&Ws[r][q], in ? W + wk * ldw + wc : W, in);
  }
}

// Y[r][c] = sum_k X[r][k] * W[k][c]      (W transposed in setup: IN x OUT)
static_assert(R % 4 == 0, "W is read as float4");
constexpr int WC = 4 * TILE;  // stride between a thread's float4 column groups (64)

// Output column of a thread's j-th value: groups of 4 neighbors, WC apart
__device__ __forceinline__ int wcol(int tx, int j) { return 4 * tx + WC * (j / 4) + j % 4; }

// Thread owns rows ty + TILE*i and cols wcol(tx, j); W is read 4 columns per float4
// Launch bounds keep 2 blocks per SM (<= 128 registers)
__global__ void __launch_bounds__(TILE * TILE, 2)
matmul_student(const float* X, int ldx, const float* W, int ldw, float* Y, int ldy, int B, int n_in, int n_out) {
  __shared__ __align__(16) float Xs[NBUF][BT][BK];
  __shared__ __align__(16) float Ws[NBUF][BK][BT];
  const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * TILE + tx;
  const int row0 = blockIdx.y * BT, col0 = blockIdx.x * BT;
  const int nt = (n_in + BK - 1) / BK;
  float acc[R][R] = {};
  if constexpr (NBUF > 1) {
    for (int s = 0; s < NBUF - 1; s++) {  // prologue: start tiles 0 .. NBUF-2
      if (s < nt) load_tile<true>(Xs[s], Ws[s], X, ldx, W, ldw, B, n_in, row0, col0, s * BK, tid);
      __pipeline_commit();
    }
  }
  for (int t = 0; t < nt; t++) {
    const int cur = (NBUF > 1) ? t % NBUF : 0;
    if constexpr (NBUF > 1) {
      // Start tile t+NBUF-1 into the buffer read in iteration t-1 (freed by its closing sync), then wait for tile t
      const int nxt = t + NBUF - 1;
      if (nxt < nt) load_tile<true>(Xs[nxt % NBUF], Ws[nxt % NBUF], X, ldx, W, ldw, B, n_in, row0, col0, nxt * BK, tid);
      __pipeline_commit();
      __pipeline_wait_prior(NBUF - 1);
    } else {
      load_tile<false>(Xs[0], Ws[0], X, ldx, W, ldw, B, n_in, row0, col0, t * BK, tid);
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; k++) {
      float x[R], w[R];
      for (int i = 0; i < R; i++) x[i] = Xs[cur][ty + TILE * i][k];
      for (int g = 0; g < R / 4; g++) {
        const float4 v = *reinterpret_cast<const float4*>(&Ws[cur][k][wcol(tx, 4 * g)]);
        w[4 * g] = v.x; w[4 * g + 1] = v.y; w[4 * g + 2] = v.z; w[4 * g + 3] = v.w;
      }
      // Order decides which operand repeats between consecutive FFMAs (register reuse cache)
      if constexpr (ORDER == 0) {
        for (int i = 0; i < R; i++)
          for (int j = 0; j < R; j++) acc[i][j] += x[i] * w[j];
      } else if constexpr (ORDER == 1) {
        for (int j = 0; j < R; j++)
          for (int i = 0; i < R; i++) acc[i][j] += x[i] * w[j];
      } else {  // snake: odd rows run right to left, so each row change shares w[j]
        for (int i = 0; i < R; i++)
          for (int jj = 0; jj < R; jj++) {
            const int j = (i & 1) ? R - 1 - jj : jj;
            acc[i][j] += x[i] * w[j];
          }
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < R; i++)
    for (int j = 0; j < R; j++) {
      const int r = row0 + ty + TILE * i, c = col0 + wcol(tx, j);
      if (r < B && c < n_out) Y[r * ldy + c] = acc[i][j];  // padding columns stay zero
    }
}

__global__ void bias_relu_student(float* Y, int ldy, const float* b, int B, int n_out, int relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B * n_out) {
    const int r = idx / n_out, c = idx - r * n_out;
    float v = Y[r * ldy + c] + b[c];
    if (relu) v = fmaxf(v, 0.0f);
    Y[r * ldy + c] = v;
  }
}

static int pad4(int n) { return (n + 3) & ~3; }

static void upload(float** d, const float* h, size_t count) {
  CHECK(cudaMalloc((void**)d, count * sizeof(float)));
  CHECK(cudaMemcpy(*d, h, count * sizeof(float), cudaMemcpyHostToDevice));
}

static void alloc_zero(float** d, size_t count) {
  CHECK(cudaMalloc((void**)d, count * sizeof(float)));
  CHECK(cudaMemset(*d, 0, count * sizeof(float)));
}

void mlp_student_setup(const MlpProblem& p) {
  s_B = p.B;
  for (int i = 0; i < 4; i++) { s_n[i] = p.n[i]; s_ld[i] = pad4(p.n[i]); }
  std::vector<float> x((size_t)p.B * s_ld[0], 0.0f);  // X with zero-padded rows
  for (int r = 0; r < p.B; r++)
    for (int k = 0; k < p.n[0]; k++) x[(size_t)r * s_ld[0] + k] = p.X[(size_t)r * p.n[0] + k];
  upload(&s_X, x.data(), x.size());
  for (int l = 0; l < MLP_LAYERS; l++) {
    const int n_in = p.n[l], n_out = p.n[l + 1], ld = s_ld[l + 1];
    std::vector<float> wt((size_t)n_in * ld, 0.0f);  // IN x OUT, zero-padded rows, for coalesced reads
    for (int j = 0; j < n_out; j++)
      for (int k = 0; k < n_in; k++) wt[(size_t)k * ld + j] = p.W[l][(size_t)j * n_in + k];
    upload(&s_W[l], wt.data(), wt.size());
    upload(&s_b[l], p.b[l], (size_t)n_out);
  }
  alloc_zero(&s_h1, (size_t)p.B * s_ld[1]);
  alloc_zero(&s_h2, (size_t)p.B * s_ld[2]);
  alloc_zero(&s_out, (size_t)p.B * s_ld[3]);
}

void mlp_student_compute() {
  const float* in[3] = {s_X, s_h1, s_h2};
  float* outp[3] = {s_h1, s_h2, s_out};
  for (int l = 0; l < MLP_LAYERS; l++) {
    const int n_in = s_n[l], n_out = s_n[l + 1], ldx = s_ld[l], ldy = s_ld[l + 1];
    dim3 block(TILE, TILE);
    dim3 grid((n_out + BT - 1) / BT, (s_B + BT - 1) / BT);
    matmul_student<<<grid, block>>>(in[l], ldx, s_W[l], ldy, outp[l], ldy, s_B, n_in, n_out);
    const int total = s_B * n_out;
    bias_relu_student<<<(total + 255) / 256, 256>>>(outp[l], ldy, s_b[l], s_B, n_out, l < MLP_LAYERS - 1 ? 1 : 0);
  }
  CHECK(cudaGetLastError());
}

void mlp_student_fetch(float* logits) {
  // Drop the row padding while copying back
  CHECK(cudaMemcpy2D(logits, (size_t)s_n[3] * sizeof(float), s_out, (size_t)s_ld[3] * sizeof(float),
                     (size_t)s_n[3] * sizeof(float), s_B, cudaMemcpyDeviceToHost));
}

void mlp_student_teardown() {
  cudaFree(s_X);
  for (int l = 0; l < MLP_LAYERS; l++) { cudaFree(s_W[l]); cudaFree(s_b[l]); s_W[l] = s_b[l] = nullptr; }
  cudaFree(s_h1); cudaFree(s_h2); cudaFree(s_out);
  s_X = s_h1 = s_h2 = s_out = nullptr;
}
