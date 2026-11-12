#pragma once
#include <cstddef>
#include <vector>
// Read-only.
//
// All arrays are float, row-major.
//   X       : B  x n[0]            (B inputs, one per row)
//   W[l]    : n[l+1] x n[l]        (OUT x IN: row j holds the weights of output j)
//   b[l]    : n[l+1]
// Layer l computes   Y[r][j] = sum_k H[r][k] * W[l][j*n[l] + k] + b[l][j]
// then ReLU for l = 0, 1 (not for l = 2). The result is logits: B x n[3].
struct MlpProblem {
  int B;
  int n[4];
  const float* X;
  const float* W[3];
  const float* b[3];
};

struct MlpData {              // owns the generated arrays
  std::vector<float> X, W[3], b[3];
};

// ---- provided (gen.cpp, ref_cpu.cpp, verify.cpp) ----
void mlp_generate(int B, const int n[4], unsigned seed, MlpData& d);
MlpProblem mlp_problem_from(int B, const int n[4], const MlpData& d);
void mlp_reference(const MlpProblem& p, double* out);   // out: B*n[3] doubles

struct MlpVerdict {
  double rel_l2;        // ||got - ref||_2 / ||ref||_2 over all logits
  long checked;         // rows whose top-2 margin exceeded MLP_MARGIN
  long mismatched;      // of those, rows whose argmax differs from the reference
  bool pass;
};
MlpVerdict mlp_verify(const float* got, const double* ref, int B, int nout);

// ---- provided naive GPU version (baseline.cu) ----
void mlp_baseline_setup(const MlpProblem& p);  // allocate + copy weights/inputs (untimed)
void mlp_baseline_compute();                   // ONE full forward pass over all B rows
void mlp_baseline_fetch(float* logits);        // copy B*n[3] logits back (untimed)
void mlp_baseline_teardown();

// ---- Student version (student.cu) ----
// setup may change the input layout, but must not precompute layer outputs.
// compute must run one full forward pass on every call.
void mlp_student_setup(const MlpProblem& p);
void mlp_student_compute();
void mlp_student_fetch(float* logits);
void mlp_student_teardown();
