// Read-only driver. Usage:
//   ./mlp B n0 n1 n2 n3 seed [--iters N] [--skip-ref] [--skip-baseline] [--baseline-ms X]
// Flow: generate data -> CPU reference (unless --skip-ref) -> run baseline and
// student once for the result -> verify -> time N repeated runs of each.
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include "params.h"
#include "problem.h"

struct Impl {
  const char* name;
  void (*setup)(const MlpProblem&);
  void (*compute)();
  void (*fetch)(float*);
  void (*teardown)();
};

static double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
      return std::numeric_limits<double>::infinity();
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
}

// Runs setup, one warm-up/first compute (result kept), `iters` timed computes,
// and checks the final result equals the first one.
static void run_impl(const Impl& im, const MlpProblem& p, int iters,
                     std::vector<float>& first, double& per_iter_ms, double& consistency) {
  const size_t n = (size_t)p.B * p.n[3];
  first.assign(n, 0.0f);
  std::vector<float> last(n, 0.0f);
  im.setup(p);
  im.compute();
  im.fetch(first.data());
  cudaEvent_t e0, e1;
  CHECK(cudaEventCreate(&e0));
  CHECK(cudaEventCreate(&e1));
  CHECK(cudaEventRecord(e0));
  for (int i = 0; i < iters; i++) im.compute();
  CHECK(cudaEventRecord(e1));
  CHECK(cudaEventSynchronize(e1));
  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, e0, e1));
  CHECK(cudaGetLastError());
  im.fetch(last.data());
  per_iter_ms = (double)ms / iters;
  consistency = rel_l2(last, first);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  im.teardown();
  printf("[%s] iters=%d total_ms=%.3f per_iter_ms=%.4f repeat_rel_l2=%.3e\n", im.name, iters,
         (double)ms, per_iter_ms, consistency);
}

static void report(const char* who, const std::vector<float>& got, const std::vector<double>& ref,
                   const MlpProblem& p, bool& all_pass) {
  MlpVerdict v = mlp_verify(got.data(), ref.data(), p.B, p.n[3]);
  printf("VERIFY %s_vs_ref rel_l2=%.3e argmax_checked=%ld argmax_mismatch=%ld %s\n", who, v.rel_l2,
         v.checked, v.mismatched, v.pass ? "PASS" : "FAIL");
  all_pass = all_pass && v.pass;
}

int main(int argc, char** argv) {
  if (argc < 7) {
    fprintf(stderr, "usage: %s B n0 n1 n2 n3 seed [--iters N] [--skip-ref] [--skip-baseline] [--baseline-ms X]\n", argv[0]);
    return 1;
  }
  const int B = atoi(argv[1]);
  int n[4];
  for (int i = 0; i < 4; i++) n[i] = atoi(argv[2 + i]);
  const unsigned seed = (unsigned)strtoul(argv[6], nullptr, 10);
  int iters = 1;
  bool skip_ref = false, skip_base = false;
  double baseline_ms_given = -1.0;
  for (int a = 7; a < argc; a++) {
    if (!strcmp(argv[a], "--iters") && a + 1 < argc) iters = atoi(argv[++a]);
    else if (!strcmp(argv[a], "--skip-ref")) skip_ref = true;
    else if (!strcmp(argv[a], "--skip-baseline")) skip_base = true;
    else if (!strcmp(argv[a], "--baseline-ms") && a + 1 < argc) baseline_ms_given = atof(argv[++a]);
    else { fprintf(stderr, "unknown argument: %s\n", argv[a]); return 1; }
  }
  if (B < 1 || iters < 1) { fprintf(stderr, "need B >= 1 and iters >= 1\n"); return 1; }
  for (int i = 0; i < 4; i++) if (n[i] < 1) { fprintf(stderr, "all dimensions must be >= 1\n"); return 1; }

  printf("[mlp] B=%d dims=%d->%d->%d->%d seed=%u iters=%d\n", B, n[0], n[1], n[2], n[3], seed, iters);
  MlpData data;
  mlp_generate(B, n, seed, data);
  MlpProblem prob = mlp_problem_from(B, n, data);

  std::vector<double> ref;
  if (!skip_ref) {
    ref.resize((size_t)B * n[3]);
    mlp_reference(prob, ref.data());
    printf("[reference] done\n");
  }

  bool all_pass = true;
  std::vector<float> base_first, stu_first;
  double base_ms = baseline_ms_given, stu_ms = 0.0, c = 0.0;

  if (!skip_base) {
    Impl base{"baseline", mlp_baseline_setup, mlp_baseline_compute, mlp_baseline_fetch, mlp_baseline_teardown};
    run_impl(base, prob, iters, base_first, base_ms, c);
    if (c > 1e-6) { printf("VERIFY baseline repeat_consistency FAIL (%.3e)\n", c); all_pass = false; }
    if (!skip_ref) report("baseline", base_first, ref, prob, all_pass);
  }

  Impl stu{"student", mlp_student_setup, mlp_student_compute, mlp_student_fetch, mlp_student_teardown};
  run_impl(stu, prob, iters, stu_first, stu_ms, c);
  if (c > 1e-6) { printf("VERIFY student repeat_consistency FAIL (%.3e)\n", c); all_pass = false; }
  if (!skip_ref) {
    report("student", stu_first, ref, prob, all_pass);
  } else if (!skip_base) {
    const double err = rel_l2(stu_first, base_first);
    const bool ok = std::isfinite(err) && err <= MLP_TOL_REL;
    printf("VERIFY student_vs_baseline rel_l2=%.3e (reference skipped) %s\n", err, ok ? "PASS" : "FAIL");
    all_pass = all_pass && ok;
  } else {
    printf("VERIFY skipped (no reference and no baseline)\n");
  }

  {  // plausibility floor: 2*B*n_in*n_out FLOPs per layer at the FP32 peak
    double flops = 0.0;
    for (int l = 0; l < MLP_LAYERS; l++) flops += 2.0 * (double)B * (double)n[l] * (double)n[l + 1];
    const double floor_ms = flops / (GPU_PEAK_FP32_TFLOPS * 1e12) * 1e3;
    if (stu_ms < floor_ms) {
      printf("VERIFY student implausible_time FAIL (%.4f ms per iteration is below the FP32 floor %.4f ms: compute() must do the full forward pass on every call)\n", stu_ms, floor_ms);
      all_pass = false;
    }
  }
  if (base_ms > 0.0) printf("SPEEDUP baseline/student = %.2fx (per-iteration, baseline %.4f ms, student %.4f ms)\n", base_ms / stu_ms, base_ms, stu_ms);
  if (!all_pass) printf("RESULT FAIL\n");
  else if (skip_ref && skip_base) printf("RESULT TIMING_ONLY\n");
  else printf("RESULT PASS\n");
  return all_pass ? 0 : 1;
}
