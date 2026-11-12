// Read-only. Logits within a relative tolerance, plus argmax agreement on rows
// whose reference top-2 margin is large enough to be meaningful.
#include <cstddef>
#include <cmath>
#include "params.h"
#include "problem.h"

MlpVerdict mlp_verify(const float* got, const double* ref, int B, int nout) {
  double num = 0.0, den = 0.0;
  long checked = 0, mismatched = 0;
  for (int r = 0; r < B; r++) {
    const double* rr = ref + (size_t)r * nout;
    const float* gg = got + (size_t)r * nout;
    int a1 = 0;
    for (int j = 0; j < nout; j++) {
      const double d = (double)gg[j] - rr[j];
      num += d * d;
      den += rr[j] * rr[j];
      if (rr[j] > rr[a1]) a1 = j;
    }
    double second = -1e300;
    for (int j = 0; j < nout; j++) if (j != a1 && rr[j] > second) second = rr[j];
    if (nout > 1 && rr[a1] - second > MLP_MARGIN) {
      checked++;
      int g1 = 0;
      for (int j = 0; j < nout; j++) if (gg[j] > gg[g1]) g1 = j;
      if (g1 != a1) mismatched++;
    }
  }
  MlpVerdict v;
  v.rel_l2 = (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
  v.checked = checked;
  v.mismatched = mismatched;
  v.pass = (v.rel_l2 <= MLP_TOL_REL) && (mismatched == 0);  // NaN fails the first test
  return v;
}
