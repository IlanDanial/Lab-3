// Read-only. Double-precision CPU reference for the 3-layer forward pass.
#include <cstddef>
#include <vector>
#include "params.h"
#include "problem.h"

void mlp_reference(const MlpProblem& p, double* out) {
  std::vector<double> cur((size_t)p.B * p.n[0]);
  for (size_t i = 0; i < cur.size(); i++) cur[i] = (double)p.X[i];
  for (int l = 0; l < MLP_LAYERS; l++) {
    const int in = p.n[l], outn = p.n[l + 1];
    std::vector<double> nxt((size_t)p.B * outn);
#pragma omp parallel for
    for (int r = 0; r < p.B; r++) {
      const double* h = &cur[(size_t)r * in];
      for (int j = 0; j < outn; j++) {
        const float* w = p.W[l] + (size_t)j * in;
        double s = 0.0;
        for (int k = 0; k < in; k++) s += h[k] * (double)w[k];
        s += (double)p.b[l][j];
        if (l < MLP_LAYERS - 1 && s < 0.0) s = 0.0;
        nxt[(size_t)r * outn + j] = s;
      }
    }
    cur.swap(nxt);
  }
  for (size_t i = 0; i < cur.size(); i++) out[i] = cur[i];
}
