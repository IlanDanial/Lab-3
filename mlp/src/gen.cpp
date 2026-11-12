// Read-only. Seeded generator: inputs in [0,1); weights and biases uniform in
// +-1/sqrt(fan_in) so activations stay O(1). No datasets, no training.
#include <cstddef>
#include <cmath>
#include "params.h"
#include "problem.h"

static unsigned xorshift(unsigned& s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}
static float uni01(unsigned& s) { return (float)((xorshift(s) >> 8) * (1.0 / 16777216.0)); }

void mlp_generate(int B, const int n[4], unsigned seed, MlpData& d) {
  unsigned s = seed * 2654435761u + 12345u;
  if (s == 0) s = 1;
  d.X.resize((size_t)B * n[0]);
  for (auto& v : d.X) v = uni01(s);
  for (int l = 0; l < MLP_LAYERS; l++) {
    const float scale = 1.0f / std::sqrt((float)n[l]);
    d.W[l].resize((size_t)n[l + 1] * n[l]);
    for (auto& v : d.W[l]) v = (2.0f * uni01(s) - 1.0f) * scale;
    d.b[l].resize((size_t)n[l + 1]);
    for (auto& v : d.b[l]) v = (2.0f * uni01(s) - 1.0f) * scale;
  }
}

MlpProblem mlp_problem_from(int B, const int n[4], const MlpData& d) {
  MlpProblem p;
  p.B = B;
  for (int i = 0; i < 4; i++) p.n[i] = n[i];
  p.X = d.X.data();
  for (int l = 0; l < MLP_LAYERS; l++) { p.W[l] = d.W[l].data(); p.b[l] = d.b[l].data(); }
  return p;
}
