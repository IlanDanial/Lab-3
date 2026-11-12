#pragma once
// Fixed constants. Read-only. Do not change!

// The network always has exactly 3 layers (dimensions n[0] -> n[1] -> n[2] -> n[3]).
// Layers 1 and 2 apply ReLU after the bias; layer 3 outputs raw logits.
constexpr int MLP_LAYERS = 3;

// Pass/fail tolerance: relative L2 error of the GPU logits against the
// double-precision CPU reference.
constexpr double MLP_TOL_REL = 1e-5;

// FP32 peak of the L4 GPU. The driver rejects a
// per-iteration time below (work / this peak): it means compute() skipped work.
constexpr double GPU_PEAK_FP32_TFLOPS = 30.0;

// The predicted class (argmax) must match the reference only for rows whose
// reference top-1 minus top-2 logit gap exceeds this value (near-ties can flip
// on tiny rounding differences).
constexpr double MLP_MARGIN = 1e-3;
