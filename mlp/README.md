# Lab 03, Option B: MLP Inference

Make a correct GPU program faster without changing its answer. This package
implements Option B from the Lab 3 potential-map and MLP slides. Read the
[top-level README](../README.md) first for the shared rules, timing, analysis,
and submission requirements.

## The problem

No neural-network background is needed. Three layers run in sequence. Each layer
multiplies its input by a weight matrix, adds a bias, and, in the first two
layers, replaces negative values with zero:

```
Y[r][j] = sum_k H[r][k] * W[j][k] + b[j]        then  Y = max(0, Y)   (layers 1 and 2 only)
```

- `H` is the layer input, with `B` rows. The first `H` is the input matrix `X`.
- The last layer's `B x n3` output is the answer. It has no `max(0, .)`.
- Dimensions come from the command line: `n0 -> n1 -> n2 -> n3`.
- Weights and inputs come from a seed. There is no training, softmax, or dataset.
- All arrays are row-major float. `X` is `B x n0`. `W[l]` is stored `out x in`, so
  row `j` of `W[l]` holds the weights of output `j`. `b[l]` has `n[l+1]` entries.

## What you edit

Only **`src/student.cu`**, plus `.cu`/`.cuh` files you add in `src/`. It starts
as a copy of the baseline. The Makefile builds every `src/*.cu`, so keep backups
outside `src/`. Everything else is read-only and restored before grading.

| File | Role |
|------|------|
| `include/params.h` | fixed constants and tolerances |
| `include/problem.h` | data layout and required functions |
| `include/cuda_check.h` | CUDA error checking |
| `src/gen.cpp` | seeded input generator |
| `src/ref_cpu.cpp` | double-precision CPU reference |
| `src/verify.cpp` | numerical verification |
| `src/baseline.cu` | naive GPU baseline |
| `src/main.cu` | input generation, verification, and timing |
| **`src/student.cu`** | **your implementation** |

## Expectations

Implement `mlp_student_setup`, `mlp_student_compute`, `mlp_student_fetch`, and
`mlp_student_teardown` (see `include/problem.h`).

- `setup`: allocate device memory and upload inputs; may change the input layout,
  for example by transposing `W`.
- `compute`: run **one full forward pass over all `B` rows on every call**, using
  device work only.
- `fetch`: copy the `B x n3` logits back to the host in row-major order.
- `teardown`: free your allocations.

Only `compute` is timed. Do not compute layer outputs in `setup`, and keep the
layer semantics above. The driver rejects per-iteration times below the GPU's
FP32 floor, because they mean `compute` skipped work.

## Build and run

On a GPU compute node, inside an allocation:

```bash
module load cuda gcc
make
./mlp 100 37 53 29 10 1     # B n0 n1 n2 n3 seed
./mlp 257 784 500 300 10 2 --iters 3
compute-sanitizer --tool memcheck --error-exitcode 1 ./mlp 100 37 53 29 10 1
```

See `sizes.txt` for correctness and benchmark commands. Write your own Slurm
script and timing log.

| Option | Meaning |
|--------|---------|
| `--iters N` | number of timed forward passes (default 1) |
| `--skip-ref` | skip the CPU reference for timing development |
| `--skip-baseline` | skip the baseline |
| `--baseline-ms X` | reuse a baseline per-iteration time measured with the same inputs and GPU |

Look for `VERIFY student_vs_ref ... PASS`, `repeat_rel_l2`, per-iteration times,
`SPEEDUP baseline/student`, and `RESULT PASS`. Without the CPU reference, the
driver compares the student output with the baseline when available. If both
comparisons are skipped, it prints `RESULT TIMING_ONLY`; this is not a correctness
check. The driver also checks that the final timed output agrees with the warm-up
output.

## Verification

The relative L2 error of your logits against the double-precision reference must
not exceed `MLP_TOL_REL` in `include/params.h`. The predicted class (the argmax
of each row) must also match, but only for rows where the reference's top-2 gap
exceeds `MLP_MARGIN`, because near-ties can flip on small rounding differences.
Recheck against the CPU reference after each change, test sizes that are not
multiples of your tile or block size, and require zero Compute Sanitizer errors.

The slides suggest considering shared data between threads, coalescing of `W`,
boundary checks on tiles, work per thread, and occupancy when choosing what to try.
