# Lab 03: GPU Performance Optimization

Choose **one** application:

- [Option A: Electrostatic Potential Map](epm/README.md) — sum atom contributions at every point of a grid slice.
- [Option B: MLP Inference](mlp/README.md) — evaluate three dense layers with fixed weights and inputs.

Each option includes a naive GPU baseline, a matching `src/student.cu`, a seeded
input generator, a double-precision CPU reference, a verifier, and a timing
driver. Read this page, then the option's README and `sizes.txt`, before starting.

## Build

Run these commands on a GPU compute node inside an allocation:

```bash
module load cuda gcc
cd epm                      # or: cd mlp
make
```

The default architecture is `sm_89` for the course L4 (CUDA 11.8 or newer). Use
`make ARCH=sm_86` if required by an older course toolkit. If OpenMP linking is
unavailable, use `make OMP=0`. Do not change the optimization flags.

Edit `src/student.cu` and any `.cu`/`.cuh` files you add in `src/`. The remaining
files are fixed and restored before grading.

The Makefile builds every `src/*.cu`, so keep backups outside `src/`. A copy such
as `src/student_old.cu` repeats the same function definitions and fails at link
time.

## Rules

- Use float32 for all GPU storage and arithmetic.
- `setup` may change the input layout, but it must not precompute results.
- `compute` must do the complete calculation on every call. Do not cache answers
  between calls, skip work, or specialize for a known size or seed. Grading
  re-runs your code on a hidden size and seed.
- Write your own kernels. Do not use external compute libraries or multiple GPUs.

## How to approach optimizing a workload

1. Review the provided code and inputs. Understand the computation, how the
   baseline works, and which part of the program you are optimizing.

2. Start with a hypothesis. Describe the change you want to make, the effect
   you expect, and why. For example: "If I change X, I expect Y because Z."

3. Implement the proposed change. Change one thing at a time, then recheck
   correctness against the CPU reference. Include ragged sizes (not multiples
   of your block, tile, or coarsening size) and run Compute Sanitizer; zero
   errors are required. Each option's `sizes.txt` has the command.

4. Measure and analyze the result. Compare before-and-after median timings
   using the same GPU and inputs (see [Timing](#timing)). Did the result support
   your hypothesis? Explain what helped, what did not, and anything unexpected.

5. Use what you learned to form your next hypothesis and repeat the process.

EEL 4930 students must make at least **three** distinct optimization attempts;
EEL 5934 students must make at least **five**. Each attempt needs a hypothesis
and before-and-after median timings. An attempt that does not help still counts
if you measure and explain it. Changing only a parameter of a technique you already
tried, such as a block size or tile size, is not a new attempt.

## Timing

The driver warms up, then times repeated `compute` calls with CUDA events.
Allocation, upload, and download are excluded. It prints `per_iter_ms`, the
average for that run.

For each version you report, repeat the same command at least 5 times and record
the median `per_iter_ms` in your timing log, along with the iteration count. Use
`--skip-ref` for these repeats, because the CPU reference can be slow, and check
correctness with the reference separately. Keep the GPU and inputs fixed across
comparisons, and compute speedup from the median baseline and median student times.

## Analysis

Your report must explain, using your own measurements:

- bytes moved per output, data reuse, and the compute-to-memory ratio
- achieved throughput relative to the GPU's peak (look up the peak and cite it)
- the likely bottleneck
- a change that did not help, and why

## Extra credit

**Stretch (+10):** Sweep two parameters of your final kernel at the benchmark size,
such as block or tile size and coarsening factor, for at least 12 combinations. Plot
the median `per_iter_ms` for each combination (at least 5 runs each) and include the
raw output. Explain where performance peaks and falls off, and name the resource that
limits it, using your own measurements. Your best configuration must pass the
correctness tests and Compute Sanitizer.

## Submission

Submit through Canvas:

- `student.cu` and any source files you added
- your Slurm script
- a README with build and run commands
- a report with your optimization log and analysis
- verification and sanitizer logs, and raw timing output

Include your GPU model and CUDA version. Do not submit compiled binaries.

This is an individual lab. Disclose any AI tools, their tasks and contributions,
your changes, and how you verified the result.

See the Lab 3 slides and Canvas for deadlines, the full rubric, and
graduate-extension details.
