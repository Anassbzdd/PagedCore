# FlashInfer T4 run: compiler linkage failure

Reviewed `pagedcore-flashinfer-7behvy1m.zip`, SHA-256
`9211c219924908fffeb6ec010f8b649f9f05f80498de65be9e29fc23a3336c2f`.
All **30 evidence hashes and 119 source hashes** verify. The embedded snapshot
SHA-256 is `f3eaa51ecfd13bc1ed9f333f3a320f91d0e27bc3f93f41d54ee24544fba2b576`.
The source is the recovered candidate, rather than the independently edited
Windows copy. Machine-readable review: `results/validation/2026-11-14/flashinfer-toolchain-t4-review.json`.

## What executed

The run identifies one visible Tesla T4 / SM75, Torch 2.7.1+cu126, FlashInfer
0.2.5 and Ninja 1.11.1.4. Package installation, dependency checking, checkpoint
verification, Ruff, formatting and strict mypy passed. All **808 CPU cases
passed with zero skips** on Kaggle. The actual SDK was **NVCC 13.0.88** and
the host compiler GCC 13.3.0; Torch's packaged CUDA runtime reports 12.6.

Eight native tests were selected and all failed. The first attempted JIT build
compiled three CUDA objects, then failed at host linking with an unresolved
`BatchDecodeWithPagedKVCacheKernel` template reference. The same compiler output
explicitly warns that `-static-global-template-stub=true` requires a definition
in the current translation unit. The other seven failures could not load the
shared library that the first build never produced. They are consequences of
the first failure, not independent attention-accuracy observations.

No native attention comparison completed. The 15+1 controls, frozen integrated
paging suite and 32-step cached trajectory did not execute. This archive cannot
support an improvement or regression claim about model accuracy. P5.3 stays open.

## Justified compiler recovery

NVIDIA's [CUDA 13 compiler documentation](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-compiler-driver-nvcc/index.html)
and [linkage migration note](https://developer.nvidia.com/blog/cuda-c-compiler-updates-impacting-elf-visibility-and-linkage/)
document the new internal-linkage default for template launch stubs and the
`--static-global-template-stub=false` compatibility option. Current upstream
[FlashInfer compiler support](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/jit/cpp_ext.py)
also includes this option for supported newer toolkits. This supports the
specific recovery; it does not prove the older candidate will pass on CUDA 13.

The replay notebook reads the actual NVCC release and, for CUDA 13+, supplies
that option through NVIDIA's documented `NVCC_APPEND_FLAGS` hook. It preserves
existing unrelated flags and rejects conflicting settings. Older toolkits do
not receive this new option. The retained 0.2.5 package's `jit/env.py` was
inspected to confirm `FLASHINFER_WORKSPACE_BASE`; a fresh run-local directory
isolates generated code and libraries without deleting old evidence or caches.

Every byte of the returned source snapshot is retained in
`notebooks/flashinfer_cuda13_recovery.ipynb`. The decoder, allocator, kernel
package, test cases, comparator and tolerances are unchanged. Compiler helper
source, warmup source, their hashes and environment overrides are exported as
evidence outside that frozen snapshot. This avoids mixing in differences from
the Windows copy. A one-key native compile/load smoke precedes the same eight
tests; success only establishes that first build/load and analytic one-key output.
Full kernel and integrated acceptance must still execute and be reviewed.

## Local validation

The focused tooling checks pass (14 tests, including existing notebook export
coverage). They cover the observed NVCC 13 release string, older toolkits,
idempotent flags, conflict rejection, notebook formatting variants and exact
payload preservation. Python warmup syntax is checked; native compilation cannot
be executed on the CPU-only Windows host. The full local non-GPU run passes
**820 tests, zero skips, 67 deselected**, in 533.22 seconds. Ruff check/format
and strict mypy (15 runtime source files) pass. Commands:

```text
python -m pytest -m "not gpu and not benchmark" -q --junitxml=results/validation/2026-11-14/flashinfer-toolchain-full-cpu.xml
python -m ruff check .
python -m ruff format --check .
python -m mypy --strict src/pagedcore
```

CPU execution sets `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`; Windows temporary
files use the writable validation directory. These local tests cover the Windows
copy and collector changes; the replay payload remains the separately verified
119-file source snapshot that already passed 808 CPU cases on Kaggle. No T4
success for the compiler recovery is claimed.
