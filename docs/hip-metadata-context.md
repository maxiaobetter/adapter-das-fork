# Keep HIP metadata queries free of PyTorch device initialization

On the HCU CUDA facade, SGLang capacity detection entered
`get_nvgpu_memory_capacity`, failed to execute `nvidia-smi`, and fell back to
`torch.cuda.mem_get_info()`. That fallback initializes a PyTorch CUDA context
before the worker selects its device. Argument resolution also queried device
capability directly. These are shared platform paths, not GLM model operations.

The HIP path now queries `hipDeviceTotalMem` from the runtime already loaded by
Torch, takes the minimum total capacity across visible device ordinals, and
returns MiB. Non-HIP capacity detection retains its original implementation.
Audited early property/capability queries use HIP metadata; initialized workers
retain canonical Torch properties. Architecture lookup matches HIP PCI BDF to
SMI, rather than assuming identical enumeration order. Unknown early property
fields fail explicitly.

The same change removes CUDA allocation from vendor weak-reference discovery,
keeps resource trackers out of GPU bootstrap, disables adapter loading in
TVM-FFI build subprocesses, and defers device-dependent configuration logging.
The AST redirects are limited to the listed modules and produce source hashes
and rewritten-call audit records; installed vendor files are not modified.

## Validation and limits

The DTK 26.10 isolated probe imported the combined adapter/plugin and completed
real decode argument resolution. All eight visible devices returned 65520 MiB;
`cuda_initialized_after_imports` was false and `init_attempts` was empty.
The evidence summary records exact source hashes and its probe scope.
This was a pre-rebase integration probe, not an end-to-end run of the final
rebased PR branches. It does not establish that HIP itself allocates zero
resources, quantify post-restart memory savings, or close the remaining KV
capacity difference. The running GPQA service does not include this new capacity
and argument-resolution fix. No new operator-output comparison was performed.

## Startup-order regression on the current upstream framework

The previously validated framework kept `runner_utils/__init__.py` free of
imports. The current upstream initializer imports DeepEP, then routing and
quantization, before reaching `shared_read_event`. Importing that package from
the platform seam before installing its guards therefore initializes GPU 0
through `is_gfx1250_supported`. Moving only the architecture guard exposes a
second early query in `is_fp8_fnuz` and does not fix the startup order.

The seam now binds both the common and package-level CUDA predicates, metadata
queries, and architecture/version guards before importing the event module.
It retains the upstream package exports and the existing event factory override;
no framework or vendor source is rewritten. The intended `is_cuda=True` and
`is_hip=False` answers and the HIP hardware identity remain unchanged.

The isolated deployment completed real prefill and decode argument resolution
without initializing Torch CUDA. It retained the CUDA facade and original
quantization. After restart, all four nodes had eight scheduler processes with
one device allocation each; their parent and manager processes had no VRAM
allocation. No extra GPU-0 allocation remained in the owned process trees.
Together with the plugin index-K allocation and workspace-budget repairs,
all 16 decode ranks returned to 109440 KV tokens at the unchanged decode
static memory fraction of 0.858. Prefill retained its original 0.8 fraction.

Evidence is bound to the staged source manifest, launch start ticks, KFD process
records, and loaded-library hashes in `kv-repair-readiness-gate.json` under the
2026-11-09 repair audit. The 64 cold/history/changed-input requests completed
with MTP on all decode ranks, but strict greedy history token identity did not
pass; archived native requests also vary across ranks. This is operational
validation, not proof of full mathematical equivalence. Full-dataset accuracy
and clean performance comparison remain pending. The source change passes the
CI Ruff fatal-error selection, Python syntax, and whitespace checks.

## Preserve metadata adaptation with official resolution hooks

The framework baseline is the unmodified official `release/v0.5.22` commit
`baa9befa381511cd4d56f83980951e5a45fffc6d`. The earlier extracted PP-validation
framework function has been retired. Plugin compatibility now uses the existing
named model-resolution hook and changes only the inline DSA PP rejection.

`lightop_metadata.rewrite_resolution_function` reconstructs the audited official
function and retains any already-active metadata redirects. Before accepting a
plugin AST callback it compares compiled bytecode, constants, names, arguments,
flags, closure variables, and exception tables with the loaded function. Unknown
runtime changes fail explicitly. The returned function keeps the same live
globals and leaves the original object and framework files unchanged. It does
not replace an existing wrapper or silently discard platform adaptation.

Local source verification covered the actual official function, metadata import
rewriting, and the metadata rebound path whose code starts at line one. All
three retained the original function and globals; both metadata paths retained
the metadata provider. The source file hash is
`3a89e31d3ddba628c032d5f60c8485357bfc403cff0e40f1ffcd8635b95f622c`.
This is source-preservation evidence. Container bootstrap, all-rank context
ownership, KV capacity, and accuracy for this new combination remain unverified
while the validation nodes are unreachable. Earlier service results belong to
the retired framework-seam combination.
