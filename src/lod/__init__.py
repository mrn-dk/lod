"""Lod: prefill-only option scorers with calibrated, typed decisions."""

import os

# Recent torch routes some eager ops (RoPE's bmm among them) through Triton-JIT kernels,
# which need a C toolchain with Python headers to build; without one, plain CPU inference
# fails at import. Opt out unless the caller has decided otherwise. Must be set before
# torch is imported: the routing is registered at import time.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
