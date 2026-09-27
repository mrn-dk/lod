#!/usr/bin/env python
"""Serve a checkpoint over HTTP.

    POST /v1/score    one request: a state and typed questions -> calibrated answers
    GET  /v1/models   the model ids this server answers for
    GET  /health

The wire format and its translation live in `lod.serving.api`, the budget policy in
`lod.serving.scoring`; this is the HTTP shell around them. Set LOD_API_KEY to require
`Authorization: Bearer <key>`. Needs the `serve` extra (`uv sync --extra serve`).

    scripts/serve.py --ckpt runs/<run>-release --port 8000
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")  # must precede `import torch`

import argparse
import sys
from typing import Any

import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from lod.model import flexattn
from lod.model.packing import Packer, collate
from lod.model.scorer import OptionScoringModel, forward_batch
from lod.schema import Example, Question
from lod.serving import api
from lod.serving.scoring import answer


def _load(ckpt: str, device: torch.device, dtype: torch.dtype, attn: str):
    return OptionScoringModel.load(ckpt, torch_dtype=dtype, attn_impl=attn).to(device).eval()


def _flex_works(model, device: torch.device) -> bool:
    """One tiny forward. FlexAttention can import cleanly and still fail to compile (it
    needs a C++ toolchain and the Python headers); better to find out at startup."""
    pk = Packer.for_model(model)
    packed = pk.pack(Example(state="probe", questions=[
        Question(id="q", question="Which?", options=["a", "b"])]))
    batch = collate([packed], pk.pad_id, model.backbone_dtype,
                    pad_to_tile=flexattn.BLOCK_SIZE, dense_mask=False)
    try:
        with torch.no_grad():
            forward_batch(model, batch, device)
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means fall back
        print(f"[serve] FlexAttention probe failed ({type(exc).__name__}: {str(exc)[:120]})",
              file=sys.stderr)
        return False


def build_app(ckpt: str, device: torch.device, dtype: torch.dtype,
              attn: str = "auto") -> FastAPI:
    """`attn`: 'flex' is the sparse block mask, 'sdpa' the dense [B,1,T,T] additive mask,
    'auto' prefers flex and falls back. The dense mask costs T^2 entries per request, so
    it is the long-context bottleneck."""
    want_flex = attn in ("auto", "flex") and flexattn.available()
    if attn == "flex" and not want_flex:
        raise SystemExit("--attn flex: torch FlexAttention is not available here")
    model = _load(ckpt, device, dtype, "flex" if want_flex else "sdpa")
    if want_flex and not _flex_works(model, device):
        if attn == "flex":
            raise SystemExit("--attn flex: the probe failed; rerun with --attn sdpa")
        model = _load(ckpt, device, dtype, "sdpa")
        want_flex = False
    print(f"[serve] attention: {'flex' if want_flex else 'sdpa'}   budget: "
          f"{model.max_state_tokens} state / {model.max_total_tokens} total", file=sys.stderr)
    app = FastAPI(title="lod", version=api.MODEL_ID.removeprefix("lod-"))
    api_key = os.environ.get("LOD_API_KEY")

    def _require_key(authorization: str | None) -> None:
        if not api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        if authorization[len("Bearer "):].strip() != api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    @app.get("/v1/models")
    def v1_models() -> dict[str, Any]:
        return {"models": list(api.MODELS)}

    @app.post("/v1/score")
    async def v1_score(request: Request,
                       authorization: str | None = Header(default=None)) -> Any:
        _require_key(authorization)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=422, content={
                "error": {"field": "body", "message": "body must be valid JSON"}})
        try:
            return answer(model, body, device, flex=want_flex)
        except api.ValidationError as e:
            return JSONResponse(status_code=422, content={"error": e.as_detail()})

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "model": api.MODEL_ID, "backbone": model.backbone_name,
                "device": str(device)}

    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--weights-dtype", choices=["fp32", "bf16", "fp16"], default=None,
                   help="default bf16 on cuda, fp32 on cpu")
    p.add_argument("--attn", choices=["auto", "flex", "sdpa"], default="auto")
    args = p.parse_args()

    import uvicorn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}.get(
        args.weights_dtype, torch.bfloat16 if device.type == "cuda" else torch.float32)
    uvicorn.run(build_app(args.ckpt, device, dtype, args.attn), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
