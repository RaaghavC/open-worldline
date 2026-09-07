"""Local inference and world persistence server. No external inference services."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .state import SCHEMA, BIOMES, ConflictError, WorldStore, edit_field, parse_prompt

ROOT = Path(__file__).resolve().parent.parent


class BodyLimitMiddleware:
    """Bound actual incoming bytes, including requests with chunked transfer."""
    def __init__(self, app, limit=2_000_000):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.limit:
                return await JSONResponse({"detail": "World file exceeds 2 MB."}, 413)(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class GenerateRequest(StrictRequest):
    prompt: str = Field(default="An alpine lake beneath snow-capped mountains", min_length=1, max_length=600)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    steps: int = Field(default=24, ge=8, le=64)


class EditRequest(StrictRequest):
    kind: Literal["raise", "lower", "plant", "water", "heat"]
    x: float = Field(ge=0, le=1)
    z: float = Field(ge=0, le=1)
    radius: float = Field(default=.09, ge=.015, le=.5)
    strength: float = Field(default=.25, ge=.01, le=1)
    revision: int = Field(ge=0)


class StepRequest(StrictRequest):
    rain: float = Field(default=0, ge=0, le=1)
    heat: float = Field(default=0, ge=0, le=1)
    steps: int = Field(default=10, ge=1, le=60)
    revision: int = Field(ge=0)


class BranchRequest(StrictRequest):
    name: str = Field(default="Alternate future", min_length=1, max_length=100)
    revision: int | None = Field(default=None, ge=0)


class RestoreRequest(StrictRequest):
    target: int = Field(ge=0)
    revision: int = Field(ge=0)


class ImportRequest(StrictRequest):
    state: dict
    name: str = Field(default="Imported world", min_length=1, max_length=100)


def create_app(store_path=None, checkpoint_dir=None, model_pair=None):
    app = FastAPI(title="Worldline", version="0.1.0")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
    app.add_middleware(BodyLimitMiddleware)
    store = WorldStore(store_path or os.environ.get("WORLDLINE_DB", str(ROOT / ".worldline" / "worlds.sqlite")))
    app.state.store = store
    checkpoint_dir = Path(checkpoint_dir or ROOT / "checkpoints")
    # These small models were faster on CPU in the measured laptop benchmark;
    # keep the graphics GPU available to the browser. Larger experiments can opt in.
    inference_device = os.environ.get("WORLDLINE_DEVICE", "cpu")
    model_lock = threading.RLock()
    models = model_pair

    def load_models():
        nonlocal models
        with model_lock:
            if models is None:
                try:
                    from .models import get_models
                    import torch
                    if inference_device == "cpu":
                        torch.set_num_threads(min(4, torch.get_num_threads()))
                    models = get_models(checkpoint_dir, device=inference_device)
                except FileNotFoundError as exc:
                    raise HTTPException(503, "Trained weights are missing. Run python -m worldline.train first.") from exc
            return models

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host', '')}":
            return JSONResponse({"detail": "Use Worldline from its local app address."}, 403)
        length = request.headers.get("content-length", "0")
        try:
            if int(length) > 2_000_000:
                return JSONResponse({"detail": "World file exceeds 2 MB."}, 413)
        except ValueError:
            return JSONResponse({"detail": "Invalid request length."}, 400)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        return response

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "World or revision not found."}, 404)

    @app.exception_handler(ConflictError)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, 409)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, 422)

    @app.get("/api/status")
    def status():
        import torch
        return {"name": "Worldline", "version": "0.1.0", "device": inference_device,
                "trained": all((checkpoint_dir / name).is_file() for name in ("spatial-flow.pt", "ecology.pt")), "biomes": list(BIOMES),
                "generation": "Original conditional neural flow model, 64 × 64 spatial fields",
                "rendering": "Three.js geometry and shaders", "parity_with_genie3": "not established"}

    @app.get("/api/worlds")
    def worlds():
        return store.list()

    @app.post("/api/worlds")
    def generate(body: GenerateRequest):
        start = time.perf_counter()
        biome = parse_prompt(body.prompt)
        with model_lock:
            generator, _ = load_models()
            fields = generator.sample(body.seed, biome, body.steps)
        vegetation = np.clip((fields[1] + 1) / 2, 0, 1)
        water = np.clip((-fields[0] - .1) * .6, 0, 1)
        state = {"schema": SCHEMA, "prompt": body.prompt, "seed": body.seed, "biome": biome,
                 "height": np.clip(fields[0], -1, 1).tolist(),
                 "ecology": np.stack([water, vegetation, np.zeros_like(water)]).tolist(),
                 "tick": 0, "weather": {"rain": 0, "heat": 0}}
        world = store.create(state, f"{BIOMES[biome].title()} · {body.seed}", event={"type": "generate", "steps": body.steps, "seed": body.seed})
        world["generation_ms"] = round((time.perf_counter() - start) * 1000, 1)
        return world

    @app.post("/api/import")
    def import_world(body: ImportRequest):
        return store.create(body.state, body.name, event={"type": "import"})

    @app.get("/api/worlds/{wid}")
    def get_world(wid: str):
        return store.get(wid)

    @app.get("/api/worlds/{wid}/history")
    def history(wid: str):
        return store.history(wid)

    @app.get("/api/worlds/{wid}/export")
    def export(wid: str):
        world = store.get(wid)
        return JSONResponse({"name": world["name"], "state": world["state"]}, headers={"Content-Disposition": f'attachment; filename="worldline-{world["id"]}.json"'})

    @app.post("/api/worlds/{wid}/edit")
    def edit(wid: str, body: EditRequest):
        world = store.get(wid)
        state = edit_field(world["state"], body.kind, body.x, body.z, body.radius, body.strength)
        return store.save(wid, state, {"type": "edit", **body.model_dump(exclude={"revision"})}, body.revision)

    @app.post("/api/worlds/{wid}/step")
    def step(wid: str, body: StepRequest):
        world = store.get(wid)
        state = world["state"]
        eco = np.asarray(state["ecology"], dtype=np.float32)
        start = time.perf_counter()
        with model_lock:
            _, dynamics = load_models()
            for _ in range(body.steps):
                eco = dynamics.step(eco, body.rain, body.heat)
        state["ecology"] = eco.tolist()
        state["tick"] += body.steps
        state["weather"] = {"rain": body.rain, "heat": body.heat}
        result = store.save(wid, state, {"type": "simulate", **body.model_dump(exclude={"revision"})}, body.revision)
        result["simulation_ms"] = round((time.perf_counter() - start) * 1000, 1)
        return result

    @app.post("/api/worlds/{wid}/branch")
    def branch(wid: str, body: BranchRequest):
        return store.branch(wid, body.name, body.revision)

    @app.post("/api/worlds/{wid}/restore")
    def restore(wid: str, body: RestoreRequest):
        return store.restore(wid, body.target, body.revision)

    @app.get("/api/metrics")
    def metrics():
        import json
        files = list(checkpoint_dir.glob("*metrics*.json"))
        files += list((ROOT / "tests" / "results").glob("*metrics*.json"))
        return {p.stem: json.loads(p.read_text()) for p in files}

    dist = ROOT / "web" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="app")
    else:
        @app.get("/")
        def no_build():
            return {"message": "Build the interface with npm --prefix web install and npm --prefix web run build."}
    return app


def main():
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--open", action="store_true", help="Open the local editor in your browser once it is ready")
    args = parser.parse_args()
    app = create_app()
    if args.open:
        def open_editor():
            import socket
            import webbrowser
            for _ in range(100):
                try:
                    with socket.create_connection(("127.0.0.1", args.port), timeout=.1):
                        webbrowser.open(f"http://127.0.0.1:{args.port}/")
                        return
                except OSError:
                    time.sleep(.1)
        threading.Thread(target=open_editor, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
