"""Local browser demo: render initialization, then predict every RGB frame."""
from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
from dataclasses import dataclass
import io
from pathlib import Path
import pickle
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
from pydantic import Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
import torch

from worldline.server import BodyLimitMiddleware, StrictRequest
from .model import load_model, predict_tensor
from .simulator import Room
from typing import Literal

HERE = Path(__file__).resolve().parent


class Start(StrictRequest):
    seed: int = Field(default=31415, ge=0, le=2**31-1, strict=True)
    kind: Literal["predictor", "flow"] = "predictor"


class Action(StrictRequest):
    action: int = Field(ge=0, le=5, strict=True)
    revision: int = Field(ge=0, strict=True)


class Branch(StrictRequest):
    revision: int = Field(ge=0, strict=True)


@dataclass
class Session:
    history: torch.Tensor
    rng: torch.Generator
    kind: str
    seed: int
    tick: int = 0
    touched: float = 0


def create_app(artifact_root=HERE / "artifacts", model_loader=load_model, max_sessions=16):
    if type(max_sessions) is not int or max_sessions < 1:
        raise ValueError("max_sessions must be a positive integer")
    app = FastAPI(title="Worldline Room Lab")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
    app.add_middleware(BodyLimitMiddleware, limit=4096)
    sessions: OrderedDict[str, Session] = OrderedDict()
    models = {}
    lock = threading.RLock()
    artifact_root = Path(artifact_root)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host', '')}":
            return JSONResponse({"detail": "Open Room Lab at its local address."}, 403)
        response = await call_next(request)
        if response.status_code == 413:
            response = JSONResponse({"detail": "Room request exceeds 4 KB."}, 413)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Cache-Control"] = "no-store"
        return response

    def model_for(kind):
        if kind not in models:
            try:
                model, _ = model_loader(artifact_root / kind / "model.pt", device="cpu")
                if getattr(model, "kind", None) != kind:
                    raise ValueError("checkpoint kind does not match the requested model")
                models[kind] = model.eval()
            except FileNotFoundError as exc:
                raise HTTPException(503, "The original room checkpoint is missing. Restore experiments/room_world/artifacts from the repository.") from exc
            except (OSError, RuntimeError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as exc:
                raise HTTPException(503, "The room checkpoint could not be loaded. Restore experiments/room_world/artifacts from the repository.") from exc
        return models[kind]

    def get_session(sid, revision=None):
        if sid not in sessions or time.monotonic() - sessions[sid].touched > 3600:
            sessions.pop(sid, None)
            raise HTTPException(404, "This room session expired. Start a new room.")
        session = sessions[sid]
        if revision is not None and revision != session.tick:
            raise HTTPException(409, "This room changed in another request. Start a new room to continue.")
        session.touched = time.monotonic()
        sessions.move_to_end(sid)
        return session

    def store(session):
        sid = uuid.uuid4().hex
        session.touched = time.monotonic()
        sessions[sid] = session
        while len(sessions) > max_sessions:
            sessions.popitem(last=False)
        return sid

    def payload(sid, session, elapsed=None):
        frame = session.history[0, -1].detach().numpy()
        pixels = np.moveaxis(np.round((frame+1)*127.5).clip(0, 255).astype(np.uint8), 0, -1)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        return {"id": sid, "revision": session.tick, "seed": session.seed, "kind": session.kind,
                "image": "data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode("ascii"),
                "frame_source": "initial renderer" if session.tick == 0 else "neural prediction",
                "resolution": [64, 64], "inference_ms": elapsed}

    @app.post("/api/rooms")
    def start(body: Start):
        with lock:
            model_for(body.kind)
            # Only initialization sees the simulator. No Room is stored in a session.
            first = Room(body.seed, episode_seed=0, size=64).render().astype(np.float32)/127.5-1
            history = torch.from_numpy(np.repeat(first[None], 4, axis=0)).unsqueeze(0)
            session = Session(history, torch.Generator().manual_seed(body.seed), body.kind, body.seed)
            sid = store(session)
            return payload(sid, session)

    @app.post("/api/rooms/{sid}/step")
    def step(sid: str, body: Action):
        with lock, torch.inference_mode():
            session = get_session(sid, body.revision)
            start_time = time.perf_counter()
            # A failed request must leave both pixels and sampled-noise state
            # unchanged so a retry or a branch begins from the same state.
            next_rng = torch.Generator()
            next_rng.set_state(session.rng.get_state().clone())
            try:
                value = predict_tensor(model_for(session.kind), session.history,
                                       torch.tensor([body.action]), sample_steps=8, rng=next_rng)
            except (RuntimeError, ValueError) as exc:
                raise HTTPException(503, "The model could not generate an image. The room was not advanced. Try again or restart the room.") from exc
            if value.shape != session.history[:, -1].shape or not torch.isfinite(value).all():
                raise HTTPException(500, "The model produced an invalid image. The room was not advanced. Try again or restart the room.")
            next_history = torch.cat([session.history[:, 1:], value.unsqueeze(1)], 1)
            elapsed = round((time.perf_counter()-start_time)*1000, 2)
            candidate = Session(next_history, next_rng, session.kind, session.seed, session.tick+1, session.touched)
            response = payload(sid, candidate, elapsed)
            session.history, session.rng, session.tick = next_history, next_rng, candidate.tick
            return response

    @app.post("/api/rooms/{sid}/branch")
    def branch(sid: str, body: Branch):
        with lock:
            parent = get_session(sid, body.revision)
            rng = torch.Generator()
            rng.set_state(parent.rng.get_state().clone())
            child = Session(parent.history.clone(), rng, parent.kind, parent.seed, parent.tick)
            child_id = store(child)
            return payload(child_id, child)

    @app.get("/api/rooms/{sid}")
    def read(sid: str):
        with lock:
            return payload(sid, get_session(sid))

    app.mount("/", StaticFiles(directory=HERE / "ui", html=True), name="room-lab")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    app = create_app()
    if args.open:
        def open_browser():
            import socket
            import webbrowser
            for _ in range(100):
                try:
                    with socket.create_connection(("127.0.0.1", args.port), timeout=.1):
                        webbrowser.open(f"http://127.0.0.1:{args.port}/")
                        return
                except OSError:
                    time.sleep(.1)
        threading.Thread(target=open_browser, daemon=True).start()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
