"""Persistent, exact-request replay of explicitly selected research evidence."""
import hashlib
import json
import os
from functools import lru_cache

from api.data import ROOT

CACHE_DIR = ROOT / "api/demo_bundle"


def request_key(req):
    values = req.model_dump(exclude={"use_cache"}) if hasattr(req, "model_dump") else dict(req)
    values.pop("use_cache", None)
    values["guidance"] = float(values["guidance"])
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def manifest():
    path = CACHE_DIR / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"slices": [], "results": {}}


def demo_only():
    return os.environ.get("MRONE_DEMO_ONLY", "0") == "1"


def recommended_id():
    """Open on a confirmed illustration; preserve disagreement cases in the set."""
    slices = manifest()["slices"]
    confirmed = [s for s in slices if s.get("confirmed") == 3]
    opener = next((s for s in confirmed if s["id"] == "932"), confirmed[0] if confirmed else None)
    return opener["id"] if opener else (slices[0]["id"] if slices else None)


def lookup(req):
    filename = manifest()["results"].get(request_key(req))
    if not filename:
        return None
    result = json.loads((CACHE_DIR / filename).read_text(encoding="utf-8"))
    if request_key(result["request"]) != request_key(req):
        raise ValueError("Saved demo request does not match its manifest")
    return {**result, "cached": True, "cache_kind": "Saved demo evidence"}
