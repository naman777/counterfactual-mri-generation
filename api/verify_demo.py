"""Exercise saved replay through both servers; never performs denoising.

python -m api.verify_demo --output reports/v4_dashboard_demo_20261017_01
"""
import argparse
import json
import time
from pathlib import Path

import httpx

from api import demo_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:3000/api")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    client = httpx.Client(base_url=args.url, timeout=60)
    health = client.get("/health").raise_for_status().json()
    assert health["demo_only"] is True
    records = []
    for filename in demo_cache.manifest()["results"].values():
        saved = json.loads((demo_cache.CACHE_DIR / filename).read_text(encoding="utf-8"))
        started = time.monotonic()
        response = client.post("/edit", json=saved["request"]).raise_for_status()
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert not any(e["type"] in {"frame", "error"} for e in events)
        result = next(e["result"] for e in events if e["type"] == "result")
        assert result["cached"] and result["cache_kind"] == "Saved demo evidence"
        assert result["metrics"] == saved["metrics"] and result["judges"] == saved["judges"]
        records.append(dict(id=result["id"], request=result["request"], seconds=time.monotonic()-started, exact_saved_match=True))
    bad = {**saved["request"], "seed": saved["request"]["seed"] + 999}
    assert client.post("/edit", json=bad).status_code == 409
    assert client.post("/edit", json={**saved["request"], "use_cache": False}).status_code == 409
    for kind in ("json", "png", "pdf"):
        client.get(f"/report/{records[0]['id']}?format={kind}").raise_for_status()
    for path in ("/results/field_strength", "/results/manufacturer", "/results-data/high-frequency", "/evidence", "/slices"):
        client.get(path).raise_for_status()
    receipt = dict(mode="same-machine HTTP rehearsal through Next.js proxy", second_laptop_tested=False,
                   health=health, all_saved_results_match=True, uncached_requests_blocked=True,
                   inference_frames=0, report_formats=["json", "png", "pdf"], requests=records)
    (args.output / "replay_http.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    (args.output / "README.md").write_text("# Dashboard demo verification\n\n`replay_http.json` was produced by `python -m api.verify_demo --output <this-folder>`.\nIt checks all 80 saved requests through the Next.js proxy, rejects uncached requests, and checks report formats and read-only pages.\nThis is a same-machine rehearsal. No second laptop was available. No research estimate changed.\n", encoding="utf-8")
    print(f"Verified {len(records)} exact saved replays; uncached edits blocked; JSON/PNG/PDF exports passed.")


if __name__ == "__main__":
    main()
