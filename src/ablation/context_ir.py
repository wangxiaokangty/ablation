"""对 seed 目录下的 r2va 样本批量做 MiniMax H3 Context-IR 提示词增强。

每个样本目录结构：
    <sample>/condition/prompt.txt
    <sample>/condition/references/NN_*.{png,jpg,...}   按文件名排序，对应 @图片1、@图片2 ...

输出：
    <out>/<sample>/enhanced_prompt.txt   增强后的提示词
    <out>/<sample>/task.json             完整的任务查询结果（含 usage）
    <out>/<sample>/request.json          实际发送的请求体（file_id 形式）

用法：
    MINIMAX_API_KEY=... uv run python -m ablation.context_ir [--seed seed] [--out output/context_ir]
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
TERMINAL = {"succeeded", "failed", "cancelled"}


class Client:
    def __init__(self, api_key: str, base: str):
        self.base = base.rstrip("/")
        self.auth = {"Authorization": f"Bearer {api_key}"}
        self._file_ids: dict[str, int] = {}
        self._lock = threading.Lock()

    def _request(self, method: str, path: str, **kw) -> dict:
        for attempt in range(5):
            r = requests.request(method, self.base + path, headers=self.auth, timeout=300, **kw)
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(2 ** attempt * 3)
                continue
            if r.status_code != 200:
                raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text}")
            return r.json()
        raise RuntimeError(f"{method} {path} -> retries exhausted: {r.status_code} {r.text}")

    def upload(self, path: Path) -> int:
        """上传参考素材，同一内容的文件只传一次。"""
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with self._lock:
            if digest in self._file_ids:
                return self._file_ids[digest]
        with path.open("rb") as f:
            resp = self._request("POST", "/v1/files/upload",
                                 data={"purpose": "video_generation_input"}, files={"file": f})
        if resp.get("base_resp", {}).get("status_code") != 0:
            raise RuntimeError(f"upload {path} failed: {resp}")
        file_id = resp["file"]["file_id"]
        with self._lock:
            self._file_ids[digest] = file_id
        return file_id

    def context_ir(self, payload: dict, poll_interval: float = 5, timeout: float = 900) -> dict:
        task_id = self._request("POST", "/v2/h3_context_ir", json=payload)["task_id"]
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(poll_interval)
            task = self._request("GET", f"/v2/query/video_generation/{task_id}")["task"]
            if task["status"] in TERMINAL:
                return task
        raise TimeoutError(f"task {task_id} not finished in {timeout}s")


def build_payload(client: Client, sample: Path, duration: int, ratio: str) -> dict:
    cond = sample / "condition"
    prompt = (cond / "prompt.txt").read_text(encoding="utf-8").strip()
    refs = sorted(p for p in (cond / "references").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    content = [{"type": "text", "text": prompt}]
    for ref in refs:
        content.append({"type": "image_url",
                        "image_url": {"url": f"mm_file://{client.upload(ref)}"},
                        "role": "reference_image"})
    return {"model": "MiniMax-H3", "content": content, "duration": duration, "ratio": ratio}


def process(client: Client, sample: Path, out_root: Path, duration: int, ratio: str) -> str:
    out = out_root / sample.name
    if (out / "enhanced_prompt.txt").exists():
        return f"{sample.name}: skipped (already done)"
    payload = build_payload(client, sample, duration, ratio)
    task = client.context_ir(payload)
    out.mkdir(parents=True, exist_ok=True)
    (out / "request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    if task["status"] != "succeeded":
        return f"{sample.name}: {task['status']} {task.get('error')}"
    (out / "enhanced_prompt.txt").write_text(task["content"]["prompt"], encoding="utf-8")
    return f"{sample.name}: succeeded, {len(payload['content']) - 1} refs, {task['usage'].get('total_tokens')} tokens"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=Path, default=Path("seed"))
    ap.add_argument("--out", type=Path, default=Path("output/context_ir"))
    ap.add_argument("--duration", type=int, default=10, help="4-15 秒")
    ap.add_argument("--ratio", default="adaptive")
    ap.add_argument("--base", default=os.environ.get("MINIMAX_BASE", "https://api.minimaxi.com"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("samples", nargs="*", help="只处理指定样本名，默认全部")
    args = ap.parse_args()

    client = Client(os.environ["MINIMAX_API_KEY"], args.base)
    samples = sorted(p for p in args.seed.iterdir() if (p / "condition" / "prompt.txt").exists())
    if args.samples:
        samples = [p for p in samples if p.name in args.samples]

    failed = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(process, client, s, args.out, args.duration, args.ratio): s for s in samples}
        for fut in as_completed(futs):
            try:
                print(fut.result(), flush=True)
            except Exception as e:
                failed += 1
                print(f"{futs[fut].name}: ERROR {e}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
