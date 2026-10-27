"""Dependency-free, reproducible qualification tools. No SSH credentials stored here."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from appliance import __version__

LABEL = "ai-appliance.qualification"
NAME = "ai-appliance-v01-model"


def run(args, timeout=60):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"ok": p.returncode == 0, "returncode": p.returncode,
                "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": str(e)}


def checked(args, timeout=60):
    result = run(args, timeout)
    if not result["ok"]:
        raise RuntimeError(f"{args[0]} failed: {result['stderr'] or result['stdout']}")
    return result["stdout"]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_devices(output):
    """310P npu-smi has one chip header then one physical/logical mapping line."""
    devices, pending = [], None
    for line in output.splitlines():
        head = re.match(r"\|\s*(\d+)\s+(310P\w*)\s*\|\s*(\w+)", line)
        if head:
            pending = {"card_id": int(head[1]), "chip": head[2], "health": head[3]}
            continue
        mapping = re.match(r"\|\s*(\d+)\s+(\d+)\s*\|\s*([\da-fA-F:.]+)", line)
        if pending and mapping:
            memory = re.search(r"(\d+)\s*/\s*(\d+)\s*\|", line)
            pending.update(chip_id=int(mapping[1]), device_id=int(mapping[2]), pci=mapping[3])
            if memory:
                pending.update(memory_used_mb=int(memory[1]), memory_total_mb=int(memory[2]))
            devices.append(pending)
            pending = None
    return devices


def parse_npu_processes(output):
    processes = []
    for line in output.splitlines():
        match = re.match(r"\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+)\|\s*(\d+)\s*\|", line)
        if match:
            processes.append({"card_id": int(match[1]), "chip_id": int(match[2]),
                              "pid": int(match[3]), "name": match[4].strip(),
                              "memory_mb": int(match[5])})
    return processes


def require_idle_npus():
    """This release qualifies one model on an idle host, never stops other workloads."""
    output = checked([shutil.which("npu-smi") or "/usr/local/sbin/npu-smi", "info"], 60)
    devices = parse_devices(output)
    if not devices or any(d["health"] != "OK" for d in devices):
        raise RuntimeError("NPU inventory missing or unhealthy; run doctor first")
    if "Process id" not in output:
        raise RuntimeError("Cannot verify NPU process table; refusing qualification start")
    processes = parse_npu_processes(output)
    if processes:
        raise RuntimeError("Qualification requires idle NPUs; active workloads: " + json.dumps(processes))


def doctor(args):
    npu = run([shutil.which("npu-smi") or "/usr/local/sbin/npu-smi", "info"], 120)
    report = {"version": __version__, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "host": platform.node(), "architecture": platform.machine(), "kernel": platform.release(),
              "os_release": Path("/etc/os-release").read_text() if Path("/etc/os-release").exists() else "",
              "cpu": run(["lscpu"]), "memory": run(["free", "-b"]),
              "disk": run(["df", "-B1", str(Path(args.output).resolve().parent)]),
              "npu": npu, "devices": parse_devices(npu["stdout"]),
              "docker": run(["docker", "version", "--format", "{{json .}}"]),
              "containers": run(["docker", "ps", "--format", "{{json .}}"]),
              "processes": run(["ps", "-eo", "user,pid,comm"]), "ascend_versions": {}}
    for path in ["/usr/local/Ascend/driver/version.info", "/usr/local/Ascend/version.info",
                 "/usr/local/Ascend/cann-8.5.0/opp/version.info"]:
        try:
            report["ascend_versions"][path] = Path(path).read_text()
        except OSError as e:
            report["ascend_versions"][path] = str(e)
    report["qualification"] = "inventory_only_not_inference_certification"
    write_json(args.output, report)
    print(json.dumps({"report": args.output, "devices": report["devices"],
                      "npu_ok": npu["ok"]}, ensure_ascii=False))
    return 0 if (npu["ok"] and report["devices"] and
                 all(d["health"] == "OK" for d in report["devices"])) else 1


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


def repeated_prose(text, width=12, count=3):
    """Narrow diagnostic for the qualification prose prompt, not a content filter."""
    text = re.sub(r"\s+", "", text)
    for index in range(max(0, len(text) - width + 1)):
        phrase = text[index:index + width]
        if text.count(phrase) >= count:
            return phrase
    return None


def bootstrap_download(args):
    """v0.1 qualification downloader; the full installer is planned for v0.3."""
    root = Path(args.directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    listing = fetch_json("https://modelscope.cn/api/v1/models/" + args.repo +
                         "/repo/files?Revision=master&Recursive=true")
    if listing.get("Code") != 200:
        raise RuntimeError("ModelScope listing failed")
    files = [f for f in listing["Data"]["Files"] if f.get("Type") == "blob" and
             (f["Path"].endswith((".json", ".safetensors", ".txt", ".model", ".jinja")) or f["Path"] == "LICENSE")]
    if not any(f["Path"].endswith(".safetensors") for f in files):
        raise RuntimeError("No safetensors weights in model listing")
    free = shutil.disk_usage(root).free
    if sum(f["Size"] for f in files) * 1.2 > free:
        raise RuntimeError("Insufficient free disk space")
    manifest = {"source": "modelscope", "repo": args.repo, "files": [],
                "certified": False, "tool_version": __version__}
    for entry in files:
        relative = Path(entry["Path"])
        target = (root / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(root):
            raise ValueError("Unsafe model path")
        expected = entry.get("Sha256", "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Missing trustworthy file checksum")
        if not target.exists() or target.stat().st_size != entry["Size"] or digest(target) != expected:
            url = ("https://modelscope.cn/api/v1/models/" + args.repo + "/repo?" +
                   urllib.parse.urlencode({"Revision": entry["Revision"], "FilePath": entry["Path"]}))
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            print(f"Downloading {entry['Path']} ({entry['Size']} bytes)", flush=True)
            with urllib.request.urlopen(url, timeout=120) as source, part.open("wb") as dest:
                shutil.copyfileobj(source, dest, 4 * 1024 * 1024)
            if part.stat().st_size != entry["Size"] or digest(part) != expected:
                raise RuntimeError(f"Checksum mismatch: {entry['Path']}")
            part.replace(target)
        manifest["files"].append({"path": entry["Path"], "size": entry["Size"],
                                  "sha256": expected, "revision": entry["Revision"]})
    write_json(root / "appliance-manifest.json", manifest)
    print(f"File validation passed: {len(files)} files; inference not yet certified", flush=True)


def validate_model(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "appliance-manifest.json").read_text())
    for entry in manifest["files"]:
        target = (root / entry["path"]).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError("Missing or unsafe model file")
        if target.stat().st_size != entry["size"] or digest(target) != entry["sha256"]:
            raise ValueError(f"Model checksum mismatch: {entry['path']}")
    if not (root / "config.json").is_file() or not (root / "tokenizer_config.json").is_file():
        raise ValueError("Missing model/tokenizer configuration")
    if not any(f["path"].endswith(".safetensors") for f in manifest["files"]):
        raise ValueError("Missing weights")
    return root


def container_config():
    result = run(["docker", "inspect", NAME])
    if not result["ok"]:
        # Only absence is treated as absence, not daemon/permission failures.
        message = result["stderr"].lower()
        if "no such object:" in message or "no such container:" in message:
            return None
        raise RuntimeError(result["stderr"])
    obj = json.loads(result["stdout"])[0]
    if obj["Config"].get("Labels", {}).get(LABEL) != "v0.1.0":
        raise RuntimeError("Refusing to modify container without qualification label")
    return obj


def start(args):
    if args.device != 0:
        raise ValueError("v0.1 qualifies device 0 only; other device mappings are not certified")
    if container_config() is not None:
        raise RuntimeError("Qualification container already exists; inspect or stop first")
    require_idle_npus()
    root = validate_model(args.model)
    if "@sha256:" not in args.image:
        raise ValueError("Use a digest-pinned image, not a mutable tag")
    image = json.loads(checked(["docker", "image", "inspect", args.image]))[0]
    if image["Architecture"] != "arm64":
        raise ValueError("This profile requires an ARM64 image")
    devices = [f"/dev/davinci{args.device}", "/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc"]
    command = ["docker", "run", "-d", "--name", NAME, "--label", LABEL + "=v0.1.0",
               "--restart", "no", "--shm-size", "2g", "--security-opt", "no-new-privileges:true",
               "--log-opt", "max-size=20m", "--log-opt", "max-file=3",
               "-p", f"127.0.0.1:{args.port}:8000", "-v", f"{root}:/models/test:ro"]
    for device in devices:
        if not Path(device).exists():
            raise ValueError(f"Missing device: {device}")
        command += ["--device", device]
    for src, dest in [("/usr/local/Ascend/driver/lib64", "/usr/local/Ascend/driver/lib64"),
                      ("/usr/local/Ascend/driver/version.info", "/usr/local/Ascend/driver/version.info"),
                      ("/etc/ascend_install.info", "/etc/ascend_install.info"),
                      ("/usr/local/dcmi", "/usr/local/dcmi")]:
        if Path(src).exists():
            command += ["-v", f"{src}:{dest}:ro"]
    command += ["-e", f"ASCEND_RT_VISIBLE_DEVICES={args.device}",
                "-e", "VLLM_USE_MODELSCOPE=false", "-e", "HF_HUB_OFFLINE=1",
                "-e", "VLLM_NO_USAGE_STATS=1", "-e", "DO_NOT_TRACK=1",
                "-e", "HF_HUB_DISABLE_TELEMETRY=1",
                "-e", "TRANSFORMERS_OFFLINE=1", "--entrypoint", "/bin/bash", args.image,
                "-lc", "source /usr/local/Ascend/ascend-toolkit/set_env.sh && exec vllm serve /models/test "
                "--host 0.0.0.0 --port 8000 --served-model-name qualification "
                "--dtype float16 --max-model-len 512 --max-num-seqs 1 "
                "--gpu-memory-utilization 0.35 --enforce-eager "
                "--additional-config '{\"ascend_compilation_config\":{\"enable_npugraph_ex\":false}}' "
                "--no-enable-prefix-caching --no-enable-chunked-prefill --disable-cascade-attn"]
    print(checked(command, 120).strip())


def stop(args):
    obj = container_config()
    if obj is None:
        print("Already absent")
        return
    print(checked(["docker", "stop", "--timeout", "30", NAME], 60).strip())
    # Preserve logs before removing only our labelled qualification container.
    result = run(["docker", "logs", NAME])
    if not result["ok"]:
        raise RuntimeError(result["stderr"])
    logs = result["stdout"] + result["stderr"]
    if args.logs:
        Path(args.logs).parent.mkdir(parents=True, exist_ok=True)
        Path(args.logs).write_text(logs)
    checked(["docker", "rm", NAME])


def smoke(args):
    parsed = urllib.parse.urlparse(args.base_url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("Qualification probe only accepts localhost HTTP")
    base = args.base_url.rstrip("/")
    request = {"model": "qualification", "messages": [{"role": "user", "content": args.prompt}],
               "temperature": 0, "max_tokens": 64, "stream": True,
               "chat_template_kwargs": {"enable_thinking": False},
               "stream_options": {"include_usage": True}}
    readiness_begin = time.monotonic()
    deadline, last = readiness_begin + args.wait, ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as response:
                if response.status == 200:
                    break
        except OSError as e:
            last = str(e)
        if getattr(args, "watch_container", False):
            obj = container_config()
            state = obj.get("State", {}) if obj else {"Status": "absent"}
            if state.get("Status") not in ("running", "created", "restarting"):
                error = f"Engine unavailable during readiness: {state}"
                write_json(args.output, {"tool_version": __version__, "passed": False,
                                        "phase": "readiness", "error": error})
                raise RuntimeError(error)
        time.sleep(2)
    else:
        raise RuntimeError("Readiness timeout: " + last)
    readiness_seconds = time.monotonic() - readiness_begin
    with urllib.request.urlopen(base + "/v1/models", timeout=10) as response:
        models = json.load(response)
    if "qualification" not in [m["id"] for m in models.get("data", [])]:
        raise RuntimeError("Expected model absent from model list")
    ordinary = dict(request, stream=False)
    ordinary.pop("stream_options")
    normal_req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(ordinary).encode(),
                                       {"Content-Type": "application/json"})
    normal_begin = time.monotonic()
    with urllib.request.urlopen(normal_req, timeout=180) as response:
        normal = json.load(response)
    normal_elapsed = time.monotonic() - normal_begin
    normal_text = normal.get("choices", [{}])[0].get("message", {}).get("content", "")
    normal_ok = bool(normal_text.strip()) and (not args.expect or args.expect in normal_text)
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(request).encode(),
                                 {"Content-Type": "application/json"})
    begin, first, chunks, usage, done = time.monotonic(), None, [], None, False
    with urllib.request.urlopen(req, timeout=180) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            item = json.loads(payload)
            if item.get("error"):
                raise RuntimeError(str(item["error"]))
            if item.get("usage"):
                usage = item["usage"]
            for choice in item.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    first = first or time.monotonic()
                    chunks.append(content)
    elapsed = time.monotonic() - begin
    output = "".join(chunks)
    ok = normal_ok and done and bool(output.strip()) and (not args.expect or args.expect in output)
    repetition = None
    if getattr(args, "reject_repetition", False):
        repetition = repeated_prose(output) or repeated_prose(normal_text)
        ok = ok and repetition is None
    report = {"tool_version": __version__, "passed": ok, "model": "qualification",
              "readiness_wait_seconds": readiness_seconds,
              "request": request, "models": models, "response": output, "stream_completed": done,
              "elapsed_seconds": elapsed, "first_content_seconds": first - begin if first else None,
              "usage": usage, "measurement": "single_request_not_capacity_benchmark"}
    report["non_streaming"] = {"passed": normal_ok, "elapsed_seconds": normal_elapsed,
                               "response": normal, "text": normal_text}
    report["prose_repetition_check"] = {"enabled": getattr(args, "reject_repetition", False),
                                       "repeated_phrase": repetition}
    if usage and usage.get("completion_tokens"):
        report["completion_tokens_per_wall_second"] = usage["completion_tokens"] / elapsed
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description="AI Appliance v0.1 qualification")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor")
    p.add_argument("--output", default="artifacts/inventory.json")
    p.set_defaults(func=doctor)
    p = sub.add_parser("bootstrap-download", help="qualification only, not production installer")
    p.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    p.add_argument("--directory", required=True)
    p.set_defaults(func=bootstrap_download)
    p = sub.add_parser("start")
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--device", type=int, choices=[0], default=0,
                   help="v0.1 certifies the device-0 direct-container mapping only")
    p.add_argument("--port", type=int, default=18080)
    p.set_defaults(func=start)
    p = sub.add_parser("stop")
    p.add_argument("--logs")
    p.set_defaults(func=stop)
    p = sub.add_parser("smoke")
    p.add_argument("--base-url", default="http://127.0.0.1:18080")
    p.add_argument("--wait", type=int, default=600)
    p.add_argument("--prompt", default="请只回答：中国的首都是哪里？")
    p.add_argument("--expect", default="北京")
    p.add_argument("--output", default="artifacts/smoke.json")
    p.add_argument("--reject-repetition", action="store_true",
                   help="quality gate for the fixed long-prose qualification prompt")
    p.add_argument("--watch-container", action="store_true",
                   help="fail immediately if the qualification engine exits (requires Docker access)")
    p.set_defaults(func=smoke)
    args = parser.parse_args()
    try:
        return args.func(args) or 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

