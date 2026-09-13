#!/usr/bin/env python3
"""
kaggle-tpu-lab launcher — serve Qwen3.8-27B on a free Kaggle TPU from your terminal.

    python launch.py serve                 # push the kernel and watch it come up
    python launch.py serve --reasoning-effort medium --mtp 3
    python launch.py status                # one-shot status + recent events
    python launch.py stop                  # kill the TPU session (and the local proxy)

When the endpoint goes live (seen by `serve` or `status`), proxy.py starts
automatically on the fixed local address http://127.0.0.1:9000 and injects the
current remote API key. Clients only ever need the fixed local contract —
base_url http://127.0.0.1:9000/v1 and the fixed local key (or any key, or no
key: the proxy ignores what clients send and substitutes the real remote key).
Every new session gets a new remote URL/key, but the local address/key stay
the same — a leftover proxy is found and replaced (logs go to
~/.kaggle-tpu-lab-proxy.log). It keeps running after you detach; `stop`
terminates it together with the kernel.

Requires the Kaggle CLI, authenticated:  pip install kaggle   (see README).
Only the Python standard library is used here.
"""
import argparse
import json
import os
import re
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_SRC = HERE / "kernel" / "serve_qwen38.py"
STATE_FILE = Path.home() / ".kaggle-tpu-lab.json"
PROXY_SCRIPT = HERE / "proxy.py"
PROXY_PID_FILE = Path.home() / ".kaggle-tpu-lab-proxy.pid"
PROXY_LOG_FILE = Path.home() / ".kaggle-tpu-lab-proxy.log"
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 9000
# 客户端固定使用的本地 key（proxy 会忽略它，替换成远端真实 key）
LOCAL_PROXY_KEY = "sk-kaggle-tpu-local"

_proxy_proc = None  # Popen handle of proxy.py, when started from this process

WEIGHTS_DATASET = "rahim3/qwen3-8-27b-bf16"
ENV_DATASET = "rahim3/qwen38-tpu-env-v5e8"   # XLA compile cache + cloudflared + manifest

# Friendly one-liners for each phase the kernel publishes.
PHASE_TEXT = {
    "install":            "Building the Python runtime with uv (~30 s)...",
    "installed":          "Runtime ready.",
    "mtp-patch-applied":  "MTP state-rollback patch applied.",
    "mtp-patch-failed":   "MTP patch did not apply — speculative decoding disabled for safety.",
    "cache-restored":     None,  # rendered below (depends on config coverage)
    "cache-missing":      "No compile cache found — cold compile, add ~10 min.",
    "weights-mounted":    "Weights found mounted (no download needed).",
    "weights-download":   "Downloading weights from Hugging Face (~5 min)...",
    "weights-downloaded": "Weights downloaded.",
    "server-launch":      "Starting vLLM — loading 55 GB of weights, then TPU graph compile...",
    "tunnel-url":         None,
    "compiling":          None,  # rendered with elapsed time below
    "serving":            "Server is HEALTHY.",
    "benchmark":          None,
    "ready":              None,
    "heartbeat":          None,
    "failed":             None,
    "auto-shutdown":      "Keepalive window ended — kernel shut down cleanly.",
    "stopped":            "Server exited unexpectedly.",
}


def kaggle(*args, capture=True):
    cmd = [sys.executable, "-m", "kaggle", *args]
    r = subprocess.run(cmd, capture_output=capture, text=True)
    return r


def say(msg):
    print(time.strftime("[%H:%M] "), msg, flush=True)


def check_auth():
    r = kaggle("kernels", "list", "-m", "--page-size", "1")
    if r.returncode != 0:
        sys.exit("Kaggle CLI is not working or not authenticated.\n"
                 "Install with `pip install kaggle`, then put your API token in place\n"
                 "(https://www.kaggle.com/settings -> Create New Token).\n\n"
                 f"Error was:\n{(r.stderr or r.stdout).strip()}")


def kaggle_username(cli_arg):
    if cli_arg:
        return cli_arg
    r = kaggle("config", "view")
    m = re.search(r"username[:=]\s*(\S+)", (r.stdout or "") + (r.stderr or ""))
    if m and m.group(1) not in ("None", "-"):
        return m.group(1).strip("'\"")
    sys.exit("Could not detect your Kaggle username — pass it with --user <name>.")


def _port_in_use(host, port):
    """True if something is actively listening on host:port.

    SO_REUSEADDR keeps lingering TIME_WAIT sockets (left by clients that just
    disconnected) from looking like an active listener.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def _pid_cmdline(pid):
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [a.decode(errors="replace") for a in raw.split(b"\0") if a]


def _cmdline_str(pid):
    return " ".join(_pid_cmdline(pid))[:80] or "unknown command"


def _is_proxy_process(pid):
    """True if pid is a python process running this directory's proxy.py."""
    for arg in _pid_cmdline(pid)[1:]:
        if not arg.endswith(PROXY_SCRIPT.name):
            continue
        p = Path(arg)
        if p.is_absolute():
            if p.resolve() == PROXY_SCRIPT:
                return True
        else:
            try:
                cwd = Path(f"/proc/{pid}/cwd").resolve()
            except OSError:
                continue
            if (cwd / p).resolve() == PROXY_SCRIPT:
                return True
    return False


def _listening_pid(host, port):
    """Pid of the process listening on host:port, or None.

    Linux-only best effort (reads /proc); returns None if it cannot tell.
    """
    try:
        addr = "%08X" % struct.unpack("<I", socket.inet_aton(host))[0]
    except OSError:
        return None
    want = f"{addr}:{port:04X}"
    try:
        lines = Path("/proc/net/tcp").read_text().splitlines()[1:]
    except OSError:
        return None
    inode = None
    for line in lines:
        f = line.split()
        # sl local_address rem_address st tx_queue:rx_queue ... inode
        if len(f) > 9 and f[1] == want and f[3] == "0A":  # 0A = LISTEN
            inode = f[9]
            break
    if inode is None:
        return None
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fds = list((entry / "fd").iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink(fd) == f"socket:[{inode}]":
                    return int(entry.name)
            except OSError:
                continue
    return None


def _read_proxy_pidfile():
    """(pid, endpoint, model) recorded in the pidfile, or (None, None, None)."""
    try:
        data = json.loads(PROXY_PID_FILE.read_text())
        return int(data["pid"]), data.get("endpoint"), data.get("model")
    except Exception:
        return None, None, None


def proxy_alive_pid():
    """Pid of a running proxy.py: this process' child, whatever is listening
    on the fixed port, or the pid from the pidfile — else None."""
    global _proxy_proc
    if _proxy_proc is not None and _proxy_proc.poll() is None:
        return _proxy_proc.pid
    pid = _listening_pid(PROXY_HOST, PROXY_PORT)
    if pid is not None and _is_proxy_process(pid):
        return pid
    pid, _, _ = _read_proxy_pidfile()
    if pid is not None:
        try:
            os.kill(pid, 0)
            if _is_proxy_process(pid):
                return pid
        except OSError:
            pass
    return None


def _kill_proxy_pid(pid):
    """Stop one proxy.py process and wait for the fixed port to come free."""
    global _proxy_proc
    try:
        os.kill(pid, 15)
    except OSError:
        pass
    if _proxy_proc is not None and _proxy_proc.pid == pid:
        try:
            _proxy_proc.wait(timeout=5)
        except Exception:
            pass
        _proxy_proc = None
    for _ in range(30):
        if not _port_in_use(PROXY_HOST, PROXY_PORT):
            break
        time.sleep(0.1)
    try:
        PROXY_PID_FILE.unlink()
    except OSError:
        pass


def start_proxy(endpoint, api_key=None, model=None):
    """Make sure proxy.py serves the live endpoint on the fixed local port.

    Called every time `serve`/`status` sees the endpoint go live. Any proxy
    left over from an earlier run — including one started by hand — is
    identified and replaced, so the local URL and the client key/model never
    change even though the remote URL / key / model do.

    The endpoint and model go on the command line; the API key is handed to
    the child through the REMOTE_API_KEY environment variable, so it never
    shows up in `ps` output (see proxy.py).
    """
    global _proxy_proc

    # Resolve key/model from the saved state first: the "already up" check
    # below compares against them, and an unresolved None would force a
    # pointless restart on every `status` call.
    try:
        st = json.loads(STATE_FILE.read_text())
    except Exception:
        st = {}
    api_key = api_key or st.get("api_key", "")
    model = model or st.get("model", "")

    pid = _listening_pid(PROXY_HOST, PROXY_PORT)
    if pid is not None:
        if not _is_proxy_process(pid):
            say(f"WARNING: port {PROXY_PORT} is held by pid {pid} "
                f"({_cmdline_str(pid)}) — not starting the local proxy. Free "
                "the port and re-run.")
            return
        saved_pid, saved_endpoint, saved_model = _read_proxy_pidfile()
        if (saved_pid == pid and saved_endpoint == endpoint
                and saved_model == model):
            say(f"Local proxy already up on http://{PROXY_HOST}:{PROXY_PORT} "
                f"(pid {pid})")
            return
        say(f"Replacing the local proxy on port {PROXY_PORT} "
            f"({saved_endpoint or 'unknown endpoint'}  ->  {endpoint})")
        _kill_proxy_pid(pid)
    elif _port_in_use(PROXY_HOST, PROXY_PORT):
        say(f"WARNING: port {PROXY_PORT} is already in use and its owner could "
            "not be identified — not starting the local proxy.")
        return

    if not api_key:
        say("WARNING: no API key available — skipping the local proxy.")
        return
    if not PROXY_SCRIPT.exists():
        say(f"WARNING: {PROXY_SCRIPT.name} not found next to launch.py — "
            "skipping the local proxy.")
        return

    # Remember the live endpoint/model: ntfy drops the ready event after ~12 h,
    # and `status` needs them to bring the proxy back up later in the session.
    update_state(endpoint=endpoint, api_key=api_key, model=model)

    say(f"Starting local proxy  http://{PROXY_HOST}:{PROXY_PORT}  ->  {endpoint}")
    log = None
    try:
        log = open(PROXY_LOG_FILE, "w")
    except OSError:
        pass
    argv = [sys.executable, "-u", str(PROXY_SCRIPT),  # -u: logs reach file now
            "--remote-url", endpoint,
            "--host", PROXY_HOST, "--port", str(PROXY_PORT),
            "--local-api-key", LOCAL_PROXY_KEY]
    if model:
        argv += ["--remote-model", model]
    env = {**os.environ, "REMOTE_API_KEY": api_key}
    _proxy_proc = subprocess.Popen(
        argv,
        env=env,
        stdout=log if log is not None else subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # keep serving after we detach / Ctrl-C / exit
    )
    if log is not None:
        log.close()
        say(f"Proxy log: {PROXY_LOG_FILE}")
    try:
        PROXY_PID_FILE.write_text(json.dumps(
            {"pid": _proxy_proc.pid, "endpoint": endpoint, "model": model}))
    except OSError:
        pass

    # Confirm it really bound the port (bad URL/key dies here).
    for _ in range(20):
        if _listening_pid(PROXY_HOST, PROXY_PORT) == _proxy_proc.pid:
            break
        if _proxy_proc.poll() is not None:
            break
        time.sleep(0.1)
    if _listening_pid(PROXY_HOST, PROXY_PORT) == _proxy_proc.pid:
        say(f"Local proxy is up on http://{PROXY_HOST}:{PROXY_PORT} "
            f"(pid {_proxy_proc.pid}"
            + (f", model {model}" if model else "") + ")")
    else:
        say(f"WARNING: the local proxy did not come up — see {PROXY_LOG_FILE}.")


def stop_proxy():
    """Terminate the local proxy, wherever it came from, if it is running."""
    pid = proxy_alive_pid()
    if pid is None:
        return
    _kill_proxy_pid(pid)
    say(f"Local proxy stopped (pid {pid}).")


def endpoint_alive(endpoint, api_key, timeout=8):
    """Best-effort probe: does the tunnel answer right now?

    ntfy keeps messages for ~12 h, so by the time `status` runs the `ready`
    event may be gone; a live probe tells us whether the saved endpoint is
    still worth proxying. Any sub-500 answer (even 401/404) means it is up.
    """
    req = urllib.request.Request(endpoint.rstrip("/") + "/models")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status < 500
    except urllib.error.HTTPError as e:
        return e.code < 500
    except Exception:
        return False


def cmd_serve(args):
    check_auth()
    user = kaggle_username(args.user)
    slug = args.slug
    topic = "ktl-" + uuid.uuid4().hex[:20]
    api_key = "sk-" + secrets.token_hex(16)

    cfg = {
        "ntfy_topic": topic,
        "api_key": api_key,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "mtp_tokens": args.mtp,
        "reasoning_effort_default": args.reasoning_effort,
        "keepalive_min": args.keepalive_min,
        "weights_dataset": args.weights_dataset,
    }
    if args.no_tools:
        cfg["tool_call_parser"] = ""
    if args.text_only:
        cfg["text_only"] = True
    if args.verbose:
        cfg["verbose"] = True
    if args.fast_start:
        cfg["fast_start"] = True

    src = KERNEL_SRC.read_text()
    src, n = re.subn(r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
                     f"CFG = {cfg!r}", src, count=1, flags=re.M)
    if n != 1:
        sys.exit("kernel/serve_qwen38.py is missing the __LAUNCHER_CONFIG__ line")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "serve_qwen38.py").write_text(src)
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug}",
            "title": slug,
            "code_file": "serve_qwen38.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "true",
            "enable_internet": "true",
            "dataset_sources": [args.weights_dataset, ENV_DATASET],
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1))
        say(f"Pushing kernel {user}/{slug} (TPU v5e-8)...")
        r = kaggle("kernels", "push", "-p", str(td))
        out = (r.stdout or "") + (r.stderr or "")
        if "successfully pushed" not in out:
            sys.exit(f"Push failed:\n{out.strip()}")
        for line in out.splitlines():
            if "not valid dataset sources" in line:
                say(f"WARNING: {line.strip()} — the kernel will still run, "
                    "but may need to download weights / compile cold.")

    STATE_FILE.write_text(json.dumps(
        {"kernel": f"{user}/{slug}", "topic": topic, "api_key": api_key}))
    say("Pushed. Kaggle takes a few minutes to provision the TPU and attach the "
        "datasets; the endpoint is usually live ~22 min after the kernel starts.")
    say("Watching progress (Ctrl-C is safe — the server keeps running; "
        "`python launch.py status` re-attaches, `... stop` kills it).")
    watch(f"{user}/{slug}", topic)


def read_events(topic, since):
    try:
        with urllib.request.urlopen(
                f"https://ntfy.sh/{topic}/json?poll=1&since={since}", timeout=15) as r:
            body = r.read().decode()
    except Exception:
        return []
    events = []
    for line in body.splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") != "message":
            continue
        try:
            events.append((e["time"], json.loads(e.get("message", "{}"))))
        except Exception:
            continue
    return events


def render_event(ev):
    phase = ev.get("phase", "?")
    if phase == "compiling":
        say(f"Loading / compiling... {ev.get('elapsed_s', 0) // 60} min elapsed "
            "(typically ~20 min with the env dataset, ~35 min without)")
    elif phase == "cache-restored":
        if ev.get("covers_this_config", True):
            say("XLA compile cache restored for this exact config — fast start.")
        else:
            say("XLA compile cache restored, but not for this config — its graphs "
                "compile cold (add ~10 min).")
    elif phase == "tunnel-url":
        say(f"Endpoint URL reserved: {ev.get('endpoint')}  (not live yet — wait for the banner)")
    elif phase == "serving":
        say(f"Server is HEALTHY after {ev.get('startup_secs', 0) // 60} min.")
    elif phase == "benchmark":
        say(f"Quick benchmark: {ev.get('decode_tok_s', '?')} tok/s single-stream decode "
            f"(sanity: {ev.get('sanity', '')!r})")
    elif phase == "ready":
        print("\n" + "=" * 66)
        print("  YOUR ENDPOINT IS LIVE")
        print(f"  remote   : {ev['endpoint']}   (每次会话都会变)")
        print(f"  remote key: {ev['api_key']}")
        print("-" * 66)
        print(f"  local    : http://{PROXY_HOST}:{PROXY_PORT}/v1   "
              "<- 客户端固定写这里")
        print(f"  local key: {LOCAL_PROXY_KEY}   "
              "(填它/随便填/不填都行，本地代理会忽略)")
        print(f"  model    : {ev['model']}   "
              "(客户端随便填/不填，本地代理自动替换)")
        print(f"             context: {ev.get('max_model_len', '?')}")
        print("=" * 66)
        start_proxy(ev["endpoint"], ev.get("api_key"), ev.get("model"))
        print(f"""
Try it (固定本地入口 + 固定本地 key，model 随便填，不需要知道远端值):
  curl http://{PROXY_HOST}:{PROXY_PORT}/v1/chat/completions \\
    -H "Authorization: Bearer {LOCAL_PROXY_KEY}" \\
    -H "Content-Type: application/json" -d '{{
      "model": "whatever",
      "messages": [{{"role": "user", "content": "Hello!"}}],
      "chat_template_kwargs": {{"reasoning_effort": "low"}}
    }}'
  # model 填什么都不影响：proxy 会替换成远端真实 model

客户端配置（Claude Code / Codex CLI / opencode 等）只写一次：
  base_url = http://{PROXY_HOST}:{PROXY_PORT}/v1
  api_key  = {LOCAL_PROXY_KEY}   （或任意值：本地代理忽略它，
                                  并在转发时替换成上面的远端 key）
  model    = {ev['model']}       （或任意值/留空：本地代理自动替换）
远端 URL/key/model 每次会话都变，客户端无需改动。
""")
        say(f"The kernel keeps serving for up to {ev.get('keepalive_min', '?')} min. "
            "Ctrl-C here does NOT stop it; use `python launch.py stop`.")
    elif phase == "heartbeat":
        say(f"Still serving ({ev.get('up_min', '?')} min up) — {ev.get('endpoint', '')}")
    elif phase == "failed":
        say(f"FAILED at step {ev.get('step', '?')}.")
        if ev.get("tail"):
            print("--- last server output ---")
            print(ev["tail"])
        say("Full log: `python launch.py status` after the kernel exits, or the "
            "kernel page on kaggle.com.")
    else:
        text = PHASE_TEXT.get(phase)
        say(text if text else f"{phase} {json.dumps({k: v for k, v in ev.items() if k != 'phase'})}")


def watch(kernel, topic):
    since = int(time.time()) - 600
    last_status = None
    seen_boot = False
    try:
        while True:
            for ts, ev in read_events(topic, since):
                since = max(since, ts)
                seen_boot = True
                render_event(ev)
                if ev.get("phase") in ("failed", "auto-shutdown", "stopped"):
                    stop_proxy()
                    return
            since = max(since, int(time.time()) - 1) if seen_boot else since
            r = kaggle("kernels", "status", kernel)
            out = (r.stdout or "") + (r.stderr or "")
            status = _parse_kernel_status(out)
            if status != last_status:
                if status == "QUEUED":
                    say("Kaggle: queued — waiting for a TPU v5e-8 slot...")
                elif status == "RUNNING" and not seen_boot:
                    say("Kaggle: provisioning the VM and attaching datasets "
                        "(a few minutes)...")
                elif _kernel_ended(status):
                    say(f"Kernel finished with status {status}.")
                    stop_proxy()
                    return
                last_status = status
            time.sleep(30)
    except KeyboardInterrupt:
        say("Detached. The kernel keeps running — `python launch.py status` to "
            "re-attach, `python launch.py stop` to kill it. The local proxy "
            f"(if started) keeps serving at http://{PROXY_HOST}:{PROXY_PORT}.")


def cmd_build_env(args):
    """Maintainer flow. When the kernel finishes:
        kaggle kernels output <user>/<slug> -p bundle_out
        then create/version the dataset from bundle_out/bundle (see README)."""
    check_auth()
    user = kaggle_username(args.user)
    topic = "ktl-" + uuid.uuid4().hex[:20]
    cfg = {"build_bundle": True, "ntfy_topic": topic, "weights_dataset": args.weights_dataset}
    src = KERNEL_SRC.read_text()
    src, n = re.subn(r"^CFG = None  # __LAUNCHER_CONFIG__.*$",
                     f"CFG = {cfg!r}", src, count=1, flags=re.M)
    if n != 1:
        sys.exit("kernel/serve_qwen38.py is missing the __LAUNCHER_CONFIG__ line")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "build_env.py").write_text(src)
        (td / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{args.slug}", "title": args.slug, "code_file": "build_env.py",
            "language": "python", "kernel_type": "script", "is_private": "true",
            "enable_gpu": "false", "enable_tpu": "true", "enable_internet": "true",
            "dataset_sources": [args.weights_dataset],
            "competition_sources": [], "kernel_sources": [], "model_sources": [],
        }, indent=1))
        r = kaggle("kernels", "push", "-p", str(td))
        out = (r.stdout or "") + (r.stderr or "")
        if "successfully pushed" not in out:
            sys.exit(f"Push failed:\n{out.strip()}")
    STATE_FILE.write_text(json.dumps({"kernel": f"{user}/{args.slug}", "topic": topic,
                                      "api_key": ""}))
    say(f"Pushed {user}/{args.slug}. It serves each config once (~1.5 h total) and "
        "leaves xla_cache.tar / cloudflared / manifest.json in its output.")
    watch(f"{user}/{args.slug}", topic)


def load_state():
    if not STATE_FILE.exists():
        sys.exit("No launch state found — run `python launch.py serve` first.")
    return json.loads(STATE_FILE.read_text())


def update_state(**fields):
    """Merge fields into the launch state file (best effort)."""
    try:
        st = json.loads(STATE_FILE.read_text())
    except Exception:
        st = {}
    st.update({k: v for k, v in fields.items() if v is not None})
    try:
        STATE_FILE.write_text(json.dumps(st))
    except OSError:
        pass


def _parse_kernel_status(out):
    m = re.search(r'"KernelWorkerStatus\.(\w+)"', out)
    return m.group(1) if m else "UNKNOWN"


def _kernel_ended(status):
    # Kaggle reports "CANCEL_ACKNOWLEDGED"; normalize before comparing.
    return status.replace("_", "") in ("ERROR", "COMPLETE", "CANCELACKNOWLEDGED")


def cmd_status(args):
    st = load_state()
    say(f"Kernel: {st['kernel']}")
    r = kaggle("kernels", "status", st["kernel"])
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if r.returncode != 0 and "No module named kaggle" in out:
        sys.exit(f"The Kaggle CLI is not importable by this Python:\n"
                 f"  {sys.executable}\n"
                 "Install it there, or run with an interpreter that has it, e.g.\n"
                 f"  {HERE / '.venv' / 'bin' / 'python'} launch.py status -f")
    say(out)
    status = _parse_kernel_status(out)

    events = read_events(st["topic"], int(time.time()) - 24 * 3600)
    tail = events[-8:]
    for _, ev in tail:
        render_event(ev)

    ready_ev = next((ev for _, ev in reversed(events)
                     if ev.get("phase") == "ready"), None)
    endpoint = api_key = model = None
    if ready_ev:
        endpoint = ready_ev["endpoint"]
        api_key = ready_ev.get("api_key") or st.get("api_key")
        model = ready_ev.get("model") or st.get("model")
        if not any(ev.get("phase") == "ready" for _, ev in tail):
            # Older than the last-8 replay above; start the proxy explicitly.
            say(f"API key: {api_key}")
            start_proxy(endpoint, api_key, model)
    elif st.get("endpoint"):
        # ntfy expired the ready event, but we remembered the endpoint when it
        # first went live — use it if it still answers.
        if endpoint_alive(st["endpoint"], st.get("api_key")):
            endpoint = st["endpoint"]
            api_key = st.get("api_key")
            model = st.get("model")
            say(f"Endpoint (from saved state, ntfy event expired): {endpoint}")
            start_proxy(endpoint, api_key, model)
        else:
            say(f"Saved endpoint {st['endpoint']} does not answer — that "
                "session looks finished.")

    if endpoint is None:
        say("No live endpoint to proxy right now — start one with "
            "`python launch.py serve`.")

    if args.follow:
        if endpoint is None and _kernel_ended(status):
            say(f"Nothing to follow: kernel status is {status}.")
        else:
            watch(st["kernel"], st["topic"])


def cmd_stop(args):
    stop_proxy()
    st = load_state()
    say(f"Deleting kernel {st['kernel']} (terminates the TPU session)...")
    p = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "delete",
                        st["kernel"]], input="yes\n", capture_output=True, text=True)
    say((p.stdout + p.stderr).strip() or "done")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="push the serving kernel and watch it come up")
    s.add_argument("--user", help="Kaggle username (auto-detected if possible)")
    s.add_argument("--slug", default="qwen38-tpu-serve", help="kernel name")
    s.add_argument("--max-model-len", type=int, default=262144,
                   help="context length (default: native 262k; use 131072 with "
                        "--max-num-seqs 16 for max multi-stream throughput)")
    s.add_argument("--max-num-seqs", type=int, default=4)
    s.add_argument("--mtp", type=int, default=3,
                   help="MTP speculative tokens (0 disables). +34%% decode in our A/B test; made "
                        "lossless by the bundled GDN state-rollback patch "
                        "(verified 12/12 greedy exact-match)")
    s.add_argument("--reasoning-effort", default="xhigh",
                   choices=["xhigh", "medium", "low"],
                   help="server-side default; clients can still override per request")
    s.add_argument("--keepalive-min", type=int, default=480,
                   help="auto-shutdown after this many minutes of serving")
    s.add_argument("--weights-dataset", default=WEIGHTS_DATASET)
    s.add_argument("--no-tools", action="store_true",
                   help="disable tool-calling support")
    s.add_argument("--text-only", action="store_true",
                   help="skip the vision tower: ~8 min faster start, image inputs "
                        "then error out")
    s.add_argument("--verbose", action="store_true",
                   help="show every vLLM log line in the kernel log")
    s.add_argument("--fast-start", action="store_true",
                   help="skip TPU graph precompile: endpoint live in ~4 min (with the env "
                        "dataset), common request shapes are warmed right after; an "
                        "unusual request shape stalls ~1 min the first time")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("build-env", help="(maintainers) push a kernel that builds the "
                       "env dataset: venv + XLA cache + cloudflared")
    s.add_argument("--user", help="Kaggle username (auto-detected if possible)")
    s.add_argument("--slug", default="qwen38-env-bundle")
    s.add_argument("--weights-dataset", default=WEIGHTS_DATASET)
    s.set_defaults(fn=cmd_build_env)

    s = sub.add_parser("status", help="show current kernel status + recent events")
    s.add_argument("--follow", "-f", action="store_true", help="keep watching")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("stop", help="terminate the TPU session")
    s.set_defaults(fn=cmd_stop)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
