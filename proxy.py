#!/usr/bin/env python3
"""
Kaggle TPU Reverse Proxy

把本地请求（默认 http://127.0.0.1:9000）转发到 Kaggle TPU 远端 endpoint，
并自动注入远端 API key。

契约（客户端永远只认这几个固定值，远端变化全部由本代理兜住）：
    base_url = http://127.0.0.1:9000/v1
    本地 key = sk-kaggle-tpu-local    （其实填什么、甚至不填都行）
    model    = 随便填、甚至不填也行    （本代理会自动替换成远端真实 model）

客户端传来的 Authorization 会被本代理忽略并替换成当前会话的真实远端 key；
请求体 JSON 里的 "model" 也会被替换成真实远端 model —— 远端 URL / key / model
变化时，客户端的配置写死一次就行。

一般不需要手动运行：`python launch.py serve` / `status` 发现 endpoint 就绪
后会自动以正确参数启动本文件。手动运行示例：

    python proxy.py --remote-url https://xxx.trycloudflare.com \
                    --remote-api-key "sk-...."

    # 或者 API key 走环境变量（不要把密钥写进源码，本文件会被 git 跟踪）：
    export REMOTE_API_KEY="sk-...."
    python proxy.py --remote-url https://xxx.trycloudflare.com
"""

import argparse
import http.client
import json
import os
import random
import signal
import socket
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# 默认远端地址，仅作 fallback。launch.py 会自动传 --remote-url，
# 通常不需要改这里。
DEFAULT_REMOTE_URL = "https://toner-shopper-feelings-ending.trycloudflare.com"

# 本地固定入口：客户端永远只认这个地址，远端 URL 每次变都无所谓
LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 9000

# 本地固定 key：客户端填它 / 随便填 / 不填都行。
# 本文件一律忽略客户端传来的 Authorization，转发时替换成远端真实 key，
# 所以客户端配置永远不用随远端变化。
LOCAL_API_KEY = "sk-kaggle-tpu-local"

# 远端读超时（秒）：生成可能要几分钟，读阶段给足时间
REMOTE_TIMEOUT = 600

# 建连 + TLS 握手超时（秒）：必须短。以前和读超时共用一个 600s，握手一旦
# 卡住要等 10 分钟才重试（日志里的 "handshake operation timed out"）。
CONNECT_TIMEOUT = float(os.environ.get("PROXY_CONNECT_TIMEOUT", "15"))

# 允许的最大请求体（字节），超过则返回 413
MAX_REQUEST_BODY = 256 * 1024 * 1024

# ------------------------------------------------------------
# 抗瞬时故障：DNS 缓存 + 建连重试 + 连接复用
#
# cloudflared 隧道域名偶尔解析抖动（Errno -3），新建的 TLS 连接也偶尔握手
# 超时 / 被 reset（Errno 104）——每个请求都新建连接就会频繁踩到。所以：
#   1) DNS 解析结果缓存 DNS_CACHE_TTL 秒；
#   2) 建连/发送失败退避重试（请求体已在内存里，重发是安全的）；
#   3) 响应读完且连接可复用时放回池里 —— 后续请求省掉 DNS+TCP+TLS 握手，
#      这是减少 handshake/reset 错误的关键；
#   4) 建连用短超时，读响应用长超时。
# ------------------------------------------------------------
UPSTREAM_RETRIES = int(os.environ.get("PROXY_UPSTREAM_RETRIES", "3"))
RETRY_BACKOFF = 0.3        # 秒；按 2^n 递增，并叠加随机抖动
DNS_CACHE_TTL = float(os.environ.get("PROXY_DNS_TTL", "60"))
KEEPALIVE_MAX = int(os.environ.get("PROXY_KEEPALIVE_MAX", "4"))
KEEPALIVE_TTL = float(os.environ.get("PROXY_KEEPALIVE_TTL", "15"))

_dns_cache = {}
_dns_lock = threading.Lock()
_real_getaddrinfo = socket.getaddrinfo

_pool = []                 # [(conn, idle_since)]
_pool_lock = threading.Lock()


def _cached_getaddrinfo(host, port, *args, **kwargs):
    key = (host, port, args, tuple(sorted(kwargs.items())))
    now = time.time()
    with _dns_lock:
        hit = _dns_cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    infos = _real_getaddrinfo(host, port, *args, **kwargs)
    with _dns_lock:
        _dns_cache[key] = (now + DNS_CACHE_TTL, infos)
    return infos


def _drop_dns_cache(host):
    with _dns_lock:
        for key in [k for k in _dns_cache if k[0] == host]:
            del _dns_cache[key]


def install_dns_cache():
    """Route this process' name lookups through the TTL cache above."""
    socket.getaddrinfo = _cached_getaddrinfo


def _safe_close(conn):
    try:
        conn.close()
    except Exception:
        pass


def _tune_socket(sock):
    """TCP keepalive：更快发现死连接，也保住 NAT/防火墙映射。"""
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for name, value in (("TCP_KEEPIDLE", 30), ("TCP_KEEPINTVL", 10),
                            ("TCP_KEEPCNT", 3)):
            if hasattr(socket, name):
                sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, name), value)
    except OSError:
        pass


def _new_conn():
    """DNS + TCP + TLS 全用短超时；连上后切成读响应用的长超时。"""
    conn = http.client.HTTPSConnection(
        REMOTE.netloc, timeout=CONNECT_TIMEOUT, context=SSL_CONTEXT)
    try:
        conn.connect()
        if conn.sock is not None:
            _tune_socket(conn.sock)
            conn.sock.settimeout(REMOTE_TIMEOUT)
        return conn
    except Exception:
        _safe_close(conn)
        raise


def _pool_get():
    while True:
        with _pool_lock:
            if not _pool:
                return None
            conn, idle_since = _pool.pop()
        if time.time() - idle_since > KEEPALIVE_TTL or conn.sock is None:
            _safe_close(conn)
            continue
        return conn


def _pool_put(conn):
    if conn.sock is None:
        return
    with _pool_lock:
        if len(_pool) >= KEEPALIVE_MAX:
            overflow = conn
        else:
            _pool.append((conn, time.time()))
            overflow = None
    if overflow is not None:
        _safe_close(overflow)


def close_pool():
    with _pool_lock:
        conns, _pool[:] = list(_pool), []
    for conn, _ in conns:
        _safe_close(conn)


def _retry_pause(attempt, err):
    delay = RETRY_BACKOFF * (2 ** attempt)
    print(f"[proxy] upstream failure ({err}); retry {attempt + 1}/"
          f"{UPSTREAM_RETRIES - 1} in {delay:.1f}s", file=sys.stderr, flush=True)
    time.sleep(delay + random.uniform(0, delay))


def upstream_exchange(command, path, body, headers):
    """发送请求并取回响应，透明处理瞬时故障。返回 (conn, response)。

    - 建连/发送失败：重试（请求体已在内存里，安全）。
    - 复用连接上失败：它在空闲时已被对端关掉，换新连接重试是安全的。
    - 新连接在 getresponse 阶段失败：不重发，避免重复生成。
    - 证书校验失败：直接抛，不重试。
    """
    last = None
    for attempt in range(UPSTREAM_RETRIES):
        conn = _pool_get()
        reused = conn is not None
        if conn is None:
            try:
                conn = _new_conn()
            except ssl.SSLCertVerificationError:
                raise
            except (OSError, http.client.HTTPException) as e:
                last = e
                _drop_dns_cache(REMOTE.hostname)
                if attempt + 1 < UPSTREAM_RETRIES:
                    _retry_pause(attempt, e)
                    continue
                raise
        try:
            conn.request(command, path, body=body, headers=headers)
        except ssl.SSLCertVerificationError:
            _safe_close(conn)
            raise
        except (OSError, http.client.HTTPException) as e:
            _safe_close(conn)
            last = e
            _drop_dns_cache(REMOTE.hostname)
            if reused:
                close_pool()          # 别再拿到池里其它同样过期的连接
            if attempt + 1 < UPSTREAM_RETRIES:
                _retry_pause(attempt, e)
                continue
            raise
        try:
            return conn, conn.getresponse()
        except ssl.SSLCertVerificationError:
            _safe_close(conn)
            raise
        except (OSError, http.client.HTTPException) as e:
            _safe_close(conn)
            last = e
            if reused and attempt + 1 < UPSTREAM_RETRIES:
                print("[proxy] reused keep-alive connection was closed by the "
                      "peer; retrying on a fresh one", file=sys.stderr,
                      flush=True)
                close_pool()
                continue
            raise
    raise last


# ============================================================
# 以下一般不需要修改
# ============================================================

# 运行时配置：由 main() 根据命令行参数 / 环境变量填充
REMOTE_URL = ""
REMOTE_API_KEY = ""
REMOTE_MODEL = ""          # 当前会话真实 model；客户端填什么都替换成它
REMOTE = None
REMOTE_PATH_PREFIX = ""

# TLS context 全局复用，避免每个请求都重新加载系统 CA
SSL_CONTEXT = ssl.create_default_context()


def normalize_remote_url(url):
    return url.rstrip("/")


def build_upstream_path(client_path):
    """拼接远端 path 前缀与客户端 path，避免出现重复前缀。"""
    if not REMOTE_PATH_PREFIX:
        return client_path

    if (
        client_path == REMOTE_PATH_PREFIX
        or client_path.startswith(REMOTE_PATH_PREFIX + "/")
        or client_path.startswith(REMOTE_PATH_PREFIX + "?")
    ):
        return client_path

    return REMOTE_PATH_PREFIX + client_path


def apply_model(body):
    """把请求体 JSON 里的 "model" 换成远端真实 model（客户端不用管）。

    只对看起来像推理请求的 JSON 对象动手，其它 body 原样透传。
    """
    if not REMOTE_MODEL or not body:
        return body
    try:
        obj = json.loads(body)
    except Exception:
        return body
    if not isinstance(obj, dict):
        return body
    if not ({"model", "messages", "prompt", "input"} & set(obj)):
        return body
    obj["model"] = REMOTE_MODEL
    return json.dumps(obj, ensure_ascii=False).encode()


class BodyTooLarge(Exception):
    pass


class ProxyHandler(BaseHTTPRequestHandler):

    protocol_version = "HTTP/1.1"
    server_version = "KaggleTPUProxy/1.0"
    sys_version = ""

    def log_message(self, fmt, *args):
        print(
            f"[proxy] {self.client_address[0]} - {fmt % args}",
            flush=True,
        )

    def handle_one_request(self):
        """Override to swallow normal client disconnects cleanly."""
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            # Client closed the connection (very common with streaming clients)
            pass

    # ------------------------------------------------------------------
    # 读取请求体：同时支持 Content-Length 与 Transfer-Encoding: chunked
    # ------------------------------------------------------------------

    def read_chunked_body(self):
        chunks = []
        total = 0

        while True:
            size_line = self.rfile.readline(65536).strip()
            if not size_line:
                raise ValueError("missing chunk size")

            # 允许 chunk extension（size;ext=...）
            if b";" in size_line:
                size_line = size_line.split(b";", 1)[0]

            try:
                size = int(size_line, 16)
            except ValueError:
                raise ValueError("invalid chunk size")

            if size == 0:
                # 读取并丢弃 trailer
                while True:
                    line = self.rfile.readline(65536)
                    if line in (b"\r\n", b"\n", b""):
                        break
                break

            total += size
            if total > MAX_REQUEST_BODY:
                raise BodyTooLarge()

            chunk = self.rfile.read(size)
            if len(chunk) != size:
                raise ValueError("truncated chunk")

            chunks.append(chunk)

            # 每个 chunk 数据后的 CRLF
            self.rfile.read(2)

        return b"".join(chunks)

    def read_request_body(self):
        transfer_encoding = self.headers.get("Transfer-Encoding", "")

        if "chunked" in transfer_encoding.lower():
            return self.read_chunked_body()

        content_length = self.headers.get("Content-Length")

        if not content_length:
            return None

        try:
            length = int(content_length)
        except ValueError:
            raise ValueError("invalid Content-Length")

        if length < 0:
            raise ValueError("invalid Content-Length")

        if length > MAX_REQUEST_BODY:
            raise BodyTooLarge()

        return self.rfile.read(length)

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def do_PUT(self):
        self.forward()

    def do_PATCH(self):
        self.forward()

    def do_DELETE(self):
        self.forward()

    def do_OPTIONS(self):
        self.forward()

    def do_HEAD(self):
        self.forward()

    def forward(self):

        # ----------------------------------------------------
        # 读取客户端请求体
        # ----------------------------------------------------

        try:
            body = self.read_request_body()
        except BodyTooLarge:
            self.send_error(413, "Request body too large")
            return
        except ValueError as e:
            self.send_error(400, f"Bad request body: {e}")
            return

        # 客户端 model 字段统一替换成远端真实 model（写死一次配置即可）
        body = apply_model(body)

        # ----------------------------------------------------
        # 构造转发 headers
        # ----------------------------------------------------

        headers = {}

        hop_by_hop = {
            "host",
            "authorization",
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
            "content-length",
            # body 已由我们读取，无需再让上游发 100-continue
            "expect",
        }

        for key, value in self.headers.items():
            if key.lower() in hop_by_hop:
                continue
            headers[key] = value

        # 使用我们配置的远端 API key
        headers["Authorization"] = f"Bearer {REMOTE_API_KEY}"

        # HTTP/1.1 Host
        headers["Host"] = REMOTE.netloc

        if body is not None:
            headers["Content-Length"] = str(len(body))

        upstream_path = build_upstream_path(self.path)

        # ----------------------------------------------------
        # 连接 Kaggle / Cloudflare
        # ----------------------------------------------------

        conn = None
        try:
            conn, response = upstream_exchange(
                self.command, upstream_path, body, headers)

        except Exception as e:
            print(
                f"[proxy] upstream connection failed after "
                f"{UPSTREAM_RETRIES} attempt(s): {e}",
                file=sys.stderr,
                flush=True,
            )
            if conn is not None:
                _safe_close(conn)
            self.send_error(502, "Upstream connection failed")
            return

        # ----------------------------------------------------
        # 返回 HTTP 状态码
        # ----------------------------------------------------

        self.send_response(response.status)

        # ----------------------------------------------------
        # 返回 headers
        #
        # 关键：http.client 已经自动解掉 chunked 分帧，所以绝不能
        # 再把 Transfer-Encoding: chunked 透传给客户端，否则客户端
        # 会把去帧后的字节当成 chunk 帧解析而失败。
        # 这里用 Connection: close，让连接关闭作为 body 边界。
        # ----------------------------------------------------

        skip_headers = {
            "connection",
            "keep-alive",
            "transfer-encoding",
            "content-length",
            "server",
            "date",
        }

        upstream_content_length = None

        for key, value in response.getheaders():
            lower = key.lower()

            if lower == "content-length":
                upstream_content_length = value
                continue

            # 不转发 transfer-encoding（见上方说明）
            if lower == "transfer-encoding":
                continue

            if lower in skip_headers:
                continue

            self.send_header(key, value)

        # 强制短连接，用连接关闭界定响应体
        self.send_header("Connection", "close")

        # 仅在响应未被 chunked（长度可信）时才回传 Content-Length
        if upstream_content_length is not None and not response.chunked:
            self.send_header("Content-Length", upstream_content_length)

        self.end_headers()

        # ----------------------------------------------------
        # streaming copy
        # ----------------------------------------------------

        reuse = False
        try:
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            # 响应被完整读完、且上游没说 close -> 连接可以复用
            reuse = not response.will_close

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # 客户端提前断开，上游响应没读完，不能复用
            pass

        except Exception as e:
            print(
                f"[proxy] response streaming error: {e}",
                file=sys.stderr,
                flush=True,
            )

        finally:
            if reuse and conn is not None and conn.sock is not None:
                _pool_put(conn)     # keep-alive：下次请求省掉 DNS+TCP+TLS
            elif conn is not None:
                _safe_close(conn)
            self.close_connection = True


def main():

    global REMOTE_URL, REMOTE_API_KEY, REMOTE_MODEL, REMOTE, REMOTE_PATH_PREFIX

    ap = argparse.ArgumentParser(
        description="Reverse proxy: local port -> Kaggle TPU endpoint, "
                    "injecting the remote API key.",
    )
    ap.add_argument(
        "--remote-url",
        default=os.environ.get("REMOTE_URL", DEFAULT_REMOTE_URL),
        help="remote https endpoint (launch.py passes this automatically)",
    )
    ap.add_argument(
        "--remote-api-key",
        default=os.environ.get("REMOTE_API_KEY", ""),
        help="remote API key (or set the REMOTE_API_KEY environment variable)",
    )
    ap.add_argument(
        "--remote-model",
        default=os.environ.get("REMOTE_MODEL", ""),
        help="remote model name injected into request bodies (or set "
             "REMOTE_MODEL); clients may send anything",
    )
    ap.add_argument(
        "--host", default=LOCAL_HOST,
        help="local bind address (default: %(default)s)",
    )
    ap.add_argument(
        "--port", type=int, default=LOCAL_PORT,
        help="local port (default: %(default)s)",
    )
    ap.add_argument(
        "--local-api-key", default=LOCAL_API_KEY,
        help="fixed key advertised to clients; any client key is accepted "
             "(default: %(default)s)",
    )
    args = ap.parse_args()

    REMOTE_URL = normalize_remote_url(args.remote_url)
    REMOTE_API_KEY = args.remote_api_key
    REMOTE_MODEL = args.remote_model

    REMOTE = urlsplit(REMOTE_URL)

    if REMOTE.scheme != "https":
        raise RuntimeError("remote URL must use https://")

    if not REMOTE.netloc:
        raise RuntimeError("Invalid remote URL")

    if not REMOTE_API_KEY:
        raise RuntimeError(
            "No remote API key. Pass --remote-api-key or set REMOTE_API_KEY. "
            "(Normally launch.py starts the proxy automatically with both.)"
        )

    # 远端路径前缀，例如 REMOTE_URL=https://host/v1 时为 "/v1"
    REMOTE_PATH_PREFIX = REMOTE.path.rstrip("/")

    install_dns_cache()

    print()
    print("=" * 62)
    print("Kaggle TPU Reverse Proxy")
    print("=" * 62)
    print(f"LOCAL   http://{args.host}:{args.port}/v1   <- 客户端固定写这里")
    print(f"        key   = {args.local_api_key}  (填它/随便填/不填都行，本代理忽略)")
    print(f"        model = {REMOTE_MODEL or '(未指定，原样透传)'}"
          "  (客户端随便填/不填，本代理自动替换)")
    print(f"REMOTE  {REMOTE_URL}   (每次会话会变，客户端不用管)")
    print(f"        retry = {UPSTREAM_RETRIES} attempts, "
          f"connect timeout = {CONNECT_TIMEOUT:g}s")
    print(f"        DNS cache = {DNS_CACHE_TTL:g}s, keep-alive pool = "
          f"{KEEPALIVE_MAX} conns / {KEEPALIVE_TTL:g}s idle")
    print("=" * 62)
    print(flush=True)  # stdout may be a log file; don't sit in the buffer

    try:
        server = ThreadingHTTPServer(
            (args.host, args.port),
            ProxyHandler,
        )
    except OSError as e:
        raise SystemExit(
            f"Cannot listen on {args.host}:{args.port}: {e}\n"
            "Another proxy is already running there. launch.py stops and "
            "replaces it automatically; if you started one by hand, stop it "
            "first (or use another --port)."
        )

    # launch.py stops a proxy it wants to replace with SIGTERM; treat it like
    # Ctrl-C so the server socket is closed cleanly.
    def _on_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        server.serve_forever()

    except KeyboardInterrupt:

        print("\n[proxy] stopping...")

    finally:

        server.server_close()
        close_pool()


if __name__ == "__main__":
    main()
