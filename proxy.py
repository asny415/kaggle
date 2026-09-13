#!/usr/bin/env python3
"""
Kaggle TPU Reverse Proxy

把本地请求（默认 http://127.0.0.1:9000）转发到 Kaggle TPU 远端 endpoint，
并自动注入远端 API key —— 客户端只需要知道本地地址。

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
import os
import signal
import ssl
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# 默认远端地址，仅作 fallback。launch.py 会自动传 --remote-url，
# 通常不需要改这里。
DEFAULT_REMOTE_URL = "https://toner-shopper-feelings-ending.trycloudflare.com"

# 本地监听地址
LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 9000

# 远端连接超时时间（秒）
REMOTE_TIMEOUT = 600

# 允许的最大请求体（字节），超过则返回 413
MAX_REQUEST_BODY = 256 * 1024 * 1024


# ============================================================
# 以下一般不需要修改
# ============================================================

# 运行时配置：由 main() 根据命令行参数 / 环境变量填充
REMOTE_URL = ""
REMOTE_API_KEY = ""
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

        try:
            conn = http.client.HTTPSConnection(
                REMOTE.netloc,
                timeout=REMOTE_TIMEOUT,
                context=SSL_CONTEXT,
            )

            conn.request(
                self.command,
                upstream_path,
                body=body,
                headers=headers,
            )

            response = conn.getresponse()

        except Exception as e:
            print(
                f"[proxy] upstream connection failed: {e}",
                file=sys.stderr,
                flush=True,
            )
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

        try:
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

        except Exception as e:
            print(
                f"[proxy] response streaming error: {e}",
                file=sys.stderr,
                flush=True,
            )

        finally:
            conn.close()
            self.close_connection = True


def main():

    global REMOTE_URL, REMOTE_API_KEY, REMOTE, REMOTE_PATH_PREFIX

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
        "--host", default=LOCAL_HOST,
        help="local bind address (default: %(default)s)",
    )
    ap.add_argument(
        "--port", type=int, default=LOCAL_PORT,
        help="local port (default: %(default)s)",
    )
    args = ap.parse_args()

    REMOTE_URL = normalize_remote_url(args.remote_url)
    REMOTE_API_KEY = args.remote_api_key

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

    print()
    print("=" * 60)
    print("Kaggle TPU Reverse Proxy")
    print("=" * 60)
    print(f"Local : http://{args.host}:{args.port}")
    print(f"Remote: {REMOTE_URL}")
    print("=" * 60)
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


if __name__ == "__main__":
    main()
