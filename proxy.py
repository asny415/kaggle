#!/usr/bin/env python3

# ============================================================
# Kaggle TPU Reverse Proxy
#
# 修改下面两个常量，然后重新启动脚本即可。
# ============================================================

REMOTE_URL = "https://toner-shopper-feelings-ending.trycloudflare.com/v1"
REMOTE_API_KEY = "sk-a361fe51bf570881049fdfc7c371fa27"

# 本地监听地址
LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 9000

# 远端连接超时时间（秒）
REMOTE_TIMEOUT = 600


# ============================================================
# 以下一般不需要修改
# ============================================================

import http.client
import ssl
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


def normalize_remote_url(url):
    return url.rstrip("/")


REMOTE_URL = normalize_remote_url(REMOTE_URL)
REMOTE = urlsplit(REMOTE_URL)

if REMOTE.scheme != "https":
    raise RuntimeError("REMOTE_URL must use https://")

if not REMOTE.netloc:
    raise RuntimeError("Invalid REMOTE_URL")


class ProxyHandler(BaseHTTPRequestHandler):

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        # 保留简单访问日志
        print(
            f"[proxy] {self.client_address[0]} - {fmt % args}",
            flush=True,
        )

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

        body = None

        content_length = self.headers.get("Content-Length")

        if content_length:
            try:
                length = int(content_length)
            except ValueError:
                self.send_error(400, "Invalid Content-Length")
                return

            body = self.rfile.read(length)

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

        # ----------------------------------------------------
        # 连接 Kaggle / Cloudflare
        # ----------------------------------------------------

        try:

            if REMOTE.scheme == "https":
                conn = http.client.HTTPSConnection(
                    REMOTE.netloc,
                    timeout=REMOTE_TIMEOUT,
                    context=ssl.create_default_context(),
                )
            else:
                conn = http.client.HTTPConnection(
                    REMOTE.netloc,
                    timeout=REMOTE_TIMEOUT,
                )

            conn.request(
                self.command,
                self.path,
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

            self.send_error(
                502,
                f"Upstream connection failed: {e}",
            )

            return

        # ----------------------------------------------------
        # 返回 HTTP 状态码
        # ----------------------------------------------------

        self.send_response(response.status)

        # ----------------------------------------------------
        # 返回 headers
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
        upstream_transfer_encoding = None

        for key, value in response.getheaders():

            lower = key.lower()

            if lower == "content-length":
                upstream_content_length = value
                continue

            if lower == "transfer-encoding":
                upstream_transfer_encoding = value
                continue

            if lower in skip_headers:
                continue

            self.send_header(key, value)

        # ----------------------------------------------------
        # 处理 response body
        #
        # OpenCode2 会用 streaming，因此不能简单地
        # response.read() 后一次性返回。
        # ----------------------------------------------------

        if upstream_transfer_encoding:
            self.send_header(
                "Transfer-Encoding",
                upstream_transfer_encoding,
            )

        elif upstream_content_length:
            self.send_header(
                "Content-Length",
                upstream_content_length,
            )

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

        except (BrokenPipeError, ConnectionResetError):

            # OpenCode2 主动断开，例如取消生成
            pass

        except Exception as e:

            print(
                f"[proxy] response streaming error: {e}",
                file=sys.stderr,
                flush=True,
            )

        finally:

            conn.close()


def main():

    print()
    print("=" * 60)
    print("Kaggle TPU Reverse Proxy")
    print("=" * 60)
    print(f"Local : http://{LOCAL_HOST}:{LOCAL_PORT}")
    print(f"Remote: {REMOTE_URL}")
    print("=" * 60)
    print()

    server = ThreadingHTTPServer(
        (LOCAL_HOST, LOCAL_PORT),
        ProxyHandler,
    )

    try:
        server.serve_forever()

    except KeyboardInterrupt:

        print("\n[proxy] stopping...")

    finally:

        server.server_close()


if __name__ == "__main__":
    main()
