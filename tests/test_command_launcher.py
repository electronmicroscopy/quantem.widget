import http.client
import os
import socket
import subprocess
import sys
import time
import urllib.request

from quantem.widget.command_launcher import write_command_launcher


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_range_server_answers_head_and_ranges_on_one_connection(tmp_path):
    """A streamed .qem load: HEAD for the size, then byte ranges, all on one kept-alive connection."""
    payload = os.urandom(3 << 20)
    (tmp_path / "t0-counts.qem").write_bytes(payload)
    write_command_launcher(tmp_path, "Show4DSTEM")
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(tmp_path / ".viewer" / "serve_range.py"),
         "--root", str(tmp_path), "--host", "127.0.0.1", "--port", str(port)],
    )
    try:
        for _ in range(30):
            try:
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                connection.request("HEAD", "/t0-counts.qem")
                break
            except OSError:
                time.sleep(0.05)
        head = connection.getresponse()
        assert head.status == 200 and head.read() == b""
        assert int(head.getheader("Content-Length")) == len(payload)
        assert not head.will_close
        first_socket = connection.sock
        connection.request("GET", "/t0-counts.qem", headers={"Range": "bytes=1000-2500999"})
        part = connection.getresponse()
        assert part.status == 206 and part.read() == payload[1000:2501000]
        connection.request("GET", "/t0-counts.qem")
        whole = connection.getresponse()
        assert whole.status == 200 and whole.read() == payload
        assert connection.sock is first_socket
        connection.close()
    finally:
        proc.terminate()
        proc.wait(timeout=3)


def test_range_server_reports_export_folder_root(tmp_path):
    (tmp_path / "index.html").write_text("ok", encoding="utf-8")
    launcher = write_command_launcher(tmp_path, "ShowTest")
    server = tmp_path / ".viewer" / "serve_range.py"
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            str(server),
            "--root",
            str(tmp_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        root = ""
        for _ in range(30):
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/__quantem_viewer_root__",
                    timeout=0.25,
                ) as resp:
                    root = resp.read().decode("utf-8")
                break
            except OSError:
                time.sleep(0.05)
        assert root == str(tmp_path.resolve())
        text = launcher.read_text(encoding="utf-8")
        assert "__quantem_viewer_root__" in text
        assert "sleep 1" not in text
        assert "timeout=0.1" in text
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
