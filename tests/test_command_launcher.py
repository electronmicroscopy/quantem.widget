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
