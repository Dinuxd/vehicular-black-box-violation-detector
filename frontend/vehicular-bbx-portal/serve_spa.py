from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent / "dist"


class SpaHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        parsed = urlparse(path)
        relative = unquote(parsed.path.lstrip("/"))
        return str(ROOT / relative)

    def do_GET(self):
        parsed = urlparse(self.path)
        target = ROOT / unquote(parsed.path.lstrip("/"))

        if parsed.path != "/" and not target.exists() and "." not in Path(parsed.path).name:
            self.path = "/index.html"

        return super().do_GET()


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 3001), SpaHandler)
    print("Serving SPA on http://127.0.0.1:3001")
    server.serve_forever()
