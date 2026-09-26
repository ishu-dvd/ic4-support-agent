import argparse

from .app import serve
from .store import VARIANTS


def main():
    p = argparse.ArgumentParser(prog="python3 -m server", description="Mock systems-of-record API")
    p.add_argument("--variant", default="support", choices=VARIANTS, help="which fixture to load")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--latency-ms", type=int, default=0, help="artificial delay on every request")
    a = p.parse_args()
    serve(variant=a.variant, host=a.host, port=a.port, latency_ms=a.latency_ms)


if __name__ == "__main__":
    main()
