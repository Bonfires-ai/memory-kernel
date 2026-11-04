"""``python -m memory_kernel.parser_service --port N --address 127.0.0.1``: run the managed parser service.

One uvicorn worker on one loopback address; ``pipeline.engine.ParserService`` starts and stops it. Needs the
``[parser]`` extra.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m memory_kernel.parser_service", description="run the parser service")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--address", default="127.0.0.1")
    args = p.parse_args(argv)
    try:
        import uvicorn
    except ImportError:
        print("the parser service needs the [parser] extra: pip install 'memory-kernel[parser]'", file=sys.stderr)
        return 2

    from memory_kernel.parser_service.app import create_app

    uvicorn.run(create_app(), host=args.address, port=args.port, workers=1, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
