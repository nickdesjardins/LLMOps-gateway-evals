"""Run the showcase on this machine, the way the public demo runs.

Starts uvicorn with ``DEMO_MODE=true`` (the budget on, model names in requests ignored) for this
process only, on http://127.0.0.1:8013/ by default. Everything else comes from ``.env`` and
``.env.local``: the provider key, ``REDIS_URL`` (``docker compose up -d redis`` for a local
Redis 8 with vector search), the Langfuse keys. Without a provider key it runs on the offline
provider, which calls no model.

    python scripts/run_local_demo.py
    python scripts/run_local_demo.py --offline --port 8020
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--offline", action="store_true", help="LLM_PROVIDER=offline: no model is called")
    args = parser.parse_args()
    os.environ["DEMO_MODE"] = "true"
    if args.offline:
        os.environ["LLM_PROVIDER"] = "offline"
    else:
        from app.config import Settings

        if not Settings().provider_configured():
            print("No provider key in .env / .env.local: running on the offline provider.")
            os.environ["LLM_PROVIDER"] = "offline"
    import uvicorn

    os.chdir(ROOT)
    uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
