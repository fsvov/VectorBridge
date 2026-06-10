"""Helper entrypoint for local developers.

The web app is served by FastAPI. Start it with:

    uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
"""


def main() -> None:
    print("Start VectorBridge with:")
    print("uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload")


if __name__ == "__main__":
    main()
