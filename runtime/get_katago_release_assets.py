import argparse
import json
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List CUDA/TensorRT Linux assets from the latest KataGo release."
    )
    parser.add_argument(
        "--repo",
        default="lightvector/KataGo",
        help="GitHub repo in owner/name format.",
    )
    parser.add_argument(
        "--include",
        default="linux",
        help="Substring that must appear in asset names.",
    )
    parser.add_argument(
        "--kinds",
        default="cuda,tensorrt",
        help="Comma-separated keywords to match asset names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kinds = [k.strip().lower() for k in args.kinds.split(",") if k.strip()]
    url = f"https://api.github.com/repos/{args.repo}/releases/latest"
with urllib.request.urlopen(url) as r:
    data = json.load(r)

    assets = {a["name"]: a["browser_download_url"] for a in data.get("assets", [])}
    print("ASSETS")
for name in sorted(assets):
        lower = name.lower()
        if args.include in lower and any(k in lower for k in kinds):
            print(f"{name} {assets[name]}")


if __name__ == "__main__":
    main()
