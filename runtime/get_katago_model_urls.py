import argparse
import re
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape KataGo training site for recent .bin.gz model URLs."
    )
    parser.add_argument(
        "--base-url",
        default="https://katagotraining.org",
        help="Base URL for the KataGo training site.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Max number of URLs to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url = f"{args.base_url}/networks/"
    with urllib.request.urlopen(url) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    raw_links = re.findall(r"[^\\\"\\s>]+\\.bin\\.gz", html)
    seen = set()
    links = []
    for link in raw_links:
        if link.startswith("//"):
            link = "https:" + link
        elif link.startswith("/"):
            link = args.base_url + link
        if link not in seen:
            seen.add(link)
            links.append(link)

    if links:
        for link in links[: args.limit]:
            print(link)
    else:
        print("NO_BIN_GZ_FOUND")
        for m in re.findall(r"[^\\\"\\s>]+\\.json", html)[:10]:
            print("JSON_REF", m)
        for line in html.splitlines():
            if "networks/" in line:
                print("LINE", line.strip())


if __name__ == "__main__":
    main()
