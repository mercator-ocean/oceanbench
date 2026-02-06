# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from urllib.request import urlopen, Request
from urllib.error import URLError

from oceanbench.core.version import __version__

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/mercator-ocean/oceanbench"
GITHUB_API_BASE = "https://api.github.com/repos/mercator-ocean/oceanbench/contents"
CHALLENGER_DIRECTORY = "challenger_datasets"


def _get_version_ref() -> str:
    return f"v{__version__}"


def _resolve_all_challenger_urls() -> list[str]:
    ref = _get_version_ref()
    url = f"{GITHUB_API_BASE}/{CHALLENGER_DIRECTORY}?ref={ref}"
    request = Request(url, headers={"Accept": "application/vnd.github.v3+json"})
    with urlopen(request) as response:
        entries = json.loads(response.read())
    return [
        f"{GITHUB_RAW_BASE}/{ref}/{CHALLENGER_DIRECTORY}/{entry['name']}"
        for entry in entries
        if entry["name"].endswith(".py")
    ]


def _evaluate_one(challenger: str, output_bucket: str | None, output_prefix: str | None) -> None:
    from oceanbench.core.evaluate import evaluate_challenger

    evaluate_challenger(
        challenger_python_code_uri_or_local_path=challenger,
        output_bucket=output_bucket,
        output_prefix=output_prefix,
    )


def _run_evaluate(args: argparse.Namespace) -> int:
    if args.all_challengers:
        try:
            challengers = _resolve_all_challenger_urls()
        except URLError as e:
            print(f"Error: failed to fetch challengers from GitHub: {e}", file=sys.stderr)
            return 1
        if not challengers:
            print("No challengers found on GitHub for version", _get_version_ref(), file=sys.stderr)
            return 1
    else:
        challengers = args.challengers

    if not challengers:
        print("Error: provide challenger files or use --all-challengers", file=sys.stderr)
        return 1

    output_bucket = args.output_bucket
    output_prefix = args.output_prefix

    if len(challengers) == 1:
        try:
            _evaluate_one(challengers[0], output_bucket, output_prefix)
            return 0
        except Exception as e:
            print(f"Error evaluating {challengers[0]}: {e}", file=sys.stderr)
            return 1

    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    with ProcessPoolExecutor() as executor:
        future_to_challenger = {executor.submit(_evaluate_one, c, output_bucket, output_prefix): c for c in challengers}
        for future in as_completed(future_to_challenger):
            challenger = future_to_challenger[future]
            try:
                future.result()
                successes.append(challenger)
                print(f"OK: {challenger}")
            except Exception as e:
                failures.append((challenger, str(e)))
                print(f"FAIL: {challenger}: {e}", file=sys.stderr)

    print(f"\n{len(successes)} succeeded, {len(failures)} failed")
    if failures:
        for name, error in failures:
            print(f"  - {name}: {error}", file=sys.stderr)
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="oceanbench",
        description="OceanBench CLI",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate one or more challengers",
        description="Evaluate challengers against OceanBench metrics.",
    )
    evaluate_parser.add_argument(
        "challengers",
        nargs="*",
        help="Challenger file paths or URLs to evaluate",
    )
    evaluate_parser.add_argument(
        "--all-challengers",
        action="store_true",
        help="Evaluate all challengers from GitHub for the current version",
    )
    evaluate_parser.add_argument(
        "--output-bucket",
        default=None,
        help="S3 bucket for output notebooks (env: OCEANBENCH_OUTPUT_BUCKET)",
    )
    evaluate_parser.add_argument(
        "--output-prefix",
        default=None,
        help="S3 prefix for output notebooks (env: OCEANBENCH_OUTPUT_PREFIX)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "evaluate":
        if args.all_challengers and args.challengers:
            evaluate_parser.error("--all-challengers cannot be used with positional challenger arguments")
        sys.exit(_run_evaluate(args))


if __name__ == "__main__":
    main()
