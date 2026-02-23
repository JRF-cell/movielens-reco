import argparse


def cmd_health(_: argparse.Namespace) -> int:
    print("ok")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reco", description="MovieLens recommender CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    health = sub.add_parser("health", help="Sanity check")
    health.set_defaults(func=cmd_health)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))