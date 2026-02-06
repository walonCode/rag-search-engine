#!/usr/bin/env python3

import argparse
from lib.keyword_search import Search,build_commad

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the index and docmap caches")
     
    args = parser.parse_args()

    match args.command:
        case "search":
            movies = Search(args.query)
            for i, mv in enumerate(movies):
                print(f"{i+1}: {mv["title"]}")
        case "build":
            build_commad()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()