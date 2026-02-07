#!/usr/bin/env python3

import argparse
from lib.keyword_search import (
    Search, 
    build_commad, 
    get_tf, 
    idf, 
    tfidf, 
    bm25_idf_command, 
    bm25_tf_command, 
    bm25_search
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the index and docmap caches")
     
    tf_parser = subparsers.add_parser("tf", help="Getting the term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Id for the document")
    tf_parser.add_argument("term", type=str, help="The term to look for")
    
    idf_parser = subparsers.add_parser("idf", help="Get the idf score for a give term")
    idf_parser.add_argument("term", type=str, help="The term to which the idf score is to be found")
    
    tfidf_parser = subparsers.add_parser("tfidf", help="Getting the TF_IDF score")
    tfidf_parser.add_argument("doc_id", type=int, help="Id for the document")
    tfidf_parser.add_argument("term", type=str, help="The term to look for")
    
    bm25idf_parser = subparsers.add_parser("bm25idf", help="Get the idf score for a give term")
    bm25idf_parser.add_argument("term", type=str, help="The term to which the idf score is to be found")  
    
    bm25tf_parser = subparsers.add_parser("bm25tf", help="Getting the BM25 TF score")
    bm25tf_parser.add_argument("doc_id", type=int, help="Id for the document")
    bm25tf_parser.add_argument("term", type=str, help="The term to look for")
    
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    
    args = parser.parse_args()

    match args.command:
        case "search":
            movies = Search(args.query)
            for i, mv in enumerate(movies):
                print(f"{i+1}: {mv["title"]}")
        case "build":
            build_commad()
        case "tf":
            get_tf(args.doc_id, args.term)
        case "idf":
            value = idf(args.term)
            print(f"Inverse document frequency of '{args.term}:{value:.2f}'")
        case "tfidf":
            value = tfidf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}' : '{value:.2f}'")
        case "bm25idf":
            value = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}' : {value:.2f}")
        case "bm25tf":
            value = bm25_tf_command(args.doc_id, args.term)
            print(f"BM25 TF score of '{args.term}' : {value:.2f}")
        case "bm25search":
            bm25_search(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()