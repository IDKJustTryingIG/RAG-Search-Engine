import argparse
from lib.semantic_search import verify_model, embed_text,verify_embeddings,embed_query_text,SemanticSearch,getData
from utils.constants import SEARCH_LIMIT

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    embedding_parser = subparsers.add_parser("verify_embeddings")

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate an embedding for a query")
    embed_query_parser.add_argument("query", type=str, help="query to embed")

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("search_query",type=str,help="enter search_query")
    search_parser.add_argument("limit", type=int, nargs="?", default=SEARCH_LIMIT, help="no of results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.query)

        case "search":
            search = SemanticSearch()
            search.load_or_create_embeddings(getData())
            results = search.search(args.search_query, args.limit)
            for rank, result in enumerate(results, start=1):
                print(f"{rank}. {result['title']} (score: {result['score']:.4f})")
                print(f"  {result['description'][:100]}...")
                print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()