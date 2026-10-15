import argparse
import json
import string
from nltk.stem import PorterStemmer
import pickle
from pathlib import Path
import sys
from collections import defaultdict, Counter
import math
import os
from utils.constants import BM25_K1, BM25_B, BM25_SEARCH_LIMIT, CACHE_DIR

def accessStopWords(punctuation_mapping) -> []:
    with open("./data/stopwords.txt","r",encoding="utf-8") as f:
        stopWords = f.read().splitlines()
    result = []
    for word in stopWords:
        result.append(word.translate(punctuation_mapping))
    return result
        
def removeStopWords(stops,query):
    result = []
    for word in query:
        if word in stops:
            continue
        else:
            result.append(word)
    return result

def tokenizeText(text):
    punctuation_mapping = {ord(c): None for c in string.punctuation}
    stemmer = PorterStemmer()
    stop_words = accessStopWords(punctuation_mapping)
    text = text.lower().translate(punctuation_mapping).split()
    text = removeStopWords(stop_words,text)
    text = [stemmer.stem(token) for token in text]
    return text

def tokenize(term):
    token = tokenizeText(term)
    if token is None:
        raise ValueError("tokenize text cannot return None")
    return token[0]

def getIdf(idx,token):
    # call this method after you did idx.load

    num_documents = len(idx.term_frequencies)
    doc_count = 0

    # Iterate through each document's counter
    for doc_id, term_counter in idx.term_frequencies.items():
        # Check if the term appears in this document's counter
        if token in term_counter:
            doc_count += 1

    idf = math.log((num_documents + 1) / (doc_count + 1))
    return idf

def bm25_idf_command(term) -> float:
    idx = InvertedIndex()
    idx.load()
    token = tokenize(term)
    return idx.get_bm25_idf(token)

def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    token = tokenize(term)
    return idx.get_bm25_tf(doc_id, token, k1, b)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build") # you can add help
    # build_parser.add_argument("file", type=str, help="file to build")

    tf_parser = subparsers.add_parser("tf", help="Find term frequency given doc id and term")
    tf_parser.add_argument("doc_id", type=int, help="provide document id")
    tf_parser.add_argument("term", type=str, help="provide term")

    idf_parser = subparsers.add_parser("idf", help="Find inverse document for a given term")
    idf_parser.add_argument("idf_term", type=str, help="provide term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Find tfidf for given doc id and term")
    tfidf_parser.add_argument("tfidf_doc_id", type=int, help="provide document id")
    tfidf_parser.add_argument("tfidf_term", type=str, help="provide term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=BM25_SEARCH_LIMIT, help="Maximum number of results")

    args = parser.parse_args()
    my = InvertedIndex()
    match args.command:
        case "search":
            # movies = find_movies(args.query)
            # for movie in movies:
            #     print(f"{movie}")
            try:
                with open("./cache/index.pkl","rb") as f:
                    index = pickle.load(f)
                with open("./cache/docmap.pkl","rb") as f2:
                    docs = pickle.load(f2)
            except FileNotFoundError:
                print("index.pkl file not found")
                sys.exit(1)

            tokens = tokenizeText(args.query)
            result = []
            flag = 0
            for token in tokens:
                movies = index.get(token)
                if movies is None or len(movies)==0:
                    continue
                for movie in movies:
                    if len(result)<5:
                        result.append(movie)
                    else:
                        flag = 1
                        break
                if flag==1:
                    break
            for id in result:
                print(docs.get(id).get("title"))

        case "build":
                my.build()
                my.save()

        case "tf":
            my.load()
            doc_id = args.doc_id
            term = args.term
            token = tokenize(term)
            print(my.get_tf(doc_id,token))

        case "idf":
            my.load()
            currentTerm = args.idf_term
            token = tokenize(currentTerm)
            idf = getIdf(my,token)
            print(f"Inverse document frequency of '{currentTerm}': {idf:.2f}")
                
        case "tfidf":
            docId = args.tfidf_doc_id
            tfidfTerm = args.tfidf_term
            token = tokenize(tfidfTerm)
            my.load()
            tf = my.get_tf(docId,tfidfTerm)
            idf = getIdf(my,token)
            tfIdf = tf*idf
            print(f"{tfIdf:.2f}")

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            my.load()
            results = my.bm25_search(args.query, args.limit)
            for rank, (doc_id, score) in enumerate(results, start=1):
                title = my.docmap[doc_id]["title"]
                print(f"{rank}. ({doc_id}) {title} - Score: {score:.2f}")

        case _:
            accessStopWords()
            print("hi you reached me")
            parser.print_help()

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        title,description = text.split(")(#$^@")
        self.docmap[doc_id] = {"id":doc_id,"title":title,"description":description}
        title_tokens = tokenizeText(title)
        description_tokens = tokenizeText(description)
        for token in title_tokens:
            self.index.setdefault(token, []).append(doc_id)
            self.term_frequencies[doc_id][token] += 1
        for token in description_tokens:
            self.index.setdefault(token, []).append(doc_id)
            self.term_frequencies[doc_id][token] += 1
        self.doc_lengths[doc_id] = len(title_tokens) + len(description_tokens)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_tf(self, doc_id, term):
        return self.term_frequencies.get(doc_id, {}).get(term, 0)
    
    def get_documents(self, term):
        return self.index.get(term.lower(),[])

    def get_bm25_idf(self, term: str) -> float:
        # term must already be a single token
        num_documents = len(self.docmap)
        doc_freq = len(self.index.get(term, []))
        return math.log((num_documents - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        # term must already be a single token
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id, term) -> float:
        # term must already be a single token
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query, limit=BM25_SEARCH_LIMIT):
        tokens = tokenizeText(query)
        scores = {}
        for doc_id in self.docmap:
            scores[doc_id] = sum(self.bm25(doc_id, token) for token in tokens)
        # reorder the data so that higher bm25 values are shown first
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:limit]
    
    def build(self):
        with open("./data/movies.json", "r", encoding="utf-8") as f:
            movies = json.load(f)
        movies = movies.get("movies")
        for movie in movies:
            self.__add_document(movie["id"],f"{movie["title"]})(#$^@{movie["description"]}")
        # we'll remove duplicates from index look if this is correct
        for key, value in self.index.items():
            self.index[key] = sorted(set(value))

    def save(self):
        with open("./cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("./cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("./cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        index_path = Path("./cache/index.pkl")
        docmap_path = Path("./cache/docmap.pkl")
        term_frequencies_path = Path("./cache/term_frequencies.pkl")
        if index_path.exists() and docmap_path.exists():
            with open(index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
        else:
            raise FileNotFoundError("Required cache files (index.pkl or docmap.pkl) do not exist!")


if __name__ == "__main__":
    main()