import string
from lib.search_utils import loadMovie,loadstopword
from nltk.stem import PorterStemmer
from collections import defaultdict
import pickle
import os
from pathlib import Path

stemmer = PorterStemmer()
cache_dir = Path("cached")

class InvertedIndex:
    def __init__ (self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.indexpath = f"{cache_dir}/index.pkl"
        self.docmappath = f"{cache_dir}/docmap.pkl"
    
    def add_document(self, doc_id:int, text:str):
        term_token = token(text)
        for tok in term_token:
            self.index[tok].add(doc_id)
    
    def get_document(self, term:str) -> list:
        term = term.lower()
        return sorted(self.index.get(term, []))
    
    def build(self):
        movies = loadMovie()
        for _, mv in enumerate(movies):
            text = f"{mv["title"]} {mv["description"]}"
            self.add_document(mv["id"], text)
            self.docmap[mv["id"]] = mv
        
    def save(self):
        # make the need dir
        os.makedirs("cached", exist_ok=True)
        
        with open(self.indexpath, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.docmappath, 'wb') as f:
            pickle.dump(self.docmap, f)
    
        
def token(text:str) -> list[str] :
    text = text.lower()
    text = text.translate(str.maketrans("","", string.punctuation))
    
    stop_word = loadstopword()
    result = [tok for tok in text.split() if tok and tok not in stop_word]

    return result

def Search(term:str) -> list:
    movies = loadMovie()
    result = []
    term_token = token(stemmer.stem(term))
    for mv in movies:
        mv_token = token(stemmer.stem(mv["title"]))
        for t in term_token:
            for m in mv_token:
                if t in m :
                    result.append(mv)
                if len(result) >= 5:
                    return result
    
    return result      
    
def build_commad():
    i = InvertedIndex()
    i.build()
    i.save()
    
    value = i.get_document("merida")
    print(f"First document for 'merida' = {value[0]}")