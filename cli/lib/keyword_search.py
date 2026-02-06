import math
import string
from lib.search_utils import loadMovie,loadstopword
from nltk.stem import PorterStemmer
from collections import defaultdict,Counter
import pickle
import os
from pathlib import Path

stemmer = PorterStemmer()
cache_dir = Path("cached")

class InvertedIndex:
    def __init__ (self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequency = defaultdict(Counter)
        
        self.term_frequencypath = f"{cache_dir}/tf.pkl"
        self.indexpath = f"{cache_dir}/index.pkl"
        self.docmappath = f"{cache_dir}/docmap.pkl"
    
    def add_document(self, doc_id:int, text:str):
        term_token = token(text)
        for tok in term_token:
            self.index[tok].add(doc_id)
            self.term_frequency[doc_id][tok] += 1
    
    def get_document(self, term:str) -> list:
        term = term.lower()
        return sorted(self.index.get(term, []))
    
    def build(self):
        movies = loadMovie()
        for _, mv in enumerate(movies):
            text = f"{mv['title']} {mv['description']}"
            self.add_document(mv["id"], text)
            self.docmap[mv["id"]] = mv
        
    def save(self):
        # make the need dir
        os.makedirs("cached", exist_ok=True)
        
        with open(self.indexpath, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.docmappath, 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencypath, 'wb') as f:
            pickle.dump(self.term_frequency, f)
     
    def load(self):
        with open(self.indexpath, 'rb') as f:
            data = pickle.load(f)
            self.index = data
        with open(self.docmappath, 'rb') as f:
            data = pickle.load(f)
            self.docmap = data
        with open(self.term_frequencypath, 'rb') as f:
            data = pickle.load(f)
            self.term_frequency = data    
            
    def get_tf(self, doc_id, term):
        term_token = token(term)
        if len(term_token) > 1:
            raise ValueError("token more than one")
        
        return self.term_frequency[doc_id][term_token[0]]    
    
        
def token(text:str) -> list[str] :
    text = text.lower()
    text = text.translate(str.maketrans("","", string.punctuation))
    
    stop_word = loadstopword()
    result = [tok for tok in text.split() if tok and tok not in stop_word]

    return result

def Search(term:str) -> list:
    i = InvertedIndex()
    i.load()
    
    result = []
    token_term = token(term.lower())
    for tok in token_term:
        value = i.get_document(tok)
        for t in value:
            mv = i.docmap[t]
            result.append(mv)
            if len(result) == 5 :
                return result
    return result      
 
    
def build_commad():
    i = InvertedIndex()
    i.build()
    i.save()
 
    
def get_tf(doc_id, term):
    i = InvertedIndex()
    i.load()
    value = i.get_tf(doc_id, term)
    print(f"{doc_id} - {value}")


def idf(term:str) -> float:
    i = InvertedIndex()
    i.load()
    
    total_doc_count = len(i.docmap)
    total_match_doc_count = len(i.get_document(term))
    
    return math.log((total_doc_count + 1) / (total_match_doc_count + 1))


def tfidf( doc_id:int, term:str) -> float:
    i = InvertedIndex()
    i.load()
    
    tf = i.get_tf(doc_id, term)
    idf_value = idf(term)
    
    return tf * idf_value