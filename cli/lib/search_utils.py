import json

def loadMovie() -> list:
    with open("./data/movies.json") as f:
        data =  json.load(f)
        return data["movies"]
        
def loadstopword() -> list[str]:
    with open("./data/stopword.txt", 'r') as f:
        words = f.read().splitlines()
        return words