import spacy 

nlp = None

def initialize():
    print('initializing sentiment model...')
    
    global nlp
    nlp = spacy.load("./models/output/model-best")

    print('initialized')


def is_negative(text):
    doc = nlp(text)
    print(doc.cats)
    return doc.cats['negative'] > .5