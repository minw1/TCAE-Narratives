
import spacy

nlp = spacy.load("en_core_web_trf")

text = "I know y'all"
doc = nlp(text)
for w in doc:
    print(w.text, w.pos_)