import spacy
nlp = spacy.load("en_core_web_sm")
import en_core_web_sm
nlp = en_core_web_sm.load()
doc = nlp("I'm happy. I wouldn't know. I saw you at 3 o’clock.")
print([(w.text, w.pos_) for w in doc])
