import spacy
nlp = spacy.load("en_core_web_sm")
import en_core_web_sm
nlp = en_core_web_sm.load()
doc = nlp("I'm won't go. I haven't gone")
print([(w.text, w.pos_) for w in doc])
