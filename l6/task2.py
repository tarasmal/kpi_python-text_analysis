import spacy

from util import get_text_from_file

nlp = spacy.load("en_core_web_sm")

text = get_text_from_file("lab6-3.txt")

doc = nlp(text)

stop_words_in_text = [token.text for token in doc if token.is_stop]
print("Стоп-слова в тексті:", set(stop_words_in_text))

nouns_in_text = [token.text for token in doc if token.pos_ == "NOUN"]
print("Іменники в тексті:", set(nouns_in_text))

numbers_in_text = [ent.text for ent in doc.ents if ent.label_ == "CARDINAL"]
organizations_in_text = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
print("Числа в тексті:", set(numbers_in_text))
print("Організації в тексті:", set(organizations_in_text))
