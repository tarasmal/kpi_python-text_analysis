import spacy
from spacy.matcher import Matcher
import json

nlp = spacy.load('en_core_web_sm')

with open('banks.json', 'r') as file:
    data = json.load(file)

matcher = Matcher(nlp.vocab)

transfer_patterns = [
    [{"LOWER": "transfer"}, {"LOWER": "money"}, {"LOWER": "to"}, {"POS": "PROPN"}],
    [{"LOWER": "send"}, {"LOWER": "money"}, {"LOWER": "to"}, {"POS": "PROPN"}],
    [{"LOWER": "give"}, {"LOWER": "money"}, {"LOWER": "to"}, {"POS": "PROPN"}],
    [{"LOWER": "transfer"}, {"LOWER": "money"}, {"LOWER": "to"}, {"ENT_TYPE": "PERSON"}],
    [{"LOWER": "send"}, {"LOWER": "money"}, {"LOWER": "to"}, {"ENT_TYPE": "PERSON"}],
    [{"LOWER": "give"}, {"LOWER": "money"}, {"LOWER": "to"}, {"ENT_TYPE": "PERSON"}],
]

matcher.add("TRANSFER_TO", transfer_patterns)

confirmation_patterns = [
    [{"LOWER": "yes"}],
    [{"LOWER": "yeah"}],
    [{"LOWER": "sure"}],
    [{"LOWER": "that"}, {"LOWER": "would"}, {"LOWER": "be"}, {"LOWER": "great"}],
    [{"LOWER": "okay"}],
    [{"LOWER": "ok"}],
    [{"LOWER": "alright"}]
]

confirmation_matcher = Matcher(nlp.vocab)
confirmation_matcher.add("CONFIRMATION", confirmation_patterns)

def find_transfer_to(doc):
    matches = matcher(doc)
    recipients = []
    for match_id, start, end in matches:
        span = doc[start:end]
        recipients.append(span.text)
    return recipients

def find_confirmations(doc):
    matches = confirmation_matcher(doc)
    confirmations = []
    for match_id, start, end in matches:
        span = doc[start:end]
        confirmations.append(span.text)
    return confirmations

for dialogue in data:
    for turn in dialogue['turns']:
        doc = nlp(turn['utterance'])
        recipients = find_transfer_to(doc)
        confirmations = find_confirmations(doc)
        if recipients:
            print(f"Transfer to: {recipients}")
        if confirmations:
            print(f"Confirmation: {confirmations}")
