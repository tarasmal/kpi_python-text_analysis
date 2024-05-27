import json

import spacy

nlp = spacy.load('en_core_web_sm')

with open('banks.json', 'r') as file:
    data = json.load(file)


def find_intent(doc):
    intent = None

    for token in doc:
        if token.lemma_ in ["transfer", "send", "give"] and token.dep_ == "ROOT":
            for child in token.children:
                if child.lemma_ == "money":
                    intent = "TransferMoney"
                    break

        if token.lemma_ == "check" and token.dep_ == "ROOT":
            for child in token.children:
                if child.lemma_ == "balance":
                    intent = "CheckBalance"
                    break

        if token.lemma_ == "show" and token.dep_ == "ROOT":
            for child in token.children:
                if child.lemma_ == "transaction":
                    intent = "ShowTransactions"
                    break

        if token.lemma_ == "help" and token.dep_ == "ROOT":
            for child in token.children:
                if child.lemma_ == "transaction":
                    intent = "HelpTransaction"
                    break

    return intent

for dialogue in data:
    for turn in dialogue['turns']:
        doc = nlp(turn['utterance'])
        intent = find_intent(doc)
        if intent:
            print(f"Utterance: {turn['utterance']}\nPredicted Intent: {intent}\n")
