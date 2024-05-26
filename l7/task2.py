import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import json
import random

with open('banks.json', 'r') as file:
    data = json.load(file)

train_data = []
for dialogue in data:
    for turn in dialogue['turns']:
        utterance = turn['utterance']
        if 'frames' in turn and turn['frames'] and 'state' in turn['frames'][0] and 'active_intent' in turn['frames'][0]['state']:
            intent = turn['frames'][0]['state']['active_intent']
            train_data.append((utterance, {'cats': {intent: 1.0}}))


nlp = spacy.blank('en')

if 'textcat' not in nlp.pipe_names:
    textcat = nlp.add_pipe('textcat', last=True)

for _, annotations in train_data:
    for cat in annotations['cats']:
        textcat.add_label(cat)


examples = []
for text, annotations in train_data:
    doc = nlp.make_doc(text)
    examples.append(Example.from_dict(doc, annotations))


optimizer = nlp.begin_training()
for i in range(10):
    losses = {}
    random.shuffle(examples)
    batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        nlp.update(batch, drop=0.5, losses=losses)
    print(f"Losses at iteration {i}:", losses)

nlp.to_disk("textcat_model")


def predict_intent(text):
    doc = nlp(text)
    intent = max(doc.cats, key=doc.cats.get)
    return intent

test_texts = [
    "I want to transfer some money.",
    "Can you check my account balance?",
    "I need help with a transaction.",
    "Please show me my recent transactions.",
    "What is the current balance of my savings account?"
]

for text in test_texts:
    print(f"Text: {text} \nPredicted Intent: {predict_intent(text)}\n")
