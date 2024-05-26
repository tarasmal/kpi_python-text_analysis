import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random

banks = ["Bank A", "Bank B", "Bank C", "Bank D", "Bank E"]
services = ["business loans", "savings accounts", "credit cards", "insurance", "wire transfers", "online banking", "mortgage loans"]

def generate_examples(banks, services):
    examples = []

    for bank in banks:
        text = bank
        entities = [(0, len(bank), "BANK")]
        examples.append({"text": text, "entities": entities})

    for service in services:
        text = service
        entities = [(0, len(service), "SERVICE")]
        examples.append({"text": text, "entities": entities})

    complex_texts = [
        f"{banks[0]} offers {services[0]} and {services[1]}.",
        f"{banks[1]} provides {services[2]} and {services[3]}.",
        f"{banks[2]} specializes in {services[4]} and {services[5]}.",
        f"{banks[3]} assists with {services[6]}.",
        f"{banks[4]} offers {services[0]} and {services[3]}.",
    ]

    for text in complex_texts:
        entities = []
        for bank in banks:
            start = text.find(bank)
            if start != -1:
                end = start + len(bank)
                entities.append((start, end, "BANK"))
        for service in services:
            start = text.find(service)
            if start != -1:
                end = start + len(service)
                entities.append((start, end, "SERVICE"))
        examples.append({"text": text, "entities": entities})

    return examples

def create_training_data(examples):
    nlp = spacy.blank("en")
    db = DocBin()

    for example in examples:
        doc = nlp.make_doc(example["text"])
        ents = []
        for start, end, label in example["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is None:
                print(f"Skipping entity with start: {start} and end: {end} in text: '{example['text'][start:end]}'")
                continue
            ents.append(span)
        doc.ents = ents
        db.add(doc)

    return db

examples = generate_examples(banks, services)

training_data = create_training_data(examples)
training_data.to_disk("training_data.spacy")

def train_model():
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")

    ner.add_label("BANK")
    ner.add_label("SERVICE")

    db = DocBin().from_disk("training_data.spacy")
    training_data = list(db.get_docs(nlp.vocab))

    optimizer = nlp.begin_training()
    for itn in range(100):
        random.shuffle(training_data)
        losses = {}
        for batch in spacy.util.minibatch(training_data, size=2):
            examples = [Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}) for doc in batch]
            nlp.update(examples, sgd=optimizer, drop=0.35, losses=losses)
        print(f"Losses at iteration {itn}:", losses)

    nlp.to_disk("trained_ner_model")

train_model()

nlp = spacy.load("trained_ner_model")
long_text = (
    "Bank A offers attractive business loans and personal loans. "
    "Bank B provides excellent conditions for opening savings accounts. "
    "Bank C offers various services, including credit cards and insurance. "
    "Bank D will assist you with wire transfers, online banking, and mortgage loans. "
    "Bank E offers favorable mortgage loan conditions and investment opportunities."
)

doc = nlp(long_text)
for ent in doc.ents:
    print(ent.text, ent.label_)
