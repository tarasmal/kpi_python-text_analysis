import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

nlp = spacy.load('en_core_web_sm', disable=["ner", "parser"])

def spacy_preprocess(text):
    doc = nlp(text)
    print('3')
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

def main():


    data = pd.read_csv('twitter1.csv')
    texts = data['comment'].astype(str).tolist()
    preprocessed_texts = list(nlp.pipe(texts, batch_size=100, n_process=8))
    data['comment'] = [' '.join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop]) for doc in preprocessed_texts]
    X = data['comment']
    y = data['emotion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB())
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    main()
