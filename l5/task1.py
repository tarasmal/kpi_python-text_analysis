import re
from pprint import pprint

import gensim
import nltk
import pandas as pd
import spacy
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords


def remove_links(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def to_human_readable_word_by_id(corpus):
    return [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100,
                                       update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


if __name__ == '__main__':
    nltk.download('stopwords')

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    data = pd.read_csv("science.csv")
    data['Comment'] = data['Comment'].apply(remove_links)

    data_words = list(sent_to_words(data["Comment"]))

    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    data_words_nostops = remove_stopwords(data_words)

    data_words_bigrams = make_bigrams(data_words_nostops)

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    id2word = corpora.Dictionary(data_lemmatized)

    texts = data_lemmatized

    corpus = [id2word.doc2bow(text) for text in texts]
    # print(corpus[:1])
    # print(to_human_readable_word_by_id(corpus[:1]))

    # lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
    #                                            id2word=id2word,
    #                                            num_topics=20,
    #                                            random_state=100,
    #                                            update_every=1,
    #                                            chunksize=100,
    #                                            passes=10,
    #                                            alpha='auto',
    #                                            per_word_topics=True)
    #
    # pprint(lda_model.print_topics())

    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)


    for num_topics, coherence in zip(range(2, 40, 6), coherence_values):
        print(f"Кількість тем: {num_topics}, Когерентність: {coherence}")

    best_model_index = coherence_values.index(max(coherence_values))
    best_model = model_list[best_model_index]
    print(f"Оптимальна кількість тем: {range(2, 40, 6)[best_model_index]}, Когерентність: {coherence_values[best_model_index]}")

    topics = best_model.print_topics(num_words=10)
    for topic in topics:
        pprint(topic)
