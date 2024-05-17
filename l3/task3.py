import ssl

import gensim
import numpy as np
from nltk.corpus import stopwords
import nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))



def task3(documents, cluster_labels):

    word_targets = ['mobile', 'Athens']
    word_clusters = {word: np.argmax(np.bincount(cluster_labels[[i for i, doc in enumerate(documents) if word in doc]]))
                     for word in word_targets}
    similar_words = {}
    for word, cluster in word_clusters.items():
        relevant_docs = [documents[i] for i in range(len(documents)) if cluster_labels[i] == cluster]
        relevant_sentences = [doc.split() for doc in relevant_docs]
        temp_model_w2v = gensim.models.Word2Vec(relevant_sentences, vector_size=20, window=3, min_count=1, workers=8, epochs=300)
        if word in temp_model_w2v.wv:
            similar_words[word] = [(similar_word, sim_score) for similar_word, sim_score in
                                   temp_model_w2v.wv.most_similar(word, topn=10)
                                   if similar_word not in stop_words]
    return similar_words