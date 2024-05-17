from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer


def task2(documents):
    vectorizer_tfidf_adjusted = TfidfVectorizer(stop_words='english')
    X_tfidf_adjusted = vectorizer_tfidf_adjusted.fit_transform(documents)
    cluster_model_adjusted = AgglomerativeClustering(n_clusters=4, linkage='ward')
    cluster_labels = cluster_model_adjusted.fit_predict(X_tfidf_adjusted.toarray())
    return cluster_labels