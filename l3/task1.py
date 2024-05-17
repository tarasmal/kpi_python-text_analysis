from sklearn.feature_extraction.text import CountVectorizer


def task1(documents):
    vectorizer_bow = CountVectorizer()
    X_bow = vectorizer_bow.fit_transform(documents)
    bow_feature_names = vectorizer_bow.get_feature_names_out()
    mariner_vector_bow = X_bow.toarray()[:,
                         bow_feature_names.tolist().index('mariner')] if 'mariner' in bow_feature_names else None
    return mariner_vector_bow
