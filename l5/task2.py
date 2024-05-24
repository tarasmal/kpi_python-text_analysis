import nltk
from nltk import ngrams
from nltk.corpus import gutenberg
from collections import Counter

nltk.download('gutenberg')
text = gutenberg.raw('chesterton-thursday.txt')

words = nltk.word_tokenize(text)
words = [word.lower() for word in words if word.isalpha()]

trigrams = ngrams(words, 3)
trigram_freq = Counter(trigrams)

most_common_trigrams = trigram_freq.most_common(10)
for trigram, freq in most_common_trigrams:
    print(f"Триграма: {' '.join(trigram)}, Частота: {freq}")
