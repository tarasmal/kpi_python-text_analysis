from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk import pos_tag
def split_text_into_sentences(text):
    return sent_tokenize(text)

def split_text_into_words(text):
    return [token for token in word_tokenize(text) if token.isalpha()]

def count_sentences_and_print_last(sentences):
    return len(sentences), sentences[-1]

def set_part_of_speech_for_words(words):
    tagged_tokens = pos_tag(words)
    return tagged_tokens

def count_most_repeatable_words(words):
    words_dict = dict(FreqDist(words))
    most_repeatable = {k: v for k, v in sorted(words_dict.items(), key=lambda item: item[1], reverse=True)[:10]}

    return most_repeatable

def print_part_of_task_output(print_part):
    print_part()
    print('\n')