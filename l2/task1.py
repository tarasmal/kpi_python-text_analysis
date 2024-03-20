import nltk
from util import get_text_from_file
from l2_util import *
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('all')


def main():
    text = get_text_from_file('text1.txt')
    sentences = split_text_into_sentences(text)
    words = split_text_into_words(text)
    number_of_sentences, last_sentence = count_sentences_and_print_last(sentences)
    most_repeatable_words = count_most_repeatable_words(words)
    words_with_parts_of_speech = set_part_of_speech_for_words(words)
    print_part_of_task_output(lambda: print(f'Кількість речень: {number_of_sentences}'))
    print_part_of_task_output(lambda: print(f'Останнє речення: {last_sentence}'))
    print_part_of_task_output(lambda: print(f'Найбільш повторювані 10 слів у тексті: {most_repeatable_words}'))
    print_part_of_task_output(lambda: print(f'Частини мови: {words_with_parts_of_speech}'))


main()
