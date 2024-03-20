from nltk.corpus import brown
from l2_util import set_part_of_speech_for_words, print_part_of_task_output
from functools import reduce


def get_second_text():
    second = brown.sents(categories='science_fiction')[1]
    return second

def get_words_count():
    texts = brown.sents(categories='science_fiction')
    return reduce(lambda a, b: a + len(b), texts, 0)

def delete_verbs_from_text(words_with_tags):
    return " ".join([word for word, tag in words_with_tags if not tag.startswith('VB')])


def main():
    words = get_second_text()
    count = get_words_count()
    words_with_tags = set_part_of_speech_for_words(words)
    text_without_verbs = delete_verbs_from_text(words_with_tags)
    print_part_of_task_output(lambda: print(f'Загальна кількість слів у категорії science_fiction: {count}'))
    print_part_of_task_output(lambda: print(f'Оригінальний текст: {" ".join(words)}'))
    print_part_of_task_output(lambda: print(f'Другий текст без дієслів: {text_without_verbs}'))


main()
