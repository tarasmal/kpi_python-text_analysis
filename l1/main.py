import re
from util import get_text_from_file



def mask_phone_numbers(match):
    phone = match.group()

    def replace_digits(phone_to_be_masked):
        pattern = r'\d'
        new_phone = ""
        number_counter = 0
        for ch in phone_to_be_masked:
            if re.match(pattern, ch):
                if number_counter < 2:
                    new_phone += ch
                    number_counter += 1
                else:
                    new_phone += "*"
            else:
                new_phone += ch
        return new_phone

    return replace_digits(phone)


phone_number_pattern = r'(?:\(\d{1,3}\)|\d{1,3})-\d{2}-\d{2}'


def is_phone(word):
    return True if re.match(phone_number_pattern, word) else False


def change_phone_numbers(text):
    for index in range(len(text)):
        if is_phone(text[index]):
            masked_phone = re.sub(phone_number_pattern, mask_phone_numbers, text[index])
            text[index] = masked_phone
    return " ".join(text)


def main():
    text = get_text_from_file()
    split_text = text.split(" ")
    masked_text = change_phone_numbers(split_text)
    print(f'Text: {text}')
    print(f'Masked text: {masked_text}')


main()
