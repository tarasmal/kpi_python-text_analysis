from util import get_text_from_file

def get_slice(text):
    return text[:40]


def main ():
    text = get_text_from_file('text.txt')
    slice = get_slice(text)
    print(f'Відрізок тексту: {slice}')
    print(f'Кількість літер t: {slice.count("t")}')
    print(f'Індекс літери s: {slice.index("s")}')
    print(f'Індекс з якого починається and: {slice.index("and")}')
    print(f'Upper: {slice.upper()}')
    print(f'Lower: {slice.lower()}')
    print(f'Title: {slice.title()}')
    print(f'Capitalize: {slice.capitalize()}')
    print(f"Replace and with or: {slice.replace('and', 'or')}")

main()