def get_text_from_file(file_path="./text.txt"):
    with open(file_path, 'r', encoding="utf-8") as file:
        file_content = file.read()
    return file_content
