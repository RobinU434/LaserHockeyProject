def generate_separator(content: str, width: int, fill_char: str = "="):
    assert (
        len(content) + 2 < width
    ), f"Content: {content} + 2 white space has to be bigger than desired with"
    content = " " + content.strip(" ") + " "
    fill_len = (width - len(content)) // 2
    content = fill_char * fill_len + content
    content = content + fill_char * (width - len(content))
    return content


