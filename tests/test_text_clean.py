from text_clean import TextCleaner, clean_text


def test_clean_text_handles_none_and_noise():
    assert clean_text(None) == ""
    assert clean_text("HELLO   https://example.com test@example.com") == "hello"


def test_text_cleaner_transform_returns_list():
    cleaner = TextCleaner()
    assert cleaner.transform(["Hello", None]) == ["hello", ""]
