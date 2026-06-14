from text_clean import TextCleaner, clean_text, strip_source_artifacts


def test_clean_text_handles_none_and_noise():
    assert clean_text(None) == ""
    assert clean_text("HELLO   https://example.com test@example.com") == "hello"


def test_text_cleaner_transform_returns_list():
    cleaner = TextCleaner()
    assert cleaner.transform(["Hello", None]) == ["hello", ""]


def test_strip_source_artifacts_removes_dateline_and_mentions():
    raw = "WASHINGTON (Reuters) - The Senate voted, Reuters reported."
    stripped = strip_source_artifacts(raw)
    assert "reuters" not in stripped.lower()
    assert "(Reuters)" not in stripped
    assert "senate voted" in stripped.lower()


def test_strip_source_artifacts_handles_no_dateline():
    assert strip_source_artifacts("A normal headline with no source tag") == (
        "A normal headline with no source tag"
    )
    assert strip_source_artifacts(None) == ""


def test_text_cleaner_strip_source_flag_is_applied():
    cleaner = TextCleaner(strip_source=True)
    out = cleaner.transform(["LONDON (Reuters) - Markets rose on Monday."])
    assert "reuters" not in out[0]
    assert "markets rose" in out[0]


def test_text_cleaner_handles_pickle_without_strip_source_attr():
    # Simulate a pipeline pickled before the strip_source flag existed.
    cleaner = TextCleaner()
    del cleaner.strip_source
    assert cleaner.transform(["WASHINGTON (Reuters) - Senate voted today."]) == [
        "washington (reuters) - senate voted today."
    ]
