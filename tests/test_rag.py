import pytest

from study_squad.rag import Hit, TextbookIndex, chunk_pages, format_context


def test_chunks_overlap_and_keep_page_numbers():
    words = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_pages([(7, words)], chunk_words=200, overlap=50)
    assert [c.page for c in chunks] == [7, 7, 7]
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert len(first) == 200
    assert first[-50:] == second[:50]  # overlap preserved
    assert chunks[-1].text.split()[-1] == "w499"  # nothing lost at the end


def test_short_page_is_one_chunk():
    assert len(chunk_pages([(1, "just a few words")])) == 1


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_pages([(1, "a b c")], chunk_words=10, overlap=10)


def test_format_context_cites_pages():
    text = format_context([Hit("alpha", 2, 0.9), Hit("beta", 5, 0.8)])
    assert text == "[p. 2] alpha\n\n[p. 5] beta"


def test_index_and_search_pdf(make_pdf, hash_embedding):
    pdf = make_pdf(
        [
            "K-means clustering partitions data into k groups around centroids. " * 15,
            "Decision trees split nodes using entropy and information gain. " * 15,
        ]
    )
    index = TextbookIndex(embedding_function=hash_embedding)
    result = index.index_pdf(pdf, "notes.pdf")
    assert result.pages == 2 and result.chunks >= 2 and not result.cached

    hits = index.search(result.collection, "entropy information gain decision trees", k=1)
    assert hits[0].page == 2

    again = index.index_pdf(pdf, "notes.pdf")
    assert again.cached and again.chunks == result.chunks


def test_pdf_without_text_raises(make_pdf, hash_embedding):
    blank = make_pdf([""])
    with pytest.raises(ValueError, match="No text"):
        TextbookIndex(embedding_function=hash_embedding).index_pdf(blank, "scan.pdf")
