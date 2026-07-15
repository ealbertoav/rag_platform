# Page-boundary chunking as an exclusive segment-first strategy

T-241 needed a way to attach page numbers to chunks from paginated sources (PDF). We implemented `PageAwareChunker` as a new exclusive `chunking.strategy: "page"` value, structured like `SectionChunker` (T-240): it segments `document.metadata["pages"]` into per-page `Document` copies first, then chunks each page independently with `RecursiveChunker` — so a chunk can never straddle two pages. Page numbers are 1-indexed (matching the existing Docling-sourced `CHUNK_PAGE_KEY` convention used for table/figure metadata), and `CHUNK_PAGE_KEY` is omitted entirely — never defaulted to `1` — for sources with no page metadata (DOCX, HTML, Markdown, PPTX).

## Considered Options

- A composable post-chunk tagger that maps any strategy's output chunks back to page numbers via character-offset lookup in `document.content`. Rejected: chunks near a page boundary can straddle two pages (ambiguous page number), and `overlap > 0` makes chunk text non-uniquely locatable via substring search. Segmenting by page first avoids both problems entirely, at the cost of not being composable with the `section` strategy in the same pass — a chunk currently gets a page tag *or* a section tag, not both.
