from __future__ import annotations

from src.core.constants import CHUNK_INDEX_KEY, CHUNK_PARENT_ID_KEY
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.rag.chunking.recursive_chunker import RecursiveChunker


class ParentChildChunker:
    """Produces two levels of chunks from each document.

    *Parent* chunks are large context windows stored alongside *child* chunks.
    Retrieval operates on children (small → precise embedding match); the
    pipeline then looks up the parent to return a richer context to the LLM.

    "Chunk()" returns both parents and children so the caller can
    persist the full set.  Children are identified by "metadata[" parent_id"]"
    pointing to the parent's "id".
    """

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        child_chunk_size: int = 400,
        overlap: int = 50,
    ) -> None:
        if child_chunk_size >= parent_chunk_size:
            raise ValueError("child_chunk_size must be smaller than parent_chunk_size")
        self._parent_splitter = RecursiveChunker(chunk_size=parent_chunk_size, overlap=overlap)
        self._child_splitter = RecursiveChunker(chunk_size=child_chunk_size, overlap=overlap)

    def chunk(self, document: Document) -> list[Chunk]:
        parents = self._parent_splitter.chunk(document)
        all_chunks: list[Chunk] = []
        child_index = 0

        for parent in parents:
            # Synthesize a temporary Document so the child splitter can use it.
            parent_doc = document.model_copy(update={"id": document.id, "content": parent.text})
            children = self._child_splitter.chunk(parent_doc)

            # Tag each child with a reference to its parent.
            tagged_children = [
                child.model_copy(
                    update={
                        "metadata": {
                            **child.metadata,
                            CHUNK_PARENT_ID_KEY: parent.id,
                            CHUNK_INDEX_KEY: child_index + j,
                        }
                    }
                )
                for j, child in enumerate(children)
            ]
            child_index += len(children)

            all_chunks.append(parent)
            all_chunks.extend(tagged_children)

        return all_chunks
