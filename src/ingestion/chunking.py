from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)


def chunk_documents(
    documents: list[Document],
) -> list[Document]:

    all_chunks = []

    for document in documents:

        language = document.metadata.get(
            "language"
        )

        if language == "python":

            splitter = (
                RecursiveCharacterTextSplitter
                .from_language(
                    language=Language.PYTHON,
                    chunk_size=500,
                    chunk_overlap=50,
                )
            )

        else:

            splitter = (
                RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                )
            )

        chunks = splitter.split_documents(
            [document]
        )

        for index, chunk in enumerate(chunks):

            chunk.metadata["chunk_index"] = index

        all_chunks.extend(chunks)

    return all_chunks