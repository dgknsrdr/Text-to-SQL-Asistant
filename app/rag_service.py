from pathlib import Path

import numpy as np

from sentence_transformers import SentenceTransformer


PROJE_KOKU = Path(__file__).resolve().parent.parent

DOCUMENT_YOLU = (
    PROJE_KOKU
    / "documents"
    / "company_policy.txt"
)


class RAGService:

    def __init__(self):

        print("RAG embedding modeli yükleniyor...")

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )


        self.chunks = self.dokuman_yukle()

        self.embeddings = (
            self.embedding_model.encode(
                self.chunks,
                normalize_embeddings=True
            )
        )

        print(
            f"RAG hazır. Chunk sayısı: {len(self.chunks)}"
        )


    def dokuman_yukle(self):

        with open(
            DOCUMENT_YOLU,
            "r",
            encoding="utf-8"
        ) as dosya:

            metin = dosya.read()


        return self.chunkla(
            metin
        )


    def chunkla(
        self,
        metin,
        chunk_size=300,
        overlap=50
    ):

        chunks = []

        baslangic = 0

        while baslangic < len(metin):

            bitis = (
                baslangic
                + chunk_size
            )

            chunk = metin[
                baslangic:bitis
            ].strip()

            if chunk:
                chunks.append(
                    chunk
                )

            baslangic += (
                chunk_size
                - overlap
            )

        return chunks


    def ara(
        self,
        soru,
        top_k=3
    ):

        soru_embedding = (
            self.embedding_model.encode(
                [soru],
                normalize_embeddings=True
            )[0]
        )


        skorlar = np.dot(
            self.embeddings,
            soru_embedding
        )


        indexler = np.argsort(
            skorlar
        )[::-1][:top_k]


        sonuclar = []

        for index in indexler:

            sonuclar.append(
                {
                    "text":
                        self.chunks[index],

                    "score":
                        float(
                            skorlar[index]
                        )
                }
            )

        return sonuclar