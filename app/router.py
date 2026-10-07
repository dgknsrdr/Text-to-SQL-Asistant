class QuestionRouter:

    def __init__(self):

        self.database_keywords = [
            "stok",
            "fiyat",
            "müşteri",
            "sipariş",
            "harcama",
            "kategori",
            "şehir",
            "ortalama",
            "toplam",
            "kaç tane",
            "en pahalı",
            "en ucuz",
            "en fazla",
            "en az"
        ]

        self.document_keywords = [
            "iade",
            "garanti",
            "politika",
            "prosedür",
            "izin",
            "teslimat",
            "değişim",
            "kural",
            "koşul",
            "şart"
        ]

    def route(self, soru: str) -> str:

        soru = soru.lower()

        # Dokümanla ilgili güçlü bir kelime varsa
        # doğrudan RAG'e gönder.
        for kelime in self.document_keywords:

            if kelime in soru:
                return "rag"

        # Sonra database kelimelerine bak.
        for kelime in self.database_keywords:

            if kelime in soru:
                return "sql"

        # Varsayılan olarak SQL
        return "sql"