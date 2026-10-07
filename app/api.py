from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.text_to_sql import TextToSQLModel
from app.router import QuestionRouter
from app.rag_service import RAGService

from database.sql_tool import sql_calistir


PROJE_KOKU = Path(__file__).resolve().parent.parent
FRONTEND = PROJE_KOKU / "frontend"


app = FastAPI(
    title="Enterprise AI Copilot",
    version="1.0.0"
)


class SoruRequest(BaseModel):
    soru: str


# Servisler uygulama açılırken bir kere yüklenecek.
text_to_sql_model = None
rag_service = None
router = None


@app.on_event("startup")
def startup():

    global text_to_sql_model
    global rag_service
    global router

    print("Router hazırlanıyor...")
    router = QuestionRouter()

    print("RAG sistemi yükleniyor...")
    rag_service = RAGService()

    print("Text-to-SQL modeli yükleniyor...")
    text_to_sql_model = TextToSQLModel()

    print("Enterprise AI Copilot hazır.")


def sonuc_metni_olustur(sonuclar):

    if not sonuclar:
        return "Bu sorguya uygun kayıt bulunamadı."

    if len(sonuclar) == 1:

        kayit = sonuclar[0]

        return ", ".join(
            f"{kolon}: {deger}"
            for kolon, deger in kayit.items()
        )

    satirlar = [
        f"Toplam {len(sonuclar)} kayıt bulundu."
    ]

    for index, kayit in enumerate(
        sonuclar,
        start=1
    ):

        metin = ", ".join(
            f"{kolon}: {deger}"
            for kolon, deger in kayit.items()
        )

        satirlar.append(
            f"{index}. {metin}"
        )

    return "\n".join(satirlar)


def rag_cevabi_olustur(sonuclar):

    if not sonuclar:
        return "Bu soruyla ilgili dokümanlarda bilgi bulunamadı."

    # Şimdilik en yüksek benzerlik skoruna sahip chunk'ı cevap olarak kullanıyoruz.
    en_iyi_sonuc = sonuclar[0]

    return en_iyi_sonuc["text"]


@app.get("/health")
def health():

    return {
        "status": "ok",
        "text_to_sql_loaded": text_to_sql_model is not None,
        "rag_loaded": rag_service is not None,
        "router_loaded": router is not None
    }


@app.post("/api/ask")
def ask(request: SoruRequest):

    soru = request.soru.strip()

    if not soru:

        return {
            "success": False,
            "error": "Soru boş olamaz."
        }

    if router is None:

        return {
            "success": False,
            "error": "Router henüz hazır değil."
        }

    try:

        route = router.route(soru)

        # ==========================================================
        # DATABASE / TEXT-TO-SQL
        # ==========================================================

        if route == "sql":

            if text_to_sql_model is None:

                return {
                    "success": False,
                    "error": "Text-to-SQL modeli henüz hazır değil."
                }

            sql = text_to_sql_model.sql_uret(
                soru
            )

            sonuc = sql_calistir(
                sql
            )

            cevap = sonuc_metni_olustur(
                sonuc
            )

            return {
                "success": True,
                "route": "database",
                "question": soru,
                "sql": sql,
                "answer": cevap,
                "data": sonuc
            }

        # ==========================================================
        # RAG / DOCUMENT
        # ==========================================================

        if route == "rag":

            if rag_service is None:

                return {
                    "success": False,
                    "error": "RAG sistemi henüz hazır değil."
                }

            sonuclar = rag_service.ara(
                soru,
                top_k=3
            )

            cevap = rag_cevabi_olustur(
                sonuclar
            )

            return {
                "success": True,
                "route": "document",
                "question": soru,
                "sql": None,
                "answer": cevap,
                "data": sonuclar
            }

        return {
            "success": False,
            "error": "Soru türü belirlenemedi."
        }

    except Exception as hata:

        print(
            "Hata:",
            hata
        )

        return {
            "success": False,
            "error": str(hata)
        }


@app.get("/")
def index():

    return FileResponse(
        FRONTEND / "index.html"
    )


app.mount(
    "/static",
    StaticFiles(
        directory=FRONTEND
    ),
    name="static"
)