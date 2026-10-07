import json
from pathlib import Path


PROJE_KOKU = Path(__file__).resolve().parent.parent
DATA_KLASORU = PROJE_KOKU / "data"


VERITABANI_SEMASI = """
products(
    id,
    name,
    category,
    price,
    stock
)

customers(
    id,
    name,
    city
)

orders(
    id,
    customer_id,
    order_date,
    total_price
)

İlişki:
orders.customer_id -> customers.id
""".strip()


def formatla(
    giris_yolu,
    cikis_yolu
):

    sayac = 0

    with open(
        giris_yolu,
        "r",
        encoding="utf-8"
    ) as giris, open(
        cikis_yolu,
        "w",
        encoding="utf-8"
    ) as cikis:

        for satir in giris:

            veri = json.loads(
                satir
            )

            soru = veri["soru"]
            sql = veri["sql"]

            text = f"""
### Veritabanı Şeması
{VERITABANI_SEMASI}

### Soru
{soru}

### SQL
{sql}
""".strip()

            cikis.write(
                json.dumps(
                    {
                        "text": text
                    },
                    ensure_ascii=False
                )
                + "\n"
            )

            sayac += 1

    print(
        cikis_yolu.name,
        ":",
        sayac
    )


formatla(
    DATA_KLASORU / "train.jsonl",
    DATA_KLASORU / "train_formatted.jsonl"
)

formatla(
    DATA_KLASORU / "validation.jsonl",
    DATA_KLASORU / "validation_formatted.jsonl"
)

formatla(
    DATA_KLASORU / "test.jsonl",
    DATA_KLASORU / "test_formatted.jsonl"
)