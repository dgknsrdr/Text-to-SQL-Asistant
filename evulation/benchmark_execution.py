import json
import re
from decimal import Decimal
from pathlib import Path

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from peft import PeftModel

from database.connection import veritabani_baglantisi


# =========================================================
# YOLLAR
# =========================================================

PROJE_KOKU = Path(__file__).resolve().parent.parent

TEST_YOLU = (
    PROJE_KOKU
    / "data"
    / "test.jsonl"
)

ADAPTER_YOLU = (
    PROJE_KOKU
    / "models"
    / "qwen_text_to_sql_lora"
)

MODEL_ADI = "Qwen/Qwen3-1.7B"


# =========================================================
# VERİTABANI ŞEMASI
# =========================================================

SCHEMA = """
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


# =========================================================
# SQL TEMİZLE
# =========================================================

def sql_temizle(cevap: str) -> str:

    cevap = cevap.replace(
        "```sql",
        ""
    )

    cevap = cevap.replace(
        "```",
        ""
    )

    cevap = cevap.strip()

    eslesme = re.search(
        r"(SELECT[\s\S]*?;)",
        cevap,
        re.IGNORECASE
    )

    if eslesme:
        return eslesme.group(1).strip()

    return cevap


# =========================================================
# GÜVENLİK
# =========================================================

def sorgu_guvenli_mi(sql: str) -> bool:

    sql_lower = sql.strip().lower()

    yasakli = [
        "delete",
        "update",
        "insert",
        "drop",
        "alter",
        "truncate",
        "create"
    ]

    if not sql_lower.startswith("select"):
        return False

    for komut in yasakli:
        if komut in sql_lower:
            return False

    return True


# =========================================================
# POSTGRESQL SORGU ÇALIŞTIR
# =========================================================

def sql_calistir(sql: str):

    if not sorgu_guvenli_mi(sql):

        raise ValueError(
            "Güvenli olmayan SQL üretildi."
        )

    with veritabani_baglantisi() as baglanti:

        with baglanti.cursor() as cursor:

            cursor.execute(sql)

            sonuc = cursor.fetchall()

    return sonuc


# =========================================================
# SONUÇ NORMALIZE
# =========================================================

def degeri_normalize_et(deger):

    if isinstance(deger, Decimal):
        return float(deger)

    return deger


def sonucu_normalize_et(sonuc):

    normalize = []

    for satir in sonuc:

        yeni_satir = tuple(
            degeri_normalize_et(deger)
            for deger in satir
        )

        normalize.append(
            yeni_satir
        )

    # ORDER BY olmayan sorgularda sıra farklı olabilir.
    try:
        return sorted(normalize)
    except TypeError:
        return normalize


# =========================================================
# MODEL YÜKLE
# =========================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA bulunamadı."
    )


print(
    "GPU:",
    torch.cuda.get_device_name(0)
)


bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.float16
)


print("\nTokenizer yükleniyor...")


tokenizer = AutoTokenizer.from_pretrained(
    str(ADAPTER_YOLU)
)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Base Qwen yükleniyor...")


base_model = AutoModelForCausalLM.from_pretrained(

    MODEL_ADI,

    quantization_config=bnb_config,

    device_map={
        "": 0
    },

    dtype=torch.float16
)


print("Fine-tuned LoRA yükleniyor...")


model = PeftModel.from_pretrained(

    base_model,

    str(ADAPTER_YOLU)
)


model.eval()


# =========================================================
# SQL ÜRET
# =========================================================

def sql_uret(soru: str):

    prompt = f"""
### Veritabanı Şeması
{SCHEMA}

### Soru
{soru}

### SQL
""".strip()

    girdiler = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(
        "cuda"
    )

    with torch.no_grad():

        cikti = model.generate(

            **girdiler,

            max_new_tokens=120,

            do_sample=False,

            pad_token_id=tokenizer.eos_token_id
        )

    yeni_tokenlar = cikti[
        0,
        girdiler["input_ids"].shape[1]:
    ]

    cevap = tokenizer.decode(

        yeni_tokenlar,

        skip_special_tokens=True
    )

    return sql_temizle(
        cevap
    )


# =========================================================
# TEST SET
# =========================================================

testler = []

with open(
    TEST_YOLU,
    "r",
    encoding="utf-8"
) as dosya:

    for satir in dosya:

        testler.append(
            json.loads(
                satir
            )
        )


# =========================================================
# EXECUTION BENCHMARK
# =========================================================

dogru = 0
hata = 0


for index, test in enumerate(
    testler,
    start=1
):

    soru = test["soru"]

    beklenen_sql = test["sql"]

    print("\n" + "=" * 70)
    print(f"TEST {index}")
    print("=" * 70)

    print(
        "\nSoru:"
    )

    print(
        soru
    )


    # Model SQL üretir
    model_sql = sql_uret(
        soru
    )


    print(
        "\nBeklenen SQL:"
    )

    print(
        beklenen_sql
    )


    print(
        "\nModel SQL:"
    )

    print(
        model_sql
    )


    try:

        beklenen_sonuc = sql_calistir(
            beklenen_sql
        )

        model_sonuc = sql_calistir(
            model_sql
        )


        beklenen_normal = sonucu_normalize_et(
            beklenen_sonuc
        )

        model_normal = sonucu_normalize_et(
            model_sonuc
        )


        basarili = (
            beklenen_normal
            == model_normal
        )


        if basarili:

            dogru += 1

            print(
                "\n✅ EXECUTION DOĞRU"
            )

        else:

            print(
                "\n❌ EXECUTION YANLIŞ"
            )

            print(
                "\nBeklenen sonuç:"
            )

            print(
                beklenen_normal
            )

            print(
                "\nModel sonucu:"
            )

            print(
                model_normal
            )


    except Exception as hata_mesaji:

        hata += 1

        print(
            "\n⚠️ SQL ÇALIŞTIRMA HATASI:"
        )

        print(
            hata_mesaji
        )


# =========================================================
# SONUÇ
# =========================================================

toplam = len(
    testler
)

yanlis = (
    toplam
    - dogru
)


accuracy = (
    dogru
    / toplam
    * 100
)


print("\n\n")

print(
    "=" * 70
)

print(
    "EXECUTION BENCHMARK SONUCU"
)

print(
    "=" * 70
)


print(
    f"Toplam test: {toplam}"
)

print(
    f"Doğru execution: {dogru}"
)

print(
    f"Yanlış / hata: {yanlis}"
)

print(
    f"SQL çalıştırma hatası: {hata}"
)

print(
    f"Execution Accuracy: %{accuracy:.2f}"
)