import json
import re
from pathlib import Path

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from peft import PeftModel


# =========================================================
# YOLLAR
# =========================================================

PROJE_KOKU = Path(__file__).resolve().parent.parent

TEST_YOLU = PROJE_KOKU / "data" / "test.jsonl"

ADAPTER_YOLU = (
    PROJE_KOKU
    / "models"
    / "qwen_text_to_sql_lora"
)

MODEL_ADI = "Qwen/Qwen3-1.7B"


# =========================================================
# ŞEMA
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
# SQL NORMALIZE
# =========================================================

def sql_normalize(sql: str):

    sql = sql.strip()

    sql = sql.replace("\n", " ")

    sql = re.sub(
        r"\s+",
        " ",
        sql
    )

    sql = sql.rstrip(";")

    return sql.lower()


# =========================================================
# MODEL
# =========================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA bulunamadı.")


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


print("Base model yükleniyor...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ADI,
    quantization_config=bnb_config,
    device_map={"": 0},
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

def sql_uret(soru):

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
    ).to("cuda")

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

    eslesme = re.search(
        r"(SELECT[\s\S]*?;)",
        cevap,
        re.IGNORECASE
    )

    if eslesme:
        return eslesme.group(1).strip()

    return cevap.strip()


# =========================================================
# TESTLERİ OKU
# =========================================================

testler = []

with open(
    TEST_YOLU,
    "r",
    encoding="utf-8"
) as dosya:

    for satir in dosya:

        testler.append(
            json.loads(satir)
        )


# =========================================================
# BENCHMARK
# =========================================================

dogru = 0


for index, test in enumerate(
    testler,
    start=1
):

    soru = test["soru"]

    beklenen_sql = test["sql"]

    uretilen_sql = sql_uret(
        soru
    )


    beklenen_normal = sql_normalize(
        beklenen_sql
    )

    uretilen_normal = sql_normalize(
        uretilen_sql
    )


    basarili = (
        beklenen_normal
        == uretilen_normal
    )


    if basarili:
        dogru += 1


    print("\n" + "=" * 60)

    print(
        f"TEST {index}"
    )

    print(
        "\nSoru:",
        soru
    )

    print(
        "\nBeklenen:"
    )

    print(
        beklenen_sql
    )

    print(
        "\nModel:"
    )

    print(
        uretilen_sql
    )

    print(
        "\nSonuç:",
        "✅ DOĞRU"
        if basarili
        else "❌ YANLIŞ"
    )


# =========================================================
# SONUÇ
# =========================================================

toplam = len(testler)

accuracy = (
    dogru
    / toplam
    * 100
)


print("\n\n")
print("=" * 60)
print("BENCHMARK SONUCU")
print("=" * 60)

print(
    f"Toplam test: {toplam}"
)

print(
    f"Doğru: {dogru}"
)

print(
    f"Yanlış: {toplam - dogru}"
)

print(
    f"Exact Match Accuracy: %{accuracy:.2f}"
)