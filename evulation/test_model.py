from pathlib import Path
import re

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from peft import PeftModel


PROJE_KOKU = Path(__file__).resolve().parent.parent

MODEL_ADI = "Qwen/Qwen3-1.7B"

ADAPTER_YOLU = (
    PROJE_KOKU
    / "models"
    / "qwen_text_to_sql_lora"
)


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


def sql_temizle(cevap):

    cevap = cevap.replace(
        "```sql",
        ""
    ).replace(
        "```",
        ""
    ).strip()

    eslesme = re.search(
        r"(SELECT[\s\S]*?;)",
        cevap,
        re.IGNORECASE
    )

    if eslesme:
        return eslesme.group(1).strip()

    return cevap


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


print("Tokenizer yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(
    str(ADAPTER_YOLU)
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Base model yükleniyor...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ADI,
    quantization_config=bnb_config,
    device_map={"": 0},
    dtype=torch.float16
)


print("LoRA yükleniyor...")

model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_YOLU)
)

model.eval()


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
            max_new_tokens=150,
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


print("\nModel hazır.")

print(
    "Çıkmak için 'exit' yaz."
)


while True:

    print("\n" + "=" * 70)

    soru = input(
        "\nSoru: "
    )

    if soru.lower() == "exit":
        break

    if not soru.strip():
        continue

    sql = sql_uret(
        soru
    )

    print(
        "\nÜretilen SQL:"
    )

    print(
        sql
    )