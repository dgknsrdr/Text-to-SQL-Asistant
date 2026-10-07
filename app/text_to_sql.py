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

ADAPTER_YOLU = (
    PROJE_KOKU
    / "models"
    / "qwen_text_to_sql_lora"
)

MODEL_ADI = "Qwen/Qwen3-1.7B"


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


class TextToSQLModel:

    def __init__(self):

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

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(ADAPTER_YOLU)
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        print("Base Qwen yükleniyor...")

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ADI,
            quantization_config=bnb_config,
            device_map={"": 0},
            dtype=torch.float16
        )

        print("Fine-tuned LoRA yükleniyor...")

        self.model = PeftModel.from_pretrained(
            base_model,
            str(ADAPTER_YOLU)
        )

        self.model.eval()

        print("Text-to-SQL modeli hazır.")


    def sql_temizle(self, cevap: str):

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


    def sql_uret(self, soru: str):

        prompt = f"""
### Veritabanı Şeması
{VERITABANI_SEMASI}

### Soru
{soru}

### SQL
""".strip()

        girdiler = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():

            cikti = self.model.generate(
                **girdiler,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.eos_token_id
                )
            )

        yeni_tokenlar = cikti[
            0,
            girdiler["input_ids"].shape[1]:
        ]

        cevap = self.tokenizer.decode(
            yeni_tokenlar,
            skip_special_tokens=True
        )

        return self.sql_temizle(
            cevap
        )