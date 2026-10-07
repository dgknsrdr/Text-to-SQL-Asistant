from pathlib import Path

import torch

from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)

from trl import (
    SFTConfig,
    SFTTrainer
)


# =========================================================
# YOLLAR
# =========================================================

PROJE_KOKU = Path(__file__).resolve().parent.parent

TRAIN_DATA_YOLU = (
    PROJE_KOKU
    / "data"
    / "train_formatted.jsonl"
)

VALIDATION_DATA_YOLU = (
    PROJE_KOKU
    / "data"
    / "validation_formatted.jsonl"
)

MODEL_YOLU = (
    PROJE_KOKU
    / "models"
    / "qwen_text_to_sql_lora"
)


MODEL_ADI = "Qwen/Qwen3-1.7B"


# =========================================================
# CUDA KONTROL
# =========================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA bulunamadı."
    )


print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "VRAM:",
    round(
        torch.cuda.get_device_properties(0).total_memory
        / 1024 ** 3,
        2
    ),
    "GB"
)


# =========================================================
# 4 BIT QUANTIZATION
# =========================================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.float16
)


# =========================================================
# TOKENIZER
# =========================================================

print(
    "\nTokenizer yükleniyor..."
)


tokenizer = AutoTokenizer.from_pretrained(

    MODEL_ADI,

    trust_remote_code=True
)


if tokenizer.pad_token is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


tokenizer.padding_side = "right"


# =========================================================
# BASE MODEL
# =========================================================

print(
    "Base model yükleniyor..."
)


model = AutoModelForCausalLM.from_pretrained(

    MODEL_ADI,

    quantization_config=bnb_config,

    device_map={
        "": 0
    },

    trust_remote_code=True,

    dtype=torch.float16
)


model.config.use_cache = False


# =========================================================
# QLORA İÇİN MODELİ HAZIRLA
# =========================================================

model = prepare_model_for_kbit_training(

    model,

    use_gradient_checkpointing=True
)


# =========================================================
# LORA CONFIG
# =========================================================

lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)


# =========================================================
# LORA EKLE
# =========================================================

model = get_peft_model(

    model,

    lora_config
)


# =========================================================
# TRAINABLE PARAMETRELER FLOAT32
# =========================================================

for isim, parametre in model.named_parameters():

    if parametre.requires_grad:

        parametre.data = (
            parametre.data.float()
        )


# =========================================================
# PARAMETRE SAYISI
# =========================================================

trainable_params = 0
all_params = 0


for _, parametre in model.named_parameters():

    all_params += (
        parametre.numel()
    )

    if parametre.requires_grad:

        trainable_params += (
            parametre.numel()
        )


print(
    "\nTrainable parameters:",
    f"{trainable_params:,}"
)

print(
    "Total parameters:",
    f"{all_params:,}"
)

print(
    "Trainable %:",
    round(
        100
        * trainable_params
        / all_params,
        4
    )
)


# =========================================================
# DATASET
# =========================================================

print(
    "\nDataset yükleniyor..."
)


dataset = load_dataset(

    "json",

    data_files={

        "train":
            str(
                TRAIN_DATA_YOLU
            ),

        "validation":
            str(
                VALIDATION_DATA_YOLU
            )
    }
)


print(
    "Train örnek:",
    len(
        dataset["train"]
    )
)

print(
    "Validation örnek:",
    len(
        dataset["validation"]
    )
)


# =========================================================
# TRAINING CONFIG
# =========================================================

egitim_ayarlari = SFTConfig(

    # -----------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------

    output_dir=str(
        MODEL_YOLU
    ),


    # -----------------------------------------------------
    # EPOCH
    # -----------------------------------------------------

    num_train_epochs=3,


    # -----------------------------------------------------
    # RTX 3050 4GB İÇİN BATCH
    # -----------------------------------------------------

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,


    # -----------------------------------------------------
    # LEARNING RATE
    # -----------------------------------------------------

    learning_rate=2e-4,


    # -----------------------------------------------------
    # LOGGING
    # -----------------------------------------------------

    logging_steps=10,

    logging_first_step=True,


    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    eval_strategy="epoch",


    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    save_strategy="epoch",

    save_total_limit=2,


    # -----------------------------------------------------
    # EN İYİ MODELİ SEÇ
    # -----------------------------------------------------

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,


    # -----------------------------------------------------
    # TRACKING
    # -----------------------------------------------------

    report_to="none",


    # -----------------------------------------------------
    # TEXT
    # -----------------------------------------------------

    max_length=512,

    dataset_text_field="text",


    # -----------------------------------------------------
    # GPU / MEMORY
    # -----------------------------------------------------

    fp16=False,

    bf16=False,

    gradient_checkpointing=True,

    optim="paged_adamw_32bit",


    # -----------------------------------------------------
    # GRADIENT
    # -----------------------------------------------------

    max_grad_norm=1.0,


    # -----------------------------------------------------
    # SEED
    # -----------------------------------------------------

    seed=42
)


# =========================================================
# TRAINER
# =========================================================

trainer = SFTTrainer(

    model=model,

    processing_class=tokenizer,

    train_dataset=(
        dataset["train"]
    ),

    eval_dataset=(
        dataset["validation"]
    ),

    args=egitim_ayarlari
)


# =========================================================
# TRAIN
# =========================================================

print(
    "\nEğitim başlıyor..."
)


trainer.train()


# =========================================================
# FINAL VALIDATION
# =========================================================

print(
    "\nValidation değerlendiriliyor..."
)


validation_result = trainer.evaluate()


print(
    "\nValidation sonuçları:"
)


for key, value in validation_result.items():

    print(
        f"{key}: {value}"
    )


# =========================================================
# MODELİ KAYDET
# =========================================================

MODEL_YOLU.mkdir(

    parents=True,

    exist_ok=True
)


trainer.save_model(

    str(
        MODEL_YOLU
    )
)


tokenizer.save_pretrained(

    str(
        MODEL_YOLU
    )
)


# =========================================================
# CUDA CACHE
# =========================================================

torch.cuda.empty_cache()


print(
    "\nModel başarıyla kaydedildi:"
)

print(
    MODEL_YOLU
)