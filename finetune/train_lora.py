"""Entraînement LoRA d'un petit modèle instruct sur des échanges métier.

    python finetune/train_lora.py --data finetune/data/exemple.jsonl

Seules les réponses de l'assistant comptent dans la perte : le modèle apprend à
répondre comme l'entreprise, pas à recopier les questions des clients.

Valeurs par défaut choisies pour une GTX 1060 (6 Go, architecture Pascal) : un
modèle de 0,5 milliard de paramètres chargé en float32, parce que Pascal n'a pas
de bfloat16 et que l'entraînement en float16 pur diverge sans mise à l'échelle
des gradients. Sur une carte plus récente, passer à 1,5B ou 3B avec --bf16.
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def encode(example, tokenizer, max_len):
    messages = example["messages"]
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                                           add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    ids = tokenizer(full, add_special_tokens=False, truncation=True,
                    max_length=max_len)["input_ids"]
    labels = [-100] * min(prompt_len, len(ids)) + ids[prompt_len:]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="finetune/out")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    data = load_dataset("json", data_files=args.data, split="train")
    data = data.map(encode, fn_kwargs={"tokenizer": tokenizer, "max_len": args.max_len},
                    remove_columns=data.column_names)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            logging_steps=5,
            save_strategy="no",
            bf16=args.bf16,
            report_to=[],
        ),
        train_dataset=data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()

    adapter_dir = Path(args.out) / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"adaptateur écrit dans {adapter_dir}")


if __name__ == "__main__":
    main()
