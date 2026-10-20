"""QLoRA fine-tune of SmolLM2 as the coach-brain decision model.

4-bit NF4 base weights (bitsandbytes) + LoRA adapters on all linear layers (PEFT), trained with the HF Trainer.
Loss is computed ONLY on the JSON answer tokens (prompt tokens are masked), so the model learns to emit the one-line
tool call that agent.js expects.

Kaggle (free T4/P100, internet on):
  python finetune_qlora.py                        # defaults: SmolLM2-360M-Instruct, 2 epochs
  python finetune_qlora.py --epochs 3 --lora-r 32

Outputs: out/adapter (LoRA weights + tokenizer). Metrics/params go to MLflow (file store at <repo>/mlruns by default).
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")      # Kaggle "T4 x2": avoid DataParallel with a 4-bit model
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,  # noqa: E402
                          TrainingArguments, set_seed)

from common import DATA_DIR, ROOT, file_sha, load_kb, load_rows, load_system, target_text, to_messages  # noqa: E402
from eval import git_sha  # noqa: E402


class DecisionDataset(Dataset):
    def __init__(self, rows, tok, max_len):
        kb, system = load_kb(), load_system()
        self.items, skipped = [], 0
        for r in rows:
            prompt = tok.apply_chat_template(to_messages(r, kb, system), tokenize=False, add_generation_prompt=True)
            p = tok(prompt, add_special_tokens=False)["input_ids"]
            t = tok(target_text(r["label"]) + tok.eos_token, add_special_tokens=False)["input_ids"]
            if len(p) + len(t) > max_len:
                skipped += 1
                continue
            self.items.append({"input_ids": p + t, "labels": [-100] * len(p) + t})
        if skipped:
            print(f"skipped {skipped} rows longer than {max_len} tokens")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collator(pad_id):
    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n], lab[i, :n], att[i, :n] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--train", default=str(DATA_DIR / "train.jsonl"))
    ap.add_argument("--val", default=str(DATA_DIR / "val.jsonl"))
    ap.add_argument("--out", default="out")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--experiment", default="ai-trainer-coach")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    set_seed(a.seed); random.seed(a.seed)
    assert torch.cuda.is_available(), "QLoRA needs a GPU (Kaggle: Settings -> Accelerator -> GPU T4/P100)"

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)          # T4/P100 have no bf16
    model = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=bnb, device_map={"": 0})
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    train_ds = DecisionDataset(load_rows(a.train), tok, a.max_len)
    val_ds = DecisionDataset(load_rows(a.val), tok, a.max_len)
    args = TrainingArguments(
        output_dir=os.path.join(a.out, "checkpoints"), num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.0, fp16=True, optim="paged_adamw_8bit",
        gradient_checkpointing=True, logging_steps=10, eval_strategy="epoch", save_strategy="no",
        report_to=[] if a.no_mlflow else ["mlflow"], remove_unused_columns=False, seed=a.seed)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=make_collator(tok.pad_token_id))

    run = None
    if not a.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"file:{ROOT / 'mlruns'}"))
            mlflow.set_experiment(a.experiment)
            run = mlflow.start_run(run_name=f"qlora-{a.base.split('/')[-1]}")
            mlflow.log_params({**{k: v for k, v in vars(a).items()}, "train_sha": file_sha(a.train), "n_train": len(train_ds)})
            mlflow.set_tags({"git_sha": git_sha(), "stage": "trained"})
        except ImportError:
            print("mlflow not installed: training without tracking")

    trainer.train()
    final = trainer.evaluate()
    print("final val:", final)
    adapter_dir = os.path.join(a.out, "adapter")
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    json.dump({"base": a.base, "val": final, "git_sha": git_sha()}, open(os.path.join(adapter_dir, "train_info.json"), "w"), indent=1)
    if run:
        import mlflow
        mlflow.log_artifacts(adapter_dir, "adapter")
        mlflow.end_run()
    print("saved", adapter_dir)


if __name__ == "__main__":
    main()
