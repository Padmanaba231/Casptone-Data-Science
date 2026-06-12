"""
bert_ner_finetune.py
====================
Fine-tuning BERT-NER menggunakan dataset yang sudah diproses oleh dataset_loader.py.

✨ PERUBAHAN v3 (SKILLS & YEARS_EXPERIENCE Fix):
  - compute_class_weights: tambah parameter skill_boost (default 5.0)
    → SKILLS mendapat boost khusus karena polisemi (kata sama bisa jadi entity berbeda)
  - compute_class_weights: tambah parameter ye_boost (default 10.0)
    → YEARS_EXPERIENCE dapat extra boost karena sangat jarang di data asli
  - finetune(): expose skill_boost dan ye_boost sebagai parameter
  - Semua parameter baru tersedia via argparse (--skill_boost, --ye_boost)
  - Catatan log diperjelas untuk monitoring class weights

Input yang diharapkan (output dari dataset_loader.py v3):
  dataset/processed/train_dataset.json
  dataset/processed/test_dataset.json
  dataset/processed/label_config.json   ← harus punya field "entity_counts"

Cara pakai:
  # Step 1 — Siapkan dataset dulu
  python dataset_loader.py

  # Step 2 — Fine-tuning (dengan default boost)
  python bert_ner_finetune.py

  # Step 2 — Fine-tuning (custom boost)
  python bert_ner_finetune.py --skill_boost 5.0 --ye_boost 15.0
"""

import os
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from datasets import Dataset as HFDataset
import evaluate


# ─────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIG = {
    # Model BERT — pilih salah satu:
    # "bert-base-uncased"                    → CV Bahasa Inggris
    # "indobenchmark/indobert-base-p1"       → CV Bahasa Indonesia
    # "cahya/bert-base-indonesian-NER"       → IndoBERT pretrained NER
    "base_model": "bert-base-uncased",
    "max_length": 512,
    "output_dir": "./bert_ner_cv_model",
}

TRAIN_CONFIG = {
    "num_train_epochs":             20,
    "per_device_train_batch_size":  4,
    "per_device_eval_batch_size":   8,
    "learning_rate":                3e-5,
    "warmup_ratio":                 0.1,
    "weight_decay":                 0.01,
    "evaluation_strategy":          "epoch",
    "save_strategy":                "epoch",
    "load_best_model_at_end":       True,
    "metric_for_best_model":        "eval_f1",
    "greater_is_better":            True,
    "logging_steps":                50,
    "fp16":                         torch.cuda.is_available(),
    "report_to":                    "none",
    "early_stopping_patience":      5,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLASS WEIGHTS — untuk WeightedTrainer
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    label_list: list,
    entity_counts: dict,
    o_weight: float    = 0.1,
    max_weight: float  = 50.0,
    skill_boost: float = 5.0,    # ✨ BARU v3: boost SKILLS karena polisemi tinggi
    ye_boost: float    = 10.0,   # ✨ BARU v3: boost YEARS_EXPERIENCE karena sangat langka
) -> torch.Tensor:
    """
    Hitung class weight per label menggunakan inverse-frequency weighting.

    ✨ v3 Perubahan:
    - SKILLS mendapat boost tambahan (skill_boost) karena meski jumlahnya banyak
      di training, kata-kata SKILLS sering ambigu (misal "Python" = kota atau skill).
      Tanpa boost, model cenderung under-predict SKILLS.
    - YEARS_EXPERIENCE mendapat boost ekstra (ye_boost) karena data aslinya
      sangat sedikit (42 entitas). Meski sudah diaugmentasi, model masih kesulitan
      karena pola kalimatnya sangat beragam.

    Rumus dasar:
        weight(entity) = max_freq / freq(entity)

    Token "O" diberi bobot sangat kecil karena sangat dominan.
    B- dan I- dari entitas yang sama mendapat bobot identik.

    Args:
        label_list   : List semua label BIO (["O", "B-NAME", "I-NAME", ...])
        entity_counts: Dict {entity_name: count} dari label_config.json
        o_weight     : Bobot untuk token "O" (default: 0.1)
        max_weight   : Cap maksimum bobot agar tidak terlalu ekstrem (default: 50.0)
        skill_boost  : Faktor pengali tambahan untuk SKILLS (default: 5.0)
        ye_boost     : Faktor pengali tambahan untuk YEARS_EXPERIENCE (default: 10.0)

    Returns:
        torch.Tensor shape (num_labels,) berisi float32
    """
    if not entity_counts:
        print("  [WARNING] entity_counts kosong, semua kelas diberi bobot 1.0")
        return torch.ones(len(label_list), dtype=torch.float)

    max_freq = max(entity_counts.values())
    weights  = []

    for label in label_list:
        if label == "O":
            weights.append(o_weight)
        else:
            entity = label[2:]  # strip "B-" atau "I-"
            freq   = entity_counts.get(entity, 1)
            w      = max_freq / (freq + 1e-9)

            # ✨ BARU v3: boost SKILLS
            # SKILLS punya banyak data tapi F1 rendah (0.133) karena polisemi.
            # Boost memaksa model lebih "peduli" pada kesalahan prediksi SKILLS.
            if entity == "SKILLS":
                w = w * skill_boost
                w = max(w, 3.0)   # minimal 3x bobot, walau frekuensi tinggi

            # ✨ BARU v3: boost YEARS_EXPERIENCE
            # Data asli hanya 42 entitas. Meski sudah diaugmentasi 800+,
            # model masih butuh sinyal loss yang lebih kuat untuk kelas ini.
            elif entity == "YEARS_EXPERIENCE":
                w = w * ye_boost
                w = max(w, 5.0)   # minimal 5x bobot

            w = min(w, max_weight)
            weights.append(w)

    return torch.tensor(weights, dtype=torch.float)


def log_class_weights(label_list: list, weights: torch.Tensor):
    """Cetak tabel class weights untuk monitoring."""
    print("\n  📊 Class Weights per Label:")
    print(f"  {'Label':25s}  {'Weight':>8s}")
    print(f"  {'-'*35}")

    label_weight_pairs = sorted(
        zip(label_list, weights.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    for label, w in label_weight_pairs:
        bar = "█" * min(int(w), 30)
        print(f"  {label:25s}  {w:8.2f}  {bar}")

    # ✨ BARU: tampilkan ringkasan boost yang diterapkan
    skills_w = next((w for l, w in zip(label_list, weights.tolist()) if l == "B-SKILLS"), None)
    ye_w     = next((w for l, w in zip(label_list, weights.tolist()) if l == "B-YEARS_EXPERIENCE"), None)
    print(f"\n  ⚡ Boost summary:")
    print(f"     B-SKILLS            : {skills_w:.2f}" if skills_w else "     B-SKILLS            : N/A")
    print(f"     B-YEARS_EXPERIENCE  : {ye_w:.2f}"     if ye_w     else "     B-YEARS_EXPERIENCE  : N/A")


# ─────────────────────────────────────────────────────────────────────────────
# 3. WEIGHTED TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """
    Custom Trainer yang mengganti CrossEntropyLoss standar dengan
    weighted CrossEntropyLoss untuk menangani class imbalance.

    Bobot kelas (class_weights) dihitung dari frekuensi entitas:
        - Entitas langka → bobot tinggi (lebih diperhatikan model)
        - Entitas dominan (SKILLS, O) → bobot relatif rendah,
          tapi SKILLS mendapat boost tambahan karena polisemi

    ignore_index=-100 memastikan subword tokens dan padding
    tidak berkontribusi ke loss.
    """

    def __init__(self, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
        else:
            weights = None

        loss_fct = nn.CrossEntropyLoss(
            weight=weights,
            ignore_index=-100,
        )

        loss = loss_fct(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────────────────────────────────────
# 4. LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_processed_dataset(train_path: str, test_path: str, label_config_path: str = None):
    """
    Load dataset yang sudah diproses oleh dataset_loader.py v3.

    Args:
        train_path        : Path ke train_dataset.json
        test_path         : Path ke test_dataset.json
        label_config_path : Path ke label_config.json

    Returns:
        (train_records, test_records, label_list, label2id, id2label, entity_counts)
    """
    print(f"[INFO] Memuat training set : {train_path}")
    with open(train_path, "r", encoding="utf-8") as f:
        train_records = json.load(f)

    print(f"[INFO] Memuat test set     : {test_path}")
    with open(test_path, "r", encoding="utf-8") as f:
        test_records = json.load(f)

    print(f"[INFO] Training : {len(train_records)} records")
    print(f"[INFO] Test     : {len(test_records)} records")

    entity_counts = {}

    if label_config_path and Path(label_config_path).exists():
        with open(label_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        label_list    = cfg["label_list"]
        label2id      = cfg["label2id"]
        id2label      = {int(k): v for k, v in cfg["id2label"].items()}
        entity_counts = cfg.get("entity_counts", {})

        print(f"[INFO] Label config dari  : {label_config_path}")
        if entity_counts:
            print(f"[INFO] entity_counts ditemukan ({len(entity_counts)} entitas) → dipakai untuk class weighting")
            # ✨ Tampilkan SKILLS dan YEARS_EXPERIENCE secara eksplisit untuk verifikasi
            skills_cnt = entity_counts.get("SKILLS", 0)
            ye_cnt     = entity_counts.get("YEARS_EXPERIENCE", 0)
            print(f"[INFO]   SKILLS count          : {skills_cnt}")
            print(f"[INFO]   YEARS_EXPERIENCE count: {ye_cnt}")
        else:
            print(f"[WARNING] entity_counts tidak ada di label_config.json!")
            print(f"          Jalankan dataset_loader.py v3 untuk generate ulang.")
    else:
        all_tags = set()
        for rec in train_records + test_records:
            all_tags.update(rec.get("ner_tags", []))
        label_list = ["O"] + sorted([t for t in all_tags if t != "O"])
        label2id   = {l: i for i, l in enumerate(label_list)}
        id2label   = {i: l for l, i in label2id.items()}
        print(f"[INFO] Label list dibangun dari data: {len(label_list)} label")

    print(f"[INFO] Jumlah label : {len(label_list)}")
    return train_records, test_records, label_list, label2id, id2label, entity_counts


# ─────────────────────────────────────────────────────────────────────────────
# 5. TOKENISASI + LABEL ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_and_align_labels(examples: dict, tokenizer, label2id: dict, max_length: int = 512):
    """
    Tokenisasi wordpiece + alignment label BIO.

    Subword pertama dari tiap kata → label asli kata tersebut
    Subword lanjutan               → -100 (diabaikan dalam loss)
    Special tokens [CLS][SEP][PAD] → -100
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
        padding=False,
    )

    all_labels = []
    for i, raw_labels in enumerate(examples["ner_tags"]):
        word_ids     = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids    = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                tag = raw_labels[word_id] if word_id < len(raw_labels) else "O"
                label_ids.append(label2id.get(tag, label2id["O"]))
            else:
                label_ids.append(-100)
            prev_word_id = word_id

        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


# ─────────────────────────────────────────────────────────────────────────────
# 6. METRIK EVALUASI
# ─────────────────────────────────────────────────────────────────────────────

seqeval = evaluate.load("seqeval")

def make_compute_metrics(id2label: dict):
    """Factory untuk compute_metrics yang capture id2label."""

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions    = np.argmax(logits, axis=-1)

        true_labels, true_preds = [], []
        for pred_seq, label_seq in zip(predictions, labels):
            row_label, row_pred = [], []
            for pred, label in zip(pred_seq, label_seq):
                if label != -100:
                    row_label.append(id2label[int(label)])
                    row_pred.append(id2label[int(pred)])
            true_labels.append(row_label)
            true_preds.append(row_pred)

        results = seqeval.compute(
            predictions=true_preds,
            references=true_labels,
            scheme="IOB2",
            mode="strict",
        )

        print("\n📊 Detail per entity:")
        for entity, scores in sorted(results.items()):
            if isinstance(scores, dict):
                f1 = scores["f1"]
                flag = "⚠️ " if f1 < 0.5 else "✅ " if f1 >= 0.7 else "   "
                print(f"  {flag}{entity:20s} → "
                      f"P: {scores['precision']:.3f}  "
                      f"R: {scores['recall']:.3f}  "
                      f"F1: {f1:.3f}")

        return {
            "eval_precision": results["overall_precision"],
            "eval_recall":    results["overall_recall"],
            "eval_f1":        results["overall_f1"],
            "eval_accuracy":  results["overall_accuracy"],
        }

    return compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 7. FUNGSI FINE-TUNING UTAMA
# ─────────────────────────────────────────────────────────────────────────────

def finetune(
    train_path: str         = "dataset/processed/train_dataset.json",
    test_path: str          = "dataset/processed/test_dataset.json",
    label_config_path: str  = "dataset/processed/label_config.json",
    output_dir: str         = MODEL_CONFIG["output_dir"],
    base_model: str         = MODEL_CONFIG["base_model"],
    max_length: int         = MODEL_CONFIG["max_length"],
    seed: int               = 42,
    use_class_weights: bool = True,
    o_weight: float         = 0.1,
    max_weight: float       = 50.0,
    skill_boost: float      = 5.0,    # ✨ BARU v3
    ye_boost: float         = 10.0,   # ✨ BARU v3
):
    """
    Fine-tuning BERT-NER end-to-end dengan class weighting.

    Args:
        train_path         : Path train_dataset.json
        test_path          : Path test_dataset.json
        label_config_path  : Path label_config.json (harus dari dataset_loader v3)
        output_dir         : Direktori penyimpanan model
        base_model         : Nama model BERT di HuggingFace
        max_length         : Panjang maksimum token
        seed               : Random seed
        use_class_weights  : Aktifkan WeightedTrainer (True by default)
        o_weight           : Bobot untuk token "O"
        max_weight         : Cap maksimum bobot kelas minoritas
        skill_boost        : ✨ BARU: faktor boost SKILLS (default 5.0)
                             Naikkan jika SKILLS masih under-detected
        ye_boost           : ✨ BARU: faktor boost YEARS_EXPERIENCE (default 10.0)
                             Naikkan jika YE masih F1=0
    """
    print(f"\n{'='*60}")
    print("  BERT-NER FINE-TUNING  (v3 — SKILLS & YE Fix)")
    print(f"  Model           : {base_model}")
    print(f"  Output          : {output_dir}")
    print(f"  Class Weighting : {'✅ AKTIF' if use_class_weights else '❌ NONAKTIF'}")
    if use_class_weights:
        print(f"  skill_boost     : {skill_boost}x  (SKILLS polisemi fix)")
        print(f"  ye_boost        : {ye_boost}x  (YEARS_EXPERIENCE rarity fix)")
    print(f"  Best metric     : {TRAIN_CONFIG['metric_for_best_model']}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load dataset
    print("[STEP 1] Memuat dataset...")
    (train_records, test_records,
     label_list, label2id, id2label,
     entity_counts) = load_processed_dataset(train_path, test_path, label_config_path)

    # ── Step 2: HuggingFace Dataset
    print("\n[STEP 2] Konversi ke HuggingFace Dataset...")
    train_hf = HFDataset.from_list(train_records)
    test_hf  = HFDataset.from_list(test_records)

    # ── Step 3: Tokenizer
    print(f"\n[STEP 3] Memuat tokenizer: {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # ── Step 4: Tokenisasi + label alignment
    print("[STEP 4] Tokenisasi dan alignment label...")
    tokenize_fn = lambda examples: tokenize_and_align_labels(
        examples, tokenizer, label2id, max_length
    )
    train_tok = train_hf.map(tokenize_fn, batched=True, remove_columns=train_hf.column_names)
    test_tok  = test_hf.map(tokenize_fn,  batched=True, remove_columns=test_hf.column_names)
    print(f"  Kolom setelah tokenisasi: {train_tok.column_names}")

    # ── Step 5a: Class weights
    class_weights = None
    if use_class_weights:
        print("\n[STEP 5a] Menghitung class weights (v3 — dengan SKILLS & YE boost)...")
        class_weights = compute_class_weights(
            label_list, entity_counts,
            o_weight=o_weight,
            max_weight=max_weight,
            skill_boost=skill_boost,   # ✨ BARU v3
            ye_boost=ye_boost,         # ✨ BARU v3
        )
        log_class_weights(label_list, class_weights)
    else:
        print("\n[STEP 5a] Class weighting dinonaktifkan")

    # ── Step 5b: Model
    print(f"\n[STEP 5b] Memuat model: {base_model}...")
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if TRAIN_CONFIG["fp16"] else None,
    )

    # ── Step 6: TrainingArguments
    print("\n[STEP 6] Konfigurasi training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=TRAIN_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAIN_CONFIG["per_device_eval_batch_size"],
        learning_rate=TRAIN_CONFIG["learning_rate"],
        warmup_ratio=TRAIN_CONFIG["warmup_ratio"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        evaluation_strategy=TRAIN_CONFIG["evaluation_strategy"],
        save_strategy=TRAIN_CONFIG["save_strategy"],
        load_best_model_at_end=TRAIN_CONFIG["load_best_model_at_end"],
        metric_for_best_model=TRAIN_CONFIG["metric_for_best_model"],
        greater_is_better=TRAIN_CONFIG["greater_is_better"],
        logging_steps=TRAIN_CONFIG["logging_steps"],
        fp16=TRAIN_CONFIG["fp16"],
        report_to=TRAIN_CONFIG["report_to"],
        seed=seed,
        save_total_limit=2,
    )

    # ── Step 7: Trainer
    TrainerClass = WeightedTrainer if use_class_weights else Trainer

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(id2label),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=TRAIN_CONFIG["early_stopping_patience"]
        )],
    )

    if use_class_weights:
        trainer = WeightedTrainer(class_weights=class_weights, **trainer_kwargs)
        print(f"\n[STEP 7] Menggunakan WeightedTrainer ✅ (skill_boost={skill_boost}, ye_boost={ye_boost})")
    else:
        trainer = Trainer(**trainer_kwargs)
        print(f"\n[STEP 7] Menggunakan Trainer standar")

    # ── Step 8: Training
    print(f"\n[STEP 8] Mulai training (max {TRAIN_CONFIG['num_train_epochs']} epoch, "
          f"early stopping patience={TRAIN_CONFIG['early_stopping_patience']})...")
    trainer.train()

    # ── Step 9: Evaluasi akhir
    print("\n[STEP 9] Evaluasi final pada test set...")
    eval_results = trainer.evaluate()

    print("\n  📊 HASIL EVALUASI FINAL:")
    print(f"  {'Metric':30s}  {'Value':>8s}")
    print(f"  {'-'*42}")
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"  {k:30s}  {v:8.4f}")
        else:
            print(f"  {k:30s}  {v}")

    # ── Step 10: Simpan model terbaik
    best_dir = os.path.join(output_dir, "best_model")
    print(f"\n[STEP 10] Menyimpan model ke: {best_dir}")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    label_cfg = {
        "label_list":    label_list,
        "label2id":      label2id,
        "id2label":      id2label,
        "entity_types":  list({l[2:] for l in label_list if l != "O"}),
        "entity_counts": entity_counts,
        "base_model":    base_model,
        "max_length":    max_length,
        # ✨ Simpan juga parameter boost untuk reproduksi
        "skill_boost":   skill_boost,
        "ye_boost":      ye_boost,
    }
    with open(os.path.join(best_dir, "label_config.json"), "w") as f:
        json.dump(label_cfg, f, ensure_ascii=False, indent=2)

    if class_weights is not None:
        weights_info = {label: round(w, 4) for label, w in zip(label_list, class_weights.tolist())}
        with open(os.path.join(best_dir, "class_weights.json"), "w") as f:
            json.dump(weights_info, f, ensure_ascii=False, indent=2)

    with open(os.path.join(best_dir, "eval_results.json"), "w") as f:
        json.dump(
            {k: float(v) if isinstance(v, float) else v for k, v in eval_results.items()},
            f, indent=2,
        )

    print(f"\n{'='*60}")
    print(f"  ✅ Model tersimpan di : {best_dir}")
    print(f"  F1        : {eval_results.get('eval_f1', 0):.4f}")
    print(f"  Precision : {eval_results.get('eval_precision', 0):.4f}")
    print(f"  Recall    : {eval_results.get('eval_recall', 0):.4f}")
    print(f"  Accuracy  : {eval_results.get('eval_accuracy', 0):.4f}")
    print(f"{'='*60}\n")

    return best_dir


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tuning BERT-NER untuk CV Rekrutmen (v3 — SKILLS & YE Fix)"
    )
    parser.add_argument("--train",              default="dataset/processed/train_dataset.json")
    parser.add_argument("--test",               default="dataset/processed/test_dataset.json")
    parser.add_argument("--label_config",       default="dataset/processed/label_config.json")
    parser.add_argument("--output_dir",         default="./bert_ner_cv_model")
    parser.add_argument("--base_model",         default="bert-base-uncased")
    parser.add_argument("--max_length",         type=int,   default=512)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--no_class_weights",   action="store_true",
                        help="Nonaktifkan class weighting (pakai Trainer standar)")
    parser.add_argument("--o_weight",           type=float, default=0.1,
                        help="Bobot token O (default: 0.1)")
    parser.add_argument("--max_weight",         type=float, default=50.0,
                        help="Cap maksimum bobot kelas minoritas (default: 50.0)")
    # ✨ BARU v3: parameter boost
    parser.add_argument("--skill_boost",        type=float, default=5.0,
                        help="Faktor boost SKILLS karena polisemi (default: 5.0). "
                             "Naikkan ke 8-10 jika SKILLS masih F1 < 0.3")
    parser.add_argument("--ye_boost",           type=float, default=10.0,
                        help="Faktor boost YEARS_EXPERIENCE karena data sangat langka (default: 10.0). "
                             "Naikkan ke 15-20 jika YE masih F1=0")
    args = parser.parse_args()

    finetune(
        train_path=args.train,
        test_path=args.test,
        label_config_path=args.label_config,
        output_dir=args.output_dir,
        base_model=args.base_model,
        max_length=args.max_length,
        seed=args.seed,
        use_class_weights=not args.no_class_weights,
        o_weight=args.o_weight,
        max_weight=args.max_weight,
        skill_boost=args.skill_boost,   # ✨ BARU v3
        ye_boost=args.ye_boost,         # ✨ BARU v3
    )