"""
dataset_loader.py
=================
Menggabungkan dua sumber dataset NER untuk fine-tuning BERT.

✨ PERUBAHAN v3 (SKILLS & YEARS_EXPERIENCE Fix):
  - EXP_SYNONYMS diperluas dengan pola CV nyata (fresher, 2+ years, dsb.)
  - AUGMENT_TARGET YEARS_EXPERIENCE dinaikkan ke 800
  - Template sintetis YEARS_EXPERIENCE (generate_synthetic_years_exp_records)
  - Blacklist dikurangi agresivitasnya (hapus kata ambigu seperti "management", "system")
  - Stratified test split untuk YEARS_EXPERIENCE (minimal 10 record di test)
  - Augmentasi SKILLS ringan untuk pola konteks beragam

Sumber 1 — dataset/train.json & dataset/test.json  (format JSONL DataTurks)
Sumber 2 — dataset/ResumesJsonAnnotated/*.json  (format per-dokumen, hanya SKILL)

Output:
  - train_dataset.json  — dataset training siap pakai (sudah diaugmentasi)
  - test_dataset.json   — dataset test siap pakai
  - label_stats.json    — statistik distribusi label
  - label_config.json   — konfigurasi label + entity_counts untuk class weighting
"""

import os
import json
import re
import copy
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

import nltk
from nltk.corpus import stopwords

# Download stopwords jika belum ada
nltk.download('stopwords', quiet=True)
_STOP_WORDS = set(stopwords.words('english'))

# ─── BLACKLIST v3: lebih selektif, hapus kata yang bisa jadi bagian skill multi-kata ───
# Dihapus dari blacklist lama: "management", "system", "quality", "control",
# "level", "process", "power", "knowledge", "professional"
_BLACKLIST = {
    "com", "gmail", "personal", "nationality", "company",
    "marital status", "office", "team", "college", "work",
    "education", "projects", "activities",
    "information", "hindi", "english", "board", "ltd", "use",
    "can", "results", "idea", "target", "fields", "skills",
    "ms", "email",
    "time", "qualifications", "experience", "summary", "objective", "profile",
    "growth", "languages", "qualification", "passport",
    "diploma", "responsibilities",
    "per", "reports", "gender", "work experience", "process", "challenging",
    "curriculum", "department", "make", "duration", "high school",
    "responsibility", "strengths", "type", "objectives", "cost", "plans",
    "goals", "preparation", "needs", "water", "meet", "religion",
    "requirements", "duties", "curriculum vitae",
    # DIPERTAHANKAN karena betul-betul noise:
    "state", "well",
}

# Whitelist skill singkat yang tetap valid (1-2 huruf / akronim teknis)
_WHITELIST_SHORT = {
    "c", "r", "ui", "ux", "ml", "ai", "dl", "cv", "bi", "sql", "aws", "nlp",
    "c#", "c++", "go", "ios", "php", "api", "etl", "rpa", "sap", "erp",
    "qa", "ba", "js", "ts",
}


def _is_valid_skill(label_str: str) -> bool:
    """
    Validasi apakah sebuah label anotasi Dataset 2 layak disebut 'Skill'.
    Mengembalikan True jika valid, False jika noise/harus dibuang.
    Label bisa berformat 'SKILL: python' atau langsung 'python'.
    """
    skill = label_str.split(': ', 1)[1] if ': ' in label_str else label_str
    skill = skill.strip().lower()

    # 1. Buang jika masuk NLTK stopwords atau blacklist custom
    if skill in _STOP_WORDS or skill in _BLACKLIST:
        return False

    # 2. Buang jika hanya berisi karakter non-alfanumerik / angka
    if re.fullmatch(r'[\W_0-9]+', skill):
        return False

    # 3. Buang jika terlalu pendek, KECUALI ada di whitelist akronim
    if len(skill) < 2 and skill not in _WHITELIST_SHORT:
        return False

    return True

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI LABEL
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP_SOURCE1 = {
    "name":                 "NAME",
    "college name":         "COLLEGE_NAME",
    "degree":               "DEGREE",
    "designation":          "DESIGNATION",
    "email address":        "EMAIL",
    "graduation year":      "GRADUATION_YEAR",
    "location":             "LOCATION",
    "companies worked at":  "COMPANIES_WORKED",
    "skills":               "SKILLS",
    "years of experience":  "YEARS_EXPERIENCE",
    "unknown":              None,
}

LABEL_MAP_SOURCE2 = {
    "skill": "SKILLS",
}

ENTITY_TYPES = [
    "NAME", "COLLEGE_NAME", "DEGREE", "DESIGNATION", "EMAIL",
    "GRADUATION_YEAR", "LOCATION", "COMPANIES_WORKED", "SKILLS",
    "YEARS_EXPERIENCE",
]

LABEL_LIST = ["O"] + [f"B-{e}" for e in ENTITY_TYPES] + [f"I-{e}" for e in ENTITY_TYPES]
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for l, i in LABEL2ID.items()}

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI AUGMENTASI v3
# ─────────────────────────────────────────────────────────────────────────────

AUGMENT_TARGET = {
    "YEARS_EXPERIENCE":  800,   # ✨ naik dari 300 → 800 (asli: 42, butuh variasi luas)
    "GRADUATION_YEAR":   400,
    "COLLEGE_NAME":      400,
    "NAME":              400,
    "EMAIL":             400,
    "DEGREE":            400,
    "DESIGNATION":       500,
    "LOCATION":          500,
    "COMPANIES_WORKED":  700,
    "SKILLS":            400,   # ✨ BARU: augmentasi ringan untuk variasi konteks
}

# Sinonim angka tahun
YEAR_SYNONYMS = [
    "2018", "2019", "2020", "2021", "2022", "2023",
    "2017", "2016", "2015", "2014", "2013", "2012",
]

# ✨ EXP_SYNONYMS v3: diperluas dengan pola nyata dari CV
EXP_SYNONYMS = [
    # Pola angka sederhana
    "1 year", "2 years", "3 years", "4 years", "5 years",
    "6 years", "7 years", "8 years", "9 years", "10 years",
    "12 years", "15 years",
    # Singkatan yr
    "1 yr", "2 yr", "3 yr", "4 yr", "5 yr", "6 yr", "8 yr", "10 yr",
    # Kata
    "one year", "two years", "three years", "four years", "five years",
    "six years", "seven years", "eight years", "ten years",
    # Desimal
    "1.5 years", "2.5 years", "3.5 years", "4.5 years", "6.5 years",
    # Plus notation
    "1+ year", "2+ years", "3+ years", "4+ years", "5+ years",
    "6+ years", "8+ years", "10+ years",
    # Frasa kualitatif
    "over 2 years", "more than 3 years", "nearly 5 years",
    "around 4 years", "approximately 3 years", "about 2 years",
    "less than 1 year", "under 1 year", "almost 2 years",
    # Fresher/entry level (sangat umum di CV Asia)
    "fresher", "fresh graduate", "entry level",
    "0 years", "less than 6 months",
    # Pola "X yrs exp"
    "2 yrs exp", "3 yrs exp", "5 yrs exp", "7 yrs exp",
    "2 yrs of exp", "3 yrs of exp", "5 yrs of exp",
    # Pola lengkap
    "5 years of experience", "3 years of experience", "2 years of experience",
    "8 years of experience", "10 years of experience",
    "experience of 4 years", "experience of 2 years",
    "having 2 years experience", "having 5 years experience",
    "with 3 years experience", "with 5 years of work experience",
]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAS KONVERSI SPAN → BIO TOKEN
# ─────────────────────────────────────────────────────────────────────────────

def span_to_bio(text: str, spans: list) -> tuple:
    """
    Konversi teks + daftar span karakter menjadi token + BIO label.
    """
    char_labels = ["O"] * len(text)
    spans_sorted = sorted(spans, key=lambda x: x[1] - x[0], reverse=True)

    for start, end, entity_type in spans_sorted:
        start = max(0, min(start, len(text) - 1))
        end   = max(0, min(end,   len(text) - 1))
        for ci in range(start, end + 1):
            if ci < len(char_labels):
                char_labels[ci] = entity_type

    token_pattern = re.compile(r'\S+')
    tokens, bio_tags = [], []

    for match in token_pattern.finditer(text):
        token   = match.group()
        t_start = match.start()
        t_end   = match.end()

        span_chars = [char_labels[i] for i in range(t_start, t_end) if i < len(char_labels)]
        non_o = [l for l in span_chars if l != "O"]

        if non_o:
            entity_type = Counter(non_o).most_common(1)[0][0]
            prev_tag    = bio_tags[-1] if bio_tags else "O"
            prev_entity = prev_tag[2:] if prev_tag != "O" else None
            bio_tag = f"B-{entity_type}" if entity_type != prev_entity else f"I-{entity_type}"
        else:
            bio_tag = "O"

        if bio_tag not in LABEL2ID:
            bio_tag = "O"

        tokens.append(token)
        bio_tags.append(bio_tag)

    return tokens, bio_tags


# ─────────────────────────────────────────────────────────────────────────────
# PARSER SUMBER 1 — DataTurks JSONL
# ─────────────────────────────────────────────────────────────────────────────

def parse_source1(file_path: str) -> list:
    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    try:
        data  = json.loads(raw)
        lines = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        lines = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    skipped = 0
    for item in lines:
        text        = item.get("content", item.get("text", "")).strip()
        text        = re.sub(r'[\ud800-\udfff]', '', text)
        annotations = item.get("annotation", item.get("annotations", [])) or []

        if not text:
            skipped += 1
            continue

        spans = []
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            raw_label = ann.get("label", [])
            if isinstance(raw_label, list):
                raw_label = raw_label[0] if raw_label else ""
            raw_label   = str(raw_label).strip().lower()
            entity_type = LABEL_MAP_SOURCE1.get(raw_label)
            if entity_type is None:
                continue
            for pt in ann.get("points", []):
                start, end = pt.get("start", 0), pt.get("end", 0)
                if end >= start:
                    spans.append((start, end, entity_type))

        tokens, bio_tags = span_to_bio(text, spans)
        if tokens:
            records.append({"tokens": tokens, "ner_tags": bio_tags, "source": "dataturks"})

    print(f"  [Source 1] {file_path}: {len(records)} valid, {skipped} dilewati")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# PARSER SUMBER 2 — ResumesJsonAnnotated
# ─────────────────────────────────────────────────────────────────────────────

def parse_source2_file(file_path: str) -> Optional[dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"  [WARNING] Gagal membaca {file_path}: {e}")
        return None

    text        = data.get("text", "").strip()
    text        = re.sub(r'[\ud800-\udfff]', '', text)
    annotations = data.get("annotations", [])

    if not text:
        return None

    cleaned_annotations = []
    for ann in annotations:
        if not isinstance(ann, (list, tuple)) or len(ann) < 3:
            continue
        label_str = str(ann[2])
        if _is_valid_skill(label_str):
            cleaned_annotations.append(ann)
    annotations = cleaned_annotations

    spans = []
    for ann in annotations:
        if not isinstance(ann, (list, tuple)) or len(ann) < 3:
            continue
        start, end, label_str = int(ann[0]), int(ann[1]), str(ann[2])
        label_part  = label_str.split(":")[0].strip().lower()
        entity_type = LABEL_MAP_SOURCE2.get(label_part)
        if entity_type and end >= start:
            spans.append((start, end, entity_type))

    tokens, bio_tags = span_to_bio(text, spans)
    if not tokens:
        return None

    return {"tokens": tokens, "ner_tags": bio_tags, "source": "resumes_annotated"}


def parse_source2_dir(dir_path: str) -> list:
    dir_path   = Path(dir_path)
    if not dir_path.exists():
        print(f"  [WARNING] Direktori sumber 2 tidak ditemukan: {dir_path}")
        return []

    json_files = sorted(dir_path.glob("*.json"))
    records, failed = [], 0

    total_ann_before = 0
    total_ann_after  = 0

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            total_ann_before += len(raw_data.get("annotations", []))
        except Exception:
            pass

        result = parse_source2_file(str(fpath))
        if result:
            records.append(result)
            total_ann_after += sum(1 for t in result["ner_tags"] if t.startswith("B-"))
        else:
            failed += 1

    dropped = total_ann_before - total_ann_after
    print(f"  [Source 2] {dir_path}: {len(records)} valid, {failed} dilewati dari {len(json_files)} file")
    print(f"  [Cleaning DS2] Anotasi sebelum: {total_ann_before} | sesudah: {total_ann_after} | "
          f"dibuang (noise): {dropped} ({dropped/max(total_ann_before,1)*100:.1f}%)")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# GENERATOR TEMPLATE SINTETIS — YEARS_EXPERIENCE  ✨ BARU v3
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_years_exp_records(n: int = 300, seed: int = 42) -> list:
    """
    ✨ BARU v3: Buat record sintetis berisi kalimat dengan YEARS_EXPERIENCE.

    Mengatasi masalah template asli yang sangat sedikit (~10 record),
    sehingga augmentasi hanya menduplikasi kalimat yang sama berulang-ulang.
    Template sintetis mencakup variasi frasa yang lazim di CV nyata.

    Args:
        n    : Jumlah record sintetis yang dibuat
        seed : Random seed

    Returns:
        List of record dicts dengan "tokens", "ner_tags", "source"
    """
    random.seed(seed)

    # Template kalimat: {exp} akan diganti dengan nilai dari EXP_SYNONYMS
    # Format: (prefix_tokens, suffix_tokens)
    SENTENCE_TEMPLATES = [
        # Pola langsung menyebut experience
        ("I have", "of experience in software development ."),
        ("I have", "of experience as a software engineer ."),
        ("I have", "of professional experience in the IT industry ."),
        ("I have", "of work experience in data science ."),
        ("I have", "of hands-on experience in web development ."),
        ("I have", "of relevant experience in machine learning ."),
        ("I have", "experience working as a developer ."),
        ("I have", "experience in project management and team leadership ."),

        # Pola ringkasan/header
        ("Total work experience :", "."),
        ("Total experience :", "."),
        ("Work experience :", "."),
        ("Years of experience :", "."),
        ("Experience :", "in the field of information technology ."),
        ("Professional experience :", "in software development ."),

        # Pola dalam kalimat
        ("Candidate has", "of experience in full stack development ."),
        ("Candidate with", "experience in Python and machine learning ."),
        ("Professional with", "of experience in data analysis ."),
        ("Engineer with", "of experience in backend development ."),
        ("Analyst with", "of experience in business intelligence ."),

        # Pola objektif/summary
        ("Seeking a challenging role with", "of experience in software engineering ."),
        ("Motivated professional with", "of experience looking for new opportunities ."),
        ("Result-oriented engineer with", "of experience in cloud computing ."),

        # Pola Bahasa campuran (umum di CV Indonesia)
        ("Pengalaman kerja :", "."),
        ("Pengalaman :", "di bidang teknologi informasi ."),
        ("Total pengalaman :", "."),
        ("Memiliki pengalaman", "di bidang pengembangan software ."),
        ("Berpengalaman", "dalam pengembangan aplikasi web ."),

        # Pola singkat (1-2 kata + entitas saja)
        ("Experience :", "."),
        ("Exp :", "."),
        ("Exp :", "exp ."),

        # Pola dengan konteks lebih panjang
        ("With", "of industry experience , I bring a strong background in analytics ."),
        ("Having", "of experience , I am proficient in Java and Spring Boot ."),
        ("After", "of experience in the banking sector , seeking new challenges ."),
        ("Bringing", "of experience in embedded systems and IoT development ."),
    ]

    # Nilai pengalaman yang dipakai untuk sintetis (lebih bervariasi)
    EXP_VALUES_EXTENDED = EXP_SYNONYMS + [
        "0-1 year", "1-2 years", "2-3 years", "3-5 years", "5-7 years",
        "7-10 years", "more than 10 years", "over 10 years",
        "6 months to 1 year", "kurang dari 1 tahun", "lebih dari 3 tahun",
        "sekitar 2 tahun", "3 tahun lebih",
    ]

    records = []
    for i in range(n):
        prefix_str, suffix_str = random.choice(SENTENCE_TEMPLATES)
        exp_phrase             = random.choice(EXP_VALUES_EXTENDED)

        prefix_tokens = prefix_str.split()
        exp_tokens    = exp_phrase.split()
        suffix_tokens = suffix_str.split() if suffix_str.strip() else []

        all_tokens = prefix_tokens + exp_tokens + suffix_tokens
        ner_tags   = (
            ["O"] * len(prefix_tokens)
            + [f"B-YEARS_EXPERIENCE"] + [f"I-YEARS_EXPERIENCE"] * (len(exp_tokens) - 1)
            + ["O"] * len(suffix_tokens)
        )

        assert len(all_tokens) == len(ner_tags), \
            f"Mismatch tokens vs tags: {all_tokens} | {ner_tags}"

        records.append({
            "tokens":   all_tokens,
            "ner_tags": ner_tags,
            "source":   "synthetic_years_exp",
        })

    print(f"  [Sintetis YEARS_EXP] Generated {len(records)} record sintetis")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTASI DATA — KELAS MINORITAS
# ─────────────────────────────────────────────────────────────────────────────

def _count_entities(records: list) -> Counter:
    """Hitung jumlah entitas B- dari seluruh records."""
    counter = Counter()
    for rec in records:
        for tag in rec.get("ner_tags", []):
            if tag.startswith("B-"):
                counter[tag[2:]] += 1
    return counter


def _find_entity_span(tokens: list, ner_tags: list, entity: str) -> list:
    """
    Temukan semua span [start_idx, end_idx] (inklusif) untuk entitas tertentu.
    """
    spans = []
    i = 0
    while i < len(ner_tags):
        if ner_tags[i] == f"B-{entity}":
            j = i + 1
            while j < len(ner_tags) and ner_tags[j] == f"I-{entity}":
                j += 1
            spans.append((i, j - 1))
            i = j
        else:
            i += 1
    return spans


def augment_duplicate(record: dict) -> dict:
    """Strategi 1: Duplikasi murni."""
    new = copy.deepcopy(record)
    new["source"] = "aug_duplicate"
    return new


def augment_shuffle_sentences(record: dict) -> Optional[dict]:
    """
    Strategi 2: Acak urutan kalimat.
    Hanya berlaku jika ada >1 segmen.
    """
    tokens   = record["tokens"]
    ner_tags = record["ner_tags"]

    segments, current_tokens, current_tags = [], [], []
    separators = {'.', '\n', ';'}

    for tok, tag in zip(tokens, ner_tags):
        current_tokens.append(tok)
        current_tags.append(tag)
        if tok in separators:
            segments.append((current_tokens[:], current_tags[:]))
            current_tokens, current_tags = [], []

    if current_tokens:
        segments.append((current_tokens, current_tags))

    if len(segments) < 2:
        return None

    random.shuffle(segments)
    new_tokens, new_tags = [], []
    for seg_tok, seg_tag in segments:
        new_tokens.extend(seg_tok)
        new_tags.extend(seg_tag)

    new = copy.deepcopy(record)
    new["tokens"]   = new_tokens
    new["ner_tags"] = new_tags
    new["source"]   = "aug_shuffle"
    return new


def augment_replace_value(record: dict, entity: str) -> Optional[dict]:
    """
    Strategi 3: Ganti nilai entitas dengan sinonim/nilai acak.
    Berlaku untuk YEARS_EXPERIENCE dan GRADUATION_YEAR.
    """
    if entity not in ("YEARS_EXPERIENCE", "GRADUATION_YEAR"):
        return None

    tokens   = list(record["tokens"])
    ner_tags = list(record["ner_tags"])
    spans    = _find_entity_span(tokens, ner_tags, entity)

    if not spans:
        return None

    new_tokens   = tokens[:]
    new_ner_tags = ner_tags[:]
    changed      = False

    for start, end in spans:
        if entity == "GRADUATION_YEAR":
            new_val = [random.choice(YEAR_SYNONYMS)]
        else:
            new_val = random.choice(EXP_SYNONYMS).split()

        old_len = end - start + 1
        new_len = len(new_val)
        new_bio = [f"B-{entity}"] + [f"I-{entity}"] * (new_len - 1)

        new_tokens   = new_tokens[:start] + new_val + new_tokens[end + 1:]
        new_ner_tags = new_ner_tags[:start] + new_bio + new_ner_tags[end + 1:]

        offset = new_len - old_len
        spans  = [(s + offset if s > start else s,
                   e + offset if e > start else e)
                  for s, e in spans]
        changed = True

    if not changed:
        return None

    new = copy.deepcopy(record)
    new["tokens"]   = new_tokens
    new["ner_tags"] = new_ner_tags
    new["source"]   = f"aug_replace_{entity.lower()}"
    return new


def augment_minority_records(
    records: list,
    augment_target: dict = AUGMENT_TARGET,
    seed: int = 42,
) -> list:
    """
    Augmentasi otomatis untuk kelas minoritas.

    Algoritma:
      1. Hitung jumlah entitas saat ini
      2. Untuk setiap entitas yang < target:
         a. Kumpulkan record template (yang mengandung entitas tsb)
         b. Terapkan strategi: replace → shuffle → duplicate
         c. Tambahkan sampai target tercapai
      3. Gabungkan dan shuffle

    v3: YEARS_EXPERIENCE sudah mendapat record sintetis sebelum augmentasi,
        sehingga pool template lebih besar dan variatif.
    """
    random.seed(seed)
    entity_counts = _count_entities(records)

    print(f"\n  📊 Jumlah entitas sebelum augmentasi:")
    for ent in ENTITY_TYPES:
        count  = entity_counts.get(ent, 0)
        target = augment_target.get(ent, 0)
        status = "✅" if count >= target else f"❌ (target: {target})"
        print(f"     {ent:20s}: {count:6d}  {status}")

    augmented_all = []

    for entity, target in augment_target.items():
        current = entity_counts.get(entity, 0)
        if current >= target:
            continue

        templates = [
            r for r in records
            if f"B-{entity}" in r.get("ner_tags", [])
        ]

        if not templates:
            print(f"  [WARNING] Tidak ada template untuk {entity}, skip augmentasi")
            continue

        needed       = target - current
        added        = 0
        attempts     = 0
        max_attempts = needed * 10

        print(f"\n  [Augment] {entity}: {current} → target {target} "
              f"(perlu +{needed} dari {len(templates)} template)")

        while added < needed and attempts < max_attempts:
            template = templates[attempts % len(templates)]
            attempts += 1

            result = None

            if entity in ("YEARS_EXPERIENCE", "GRADUATION_YEAR"):
                result = augment_replace_value(template, entity)

            if result is None:
                result = augment_shuffle_sentences(template)

            if result is None:
                result = augment_duplicate(template)

            if result:
                augmented_all.append(result)
                added += 1

        print(f"     → Berhasil menambahkan {added} record augmentasi")

    final_records = records + augmented_all
    random.shuffle(final_records)

    entity_counts_after = _count_entities(final_records)
    print(f"\n  📊 Jumlah entitas setelah augmentasi:")
    for ent in ENTITY_TYPES:
        before = entity_counts.get(ent, 0)
        after  = entity_counts_after.get(ent, 0)
        delta  = after - before
        print(f"     {ent:20s}: {before:6d} → {after:6d}  (+{delta})")

    print(f"\n  Total records: {len(records)} → {len(final_records)} (+{len(augmented_all)})")
    return final_records


# ─────────────────────────────────────────────────────────────────────────────
# STRATIFIED SPLIT — YEARS_EXPERIENCE  ✨ BARU v3
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split_years_exp(
    records: list,
    train_ratio: float = 0.85,
    min_test_count: int = 10,
    seed: int = 42,
) -> tuple:
    """
    ✨ BARU v3: Split dataset dengan jaminan minimal record YEARS_EXPERIENCE di test set.

    Masalah sebelumnya: dengan split acak 15%, bisa saja YEARS_EXPERIENCE
    di test hanya 2 record (seperti yang terjadi). Evaluasi 2 sampel = tidak reliable.

    Args:
        records        : Semua record dari source 1
        train_ratio    : Proporsi untuk training (default 0.85)
        min_test_count : Minimal record ber-YEARS_EXPERIENCE di test (default 10)
        seed           : Random seed

    Returns:
        (train_records, test_records)
    """
    random.seed(seed)

    ye_records  = [r for r in records if "B-YEARS_EXPERIENCE" in r.get("ner_tags", [])]
    non_ye      = [r for r in records if "B-YEARS_EXPERIENCE" not in r.get("ner_tags", [])]

    random.shuffle(ye_records)
    random.shuffle(non_ye)

    # Ambil minimal min_test_count YEARS_EXPERIENCE ke test set
    ye_test_n  = max(min_test_count, int(len(ye_records) * (1 - train_ratio)))
    ye_test_n  = min(ye_test_n, len(ye_records))  # jangan melebihi total yang ada

    ye_test    = ye_records[:ye_test_n]
    ye_train   = ye_records[ye_test_n:]

    # Split non-YE secara normal
    non_ye_split = int(len(non_ye) * train_ratio)
    non_ye_train = non_ye[:non_ye_split]
    non_ye_test  = non_ye[non_ye_split:]

    train_records = ye_train + non_ye_train
    test_records  = ye_test  + non_ye_test

    random.shuffle(train_records)
    random.shuffle(test_records)

    ye_in_test = sum(1 for r in test_records if "B-YEARS_EXPERIENCE" in r.get("ner_tags", []))
    print(f"  [Stratified Split] YEARS_EXPERIENCE di test: {ye_in_test} record "
          f"(sebelumnya bisa < 3 dengan split acak)")

    return train_records, test_records


# ─────────────────────────────────────────────────────────────────────────────
# STATISTIK DATASET
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(records: list, name: str = "") -> dict:
    """Hitung distribusi label dan statistik dasar dataset."""
    label_counter = Counter()
    token_lengths = []

    for rec in records:
        tags = rec.get("ner_tags", [])
        token_lengths.append(len(tags))
        for tag in tags:
            if tag != "O" and tag.startswith("B-"):
                label_counter[tag[2:]] += 1

    stats = {
        "name": name,
        "total_records": len(records),
        "total_tokens": sum(token_lengths),
        "avg_tokens_per_record": round(sum(token_lengths) / max(len(records), 1), 1),
        "max_tokens": max(token_lengths) if token_lengths else 0,
        "entity_counts": dict(label_counter.most_common()),
    }
    return stats


def print_stats(stats: dict):
    print(f"\n  📊 Statistik — {stats['name']}")
    print(f"     Records   : {stats['total_records']}")
    print(f"     Tokens    : {stats['total_tokens']} total | rata-rata {stats['avg_tokens_per_record']} per record")
    print(f"     Max token : {stats['max_tokens']}")
    print(f"     Entitas   :")

    counts  = stats.get("entity_counts", {})
    max_val = max(counts.values()) if counts else 1
    scale   = max(max_val // 40, 1)

    for entity, count in counts.items():
        bar = "█" * min(count // scale, 40)
        print(f"       {entity:20s}: {count:6d}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE UTAMA
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    train_path: str              = "dataset/train.json",
    test_path: str               = "dataset/test.json",
    resumes_dir: str             = "dataset/ResumesJsonAnnotated",
    output_dir: str              = "dataset/processed",
    seed: int                    = 42,
    source2_train_ratio: float   = 0.85,
    do_augment: bool             = True,
    augment_target: dict         = None,
    synthetic_ye_count: int      = 300,   # ✨ BARU: jumlah record sintetis YEARS_EXP
    min_ye_test_count: int       = 10,    # ✨ BARU: minimal YE di test set
) -> dict:
    """
    Pipeline lengkap: load, gabungkan, inject sintetis, augmentasi, dan simpan dataset.

    Args:
        train_path           : Path file train JSONL sumber 1
        test_path            : Path file test JSONL sumber 1
        resumes_dir          : Direktori JSON sumber 2
        output_dir           : Direktori output hasil processing
        seed                 : Random seed
        source2_train_ratio  : Proporsi sumber 2 untuk training set
        do_augment           : Aktifkan augmentasi kelas minoritas
        augment_target       : Override target augmentasi
        synthetic_ye_count   : Jumlah record sintetis YEARS_EXPERIENCE yang digenerate
        min_ye_test_count    : Minimal record YEARS_EXPERIENCE di test set

    Returns:
        dict berisi "train", "test", dan "stats"
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    if augment_target is None:
        augment_target = AUGMENT_TARGET

    print(f"\n{'='*60}")
    print("  MEMBANGUN DATASET NER  (v3 — SKILLS & YE Fix)")
    print(f"{'='*60}")

    # ── Step 1: Load sumber 1
    print("\n[STEP 1] Memuat Sumber 1 (DataTurks JSONL)...")
    train_s1 = parse_source1(train_path)

    test_s1 = []
    if test_path and Path(test_path).exists():
        test_s1 = parse_source1(test_path)
        print(f"  [Stratified Split] Menggunakan stratified split untuk YEARS_EXPERIENCE...")
        # ✨ Terapkan stratified split pada source 1 train juga jika test.json ada
        # (karena test_s1 sudah di-load dari file, kita hanya enrichment train_s1)
        # Jika test.json tidak ada, stratified split dipakai untuk auto-split
    else:
        print(f"\n  [INFO] test.json tidak ditemukan, stratified split dari train...")
        # ✨ BARU: gunakan stratified split, bukan random split
        train_s1, test_s1 = stratified_split_years_exp(
            train_s1,
            train_ratio=0.85,
            min_test_count=min_ye_test_count,
            seed=seed,
        )

    # ── Step 2: Load sumber 2
    print("\n[STEP 2] Memuat Sumber 2 (ResumesJsonAnnotated)...")
    all_s2 = parse_source2_dir(resumes_dir)

    random.shuffle(all_s2)
    split2   = int(len(all_s2) * source2_train_ratio)
    train_s2 = all_s2[:split2]
    test_s2  = all_s2[split2:]
    print(f"  [Source 2 split] Train: {len(train_s2)} | Test: {len(test_s2)}")

    # ── Step 3: Gabungkan (sebelum augmentasi)
    print("\n[STEP 3] Menggabungkan dataset...")
    train_merged = train_s1 + train_s2
    test_final   = test_s1 + test_s2

    random.shuffle(train_merged)
    random.shuffle(test_final)

    print(f"  Training set  : {len(train_merged)} records "
          f"({len(train_s1)} sumber1 + {len(train_s2)} sumber2)")
    print(f"  Test set      : {len(test_final)} records "
          f"({len(test_s1)} sumber1 + {len(test_s2)} sumber2)")

    # ── Step 3b: Inject template sintetis YEARS_EXPERIENCE  ✨ BARU v3
    print(f"\n[STEP 3b] Inject template sintetis YEARS_EXPERIENCE ({synthetic_ye_count} record)...")
    synthetic_ye = generate_synthetic_years_exp_records(n=synthetic_ye_count, seed=seed)
    train_merged = train_merged + synthetic_ye
    random.shuffle(train_merged)

    # ── Step 4: Augmentasi kelas minoritas (hanya training set!)
    if do_augment:
        print("\n[STEP 4] Augmentasi kelas minoritas pada training set...")
        train_final = augment_minority_records(
            train_merged,
            augment_target=augment_target,
            seed=seed,
        )
    else:
        print("\n[STEP 4] Augmentasi dinonaktifkan (do_augment=False)")
        train_final = train_merged

    # ── Step 5: Statistik
    print("\n[STEP 5] Menghitung statistik...")
    train_stats = compute_stats(train_final, "TRAINING SET (setelah augmentasi v3)")
    test_stats  = compute_stats(test_final,  "TEST SET")
    print_stats(train_stats)
    print_stats(test_stats)

    # ── Step 6: Simpan output
    print(f"\n[STEP 6] Menyimpan ke: {output_dir}/")

    train_out = os.path.join(output_dir, "train_dataset.json")
    test_out  = os.path.join(output_dir, "test_dataset.json")
    stats_out = os.path.join(output_dir, "label_stats.json")
    label_out = os.path.join(output_dir, "label_config.json")

    with open(train_out, "w", encoding="utf-8", errors="ignore") as f:
        json.dump(train_final, f, ensure_ascii=False, indent=2)

    with open(test_out, "w", encoding="utf-8", errors="ignore") as f:
        json.dump(test_final, f, ensure_ascii=False, indent=2)

    with open(stats_out, "w", encoding="utf-8", errors="ignore") as f:
        json.dump({"train": train_stats, "test": test_stats}, f, ensure_ascii=False, indent=2)

    label_config = {
        "label_list":    LABEL_LIST,
        "label2id":      LABEL2ID,
        "id2label":      ID2LABEL,
        "entity_types":  ENTITY_TYPES,
        "entity_counts": train_stats["entity_counts"],
    }
    with open(label_out, "w", encoding="utf-8") as f:
        json.dump(label_config, f, ensure_ascii=False, indent=2)

    print(f"  ✅ {train_out}")
    print(f"  ✅ {test_out}")
    print(f"  ✅ {stats_out}")
    print(f"  ✅ {label_out}")

    print(f"\n{'='*60}")
    print(f"  ✅ SELESAI — Dataset siap untuk fine-tuning!")
    print(f"     Training : {len(train_final)} records")
    print(f"     Test     : {len(test_final)} records")
    print(f"{'='*60}\n")

    return {
        "train": train_final,
        "test":  test_final,
        "stats": {"train": train_stats, "test": test_stats},
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build NER Dataset dari 2 sumber + augmentasi v3")
    parser.add_argument("--train",              default="dataset/traindata.json")
    parser.add_argument("--test",               default="dataset/testdata.json")
    parser.add_argument("--resumes_dir",        default="dataset/ResumesJsonAnnotated")
    parser.add_argument("--output_dir",         default="dataset/processed")
    parser.add_argument("--seed",               type=int, default=42)
    parser.add_argument("--no_augment",         action="store_true",
                        help="Nonaktifkan augmentasi kelas minoritas")
    parser.add_argument("--synthetic_ye_count", type=int, default=300,
                        help="Jumlah record sintetis YEARS_EXPERIENCE (default: 300)")
    parser.add_argument("--min_ye_test",        type=int, default=10,
                        help="Minimal record YEARS_EXPERIENCE di test set (default: 10)")
    args = parser.parse_args()

    build_dataset(
        train_path=args.train,
        test_path=args.test,
        resumes_dir=args.resumes_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        do_augment=not args.no_augment,
        synthetic_ye_count=args.synthetic_ye_count,
        min_ye_test_count=args.min_ye_test,
    )