"""
evaluate_ner.py
===============
Evaluasi perbandingan dua pendekatan NER pada testdata.json.

PARADIGMA EVALUASI YANG BENAR:
─────────────────────────────
  Ground Truth (testdata.json) berisi:
    - label  : jenis entitas (Skills, Designation, Companies worked at, dll.)
    - text   : teks aktual entitas tersebut di dalam CV
    - start/end: posisi span di CV

  Pertanyaan evaluasi:
    "Diberikan label + teks GT sebagai 'query dari HRD',
     apakah sistem BERHASIL MENEMUKAN teks tersebut di CV?"

  Dua pendekatan yang dibandingkan:

  1. BERT NER
     → Model memprediksi entitas dari teks CV
     → Cocok jika "apa yang akan dimasukkan HRD" tidak diketahui sebelumnya
     → Paradigma: teks CV → predict entitas

  2. Hybrid Search (pendekatan app7.py yg dipakai di produk)
     → HRD memasukkan requirement (mis. "python", "sql", nama perusahaan)
     → Sistem hanya perlu memverifikasi: apakah teks itu ADA di CV?
     → Detection method: exact search + fuzzy + regex
     → Paradigma: query dari HRD → cari di teks CV → found/not found

  Kenapa Hybrid relevan:
    Di aplikasi nyata, HRD sudah tahu apa yang dicari.
    Misal skill requirement: "Python", "SQL", "TensorFlow".
    Sistem tidak perlu "menebak" — hanya perlu "mencari".
    Evaluasi: dari semua GT text, berapa % yang berhasil ditemukan di CV?

METRIC:
  Untuk setiap GT instance (label, text):
    - Hybrid: cek apakah GT text ditemukan di CV (case-insensitive, fuzzy)
    - BERT  : cek apakah prediksi BERT mencakup GT text (partial match)

  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)   ← metric utama untuk Hybrid (seberapa banyak GT ditemukan)
  F1        = harmonic mean
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

# ─────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────

TESTDATA_PATH   = "./dataset/testdata.json"
BERT_MODEL_PATH = "./bert_ner_cv_model/best_model"
OUTPUT_JSON     = "./eval_results.json"
OUTPUT_TXT      = "./eval_report.txt"

# Entitas yang multi-nilai (bisa lebih dari satu per CV)
MULTI_VALUE_LABELS  = {"Skills", "Companies worked at", "Designation",
                        "College Name", "Graduation Year", "Email Address",
                        "Location", "Degree"}
# Entitas single-value
SINGLE_VALUE_LABELS = {"Name", "Years of Experience"}


# ─────────────────────────────────────────────────────────────
# 1. LOAD TESTDATA
# ─────────────────────────────────────────────────────────────

def load_testdata(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"[INFO] Loaded {len(data)} samples dari {path}")
    return data


def extract_ground_truth(sample: dict) -> dict[str, list[str]]:
    """
    Ambil ground truth dari sample.
    Return {label: [gt_text_1, gt_text_2, ...]}
    GT text adalah teks aktual yang sudah ada di dalam CV.
    """
    gt: dict[str, list[str]] = defaultdict(list)
    for ann in sample.get("annotation", []):
        for lbl in ann.get("label", []):
            for pt in ann.get("points", []):
                txt = pt.get("text", "").strip()
                if txt:
                    gt[lbl].append(txt)
    return dict(gt)


# ─────────────────────────────────────────────────────────────
# 2. HYBRID SEARCH ENGINE
#    Paradigma: diberikan query (GT text), cari di CV
#    Ini mensimulasikan alur nyata: HRD input requirement →
#    sistem cari apakah requirement itu ada di CV pelamar
# ─────────────────────────────────────────────────────────────

def normalize_text(s: str) -> str:
    """Normalisasi untuk matching: lowercase, collapse whitespace."""
    return re.sub(r'\s+', ' ', s.strip().lower())


def hybrid_search(query: str, cv_text: str, label: str) -> tuple[bool, str]:
    """
    Cari apakah 'query' (GT text) ada di 'cv_text'.
    Return (found: bool, method: str)

    Strategi pencarian bertingkat:
      1. Exact match (case-insensitive)
      2. Normalized match (collapse whitespace)
      3. Partial/substring match (query ⊂ cv atau cv_phrase ⊂ query)
      4. Untuk Skills: per-token keyword search
      5. Untuk multi-line GT: cek baris per baris
    """
    cv_lower    = cv_text.lower()
    query_norm  = normalize_text(query)
    cv_norm     = normalize_text(cv_text)

    # ── 1. Exact (case-insensitive)
    if query.lower() in cv_lower:
        return True, "exact"

    # ── 2. Normalized whitespace match
    if query_norm in cv_norm:
        return True, "normalized"

    # ── 3. Skills: GT text sering berupa SECTION PANJANG
    #    mis. "C (Less than 1 year), Database (Less than 1 year), Java..."
    #    Kita pecah per skill keyword dan cek keberadaan masing-masing
    if label == "Skills":
        found_count, total = _search_skills_section(query, cv_text)
        if total > 0 and found_count / total >= 0.5:
            return True, f"skills_section({found_count}/{total})"

    # ── 4. Multi-line GT: cek setiap baris non-trivial
    lines = [l.strip() for l in query.split('\n') if len(l.strip()) >= 3]
    if len(lines) > 1:
        found_lines = sum(1 for l in lines if normalize_text(l) in cv_norm)
        if found_lines / len(lines) >= 0.5:
            return True, f"multiline({found_lines}/{len(lines)})"

    # ── 5. Substring yang cukup panjang (untuk partial match yang bermakna)
    #    Hindari false positive dari string sangat pendek
    if len(query_norm) >= 6:
        # Cek apakah ada "long enough" overlap
        # Sliding window: cek apakah ada 60%+ substring query di cv
        words = query_norm.split()
        if len(words) >= 3:
            # Cek ngram dari query ada di cv
            for n in range(len(words), max(1, len(words)//2), -1):
                for i in range(len(words) - n + 1):
                    ngram = " ".join(words[i:i+n])
                    if len(ngram) >= 6 and ngram in cv_norm:
                        return True, f"ngram({n}grams)"
        elif query_norm in cv_norm:
            return True, "substring"

    return False, "not_found"


def _parse_skill_keywords(skills_section: str) -> list[str]:
    """
    Parse section skills GT menjadi keyword individual.
    Contoh input:
      "C (Less than 1 year), Database (Less than 1 year), Java (Less than 1 year)"
      "• Python\n• SQL\n• TensorFlow"
    """
    # Hapus keterangan level/durasi seperti "(Less than 1 year)", "(2 years)"
    cleaned = re.sub(r'\([^)]*\)', '', skills_section)
    # Hapus bullet points
    cleaned = re.sub(r'[•\-\*]', ',', cleaned)
    # Split by koma atau newline
    raw = re.split(r'[,\n]+', cleaned)
    # Filter dan bersihkan
    keywords = []
    skip_words = {
        "technical skills", "non - technical skills", "non-technical skills",
        "soft skills", "hard skills", "skills", "additional information",
        "tools", "technologies", "programming languages", "frameworks",
        "other skills", "languages",
    }
    for kw in raw:
        kw = kw.strip().lower()
        kw = re.sub(r'\s+', ' ', kw)
        if len(kw) >= 2 and kw not in skip_words:
            keywords.append(kw)
    return keywords


def _search_skills_section(query: str, cv_text: str) -> tuple[int, int]:
    """
    Untuk GT Skills yang berupa section panjang:
    Parse jadi keyword individual, cek berapa % yang ada di CV.
    Return (found_count, total_count)
    """
    keywords = _parse_skill_keywords(query)
    if not keywords:
        return 0, 0

    cv_norm = normalize_text(cv_text)
    found = 0
    for kw in keywords:
        # Whole-word search untuk keyword pendek, substring untuk panjang
        if len(kw) <= 3:
            # Short keyword: perlu word boundary
            if re.search(r'\b' + re.escape(kw) + r'\b', cv_norm):
                found += 1
        else:
            if kw in cv_norm:
                found += 1
    return found, len(keywords)


# ─────────────────────────────────────────────────────────────
# 3. BERT NER (pembanding)
# ─────────────────────────────────────────────────────────────

def load_bert_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"[WARNING] BERT model tidak ditemukan di {model_path}")
        return None, False
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model     = AutoModelForTokenClassification.from_pretrained(model_path)
        ner_pipe  = pipeline("ner", model=model, tokenizer=tokenizer,
                             aggregation_strategy="simple", device=-1)
        print(f"[INFO] BERT NER model loaded dari {model_path}")
        return ner_pipe, True
    except Exception as e:
        print(f"[WARNING] Gagal load BERT NER: {e}")
        return None, False


# Mapping: label testdata → BERT entity_group
BERT_LABEL_MAP = {
    "Name":                "NAME",
    "Designation":         "DESIGNATION",
    "Location":            "LOCATION",
    "Degree":              "DEGREE",
    "College Name":        "COLLEGE_NAME",
    "Graduation Year":     "GRADUATION_YEAR",
    "Companies worked at": "COMPANIES_WORKED",
    "Skills":              "SKILLS",
    "Years of Experience": "YEARS_EXPERIENCE",
    "Email Address":       "EMAIL",
}


def bert_predict(ner_pipe, text: str) -> dict[str, list[str]]:
    """Jalankan BERT NER, return {entity_group: [text, ...]}"""
    try:
        max_chars = 2000
        chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
        all_entities = []
        for chunk in chunks:
            preds = ner_pipe(chunk)
            all_entities.extend(preds)

        grouped: dict[str, list[str]] = defaultdict(list)
        for ent in all_entities:
            grp   = ent.get("entity_group", "").replace("B-","").replace("I-","")
            word  = ent.get("word", "").strip()
            score = ent.get("score", 0)
            if grp and word and score >= 0.5:
                grouped[grp].append(word)
        return dict(grouped)
    except Exception as e:
        return {}


def bert_check_found(bert_grouped: dict[str, list[str]], label: str, gt_text: str) -> bool:
    """
    Cek apakah BERT menemukan entitas yang cocok dengan gt_text.
    """
    bert_group = BERT_LABEL_MAP.get(label)
    if not bert_group:
        return False

    bert_preds = bert_grouped.get(bert_group, [])
    gt_norm    = normalize_text(gt_text)

    for pred in bert_preds:
        pred_norm = normalize_text(pred)
        if pred_norm in gt_norm or gt_norm in pred_norm:
            return True
    return False


# ─────────────────────────────────────────────────────────────
# 4. EVALUASI PER SAMPLE
# ─────────────────────────────────────────────────────────────

def evaluate_sample(
    cv_text: str,
    gt: dict[str, list[str]],
    bert_grouped: Optional[dict] = None,
) -> dict:
    """
    Evaluasi satu CV.
    Return {label: {hybrid_found, bert_found, total, details}}
    """
    results = {}

    for label, gt_texts in gt.items():
        label_results = {
            "total":         len(gt_texts),
            "hybrid_found":  0,
            "bert_found":    0,
            "details":       [],
        }

        for gt_text in gt_texts:
            h_found, h_method = hybrid_search(gt_text, cv_text, label)
            b_found = bert_check_found(bert_grouped, label, gt_text) if bert_grouped else None

            label_results["hybrid_found"] += int(h_found)
            if b_found is not None:
                label_results["bert_found"]  += int(b_found)

            label_results["details"].append({
                "gt_text"      : gt_text[:80],
                "hybrid_found" : h_found,
                "hybrid_method": h_method,
                "bert_found"   : b_found,
            })

        results[label] = label_results
    return results


# ─────────────────────────────────────────────────────────────
# 5. AGREGAT METRICS
# ─────────────────────────────────────────────────────────────

def aggregate(all_results: list[dict], has_bert: bool) -> dict:
    """
    Agregat hasil evaluasi semua sample.
    Karena ini adalah "search/retrieval" problem:
      - TP = GT text yang berhasil ditemukan
      - FN = GT text yang tidak ditemukan
      - FP = tidak relevan untuk search (kita tidak mengukur ini dari sisi hybrid)
      -
    Recall = TP / (TP + FN) = fraction of GT instances found
    """
    totals: dict[str, dict] = defaultdict(lambda: {
        "hybrid_tp": 0, "hybrid_fn": 0,
        "bert_tp":   0, "bert_fn":   0,
        "total":     0,
    })

    for sample_results in all_results:
        for label, m in sample_results.items():
            t = totals[label]
            t["total"]     += m["total"]
            t["hybrid_tp"] += m["hybrid_found"]
            t["hybrid_fn"] += m["total"] - m["hybrid_found"]
            if has_bert:
                t["bert_tp"] += m["bert_found"]
                t["bert_fn"] += m["total"] - m["bert_found"]

    agg = {}
    for label, t in totals.items():
        h_recall = t["hybrid_tp"] / t["total"] if t["total"] > 0 else 0.0
        b_recall = (t["bert_tp"] / t["total"]  if t["total"] > 0 else 0.0) if has_bert else None

        agg[label] = {
            "hybrid_recall": round(h_recall, 4),
            "hybrid_found":  t["hybrid_tp"],
            "bert_recall":   round(b_recall, 4) if b_recall is not None else None,
            "bert_found":    t["bert_tp"] if has_bert else None,
            "total":         t["total"],
            "missed":        t["hybrid_fn"],
        }

    # Macro avg
    h_recalls = [v["hybrid_recall"] for v in agg.values()]
    agg["__MACRO_AVG__"] = {
        "hybrid_recall": round(sum(h_recalls) / len(h_recalls), 4) if h_recalls else 0,
        "hybrid_found":  sum(v["hybrid_found"] for v in agg.values() if "hybrid_found" in v),
        "total":         sum(v["total"]         for v in agg.values() if "total" in v),
    }
    if has_bert:
        b_recalls = [v["bert_recall"] for v in agg.values() if v.get("bert_recall") is not None]
        agg["__MACRO_AVG__"]["bert_recall"] = round(sum(b_recalls)/len(b_recalls), 4) if b_recalls else 0

    return agg


# ─────────────────────────────────────────────────────────────
# 6. DISPLAY
# ─────────────────────────────────────────────────────────────

LABEL_ORDER = [
    "Name", "Designation", "Location", "Degree", "College Name",
    "Graduation Year", "Companies worked at", "Skills",
    "Years of Experience", "Email Address",
]


def print_table(agg: dict, has_bert: bool):
    print(f"\n{'='*72}")
    print("  HASIL EVALUASI: Hybrid Search vs BERT NER")
    print(f"  Metric: Recall = (GT instances berhasil ditemukan) / (total GT)")
    print(f"{'='*72}")

    hdr = f"  {'Entity':25s}  {'Support':>7s}  {'Hybrid':>8s}  {'Found':>6s}"
    if has_bert:
        hdr += f"  {'BERT':>8s}  {'BFound':>6s}  {'Winner':>8s}"
    print(hdr)
    print(f"  {'-'*65}")

    for label in LABEL_ORDER:
        if label not in agg:
            continue
        m = agg[label]
        h_r = m["hybrid_recall"]
        flag = "✅ " if h_r >= 0.7 else "⚠️ " if h_r < 0.4 else "   "
        row = (f"  {flag}{label:23s}  "
               f"{m['total']:7d}  {h_r:8.3f}  {m['hybrid_found']:6d}")
        if has_bert and m.get("bert_recall") is not None:
            b_r = m["bert_recall"]
            diff = h_r - b_r
            winner = "Tie" if abs(diff) < 0.05 else ("Hybrid" if diff > 0 else "BERT")
            row += f"  {b_r:8.3f}  {m.get('bert_found',0):6d}  {winner:>8s}"
        print(row)

    print(f"  {'-'*65}")
    if "__MACRO_AVG__" in agg:
        m = agg["__MACRO_AVG__"]
        h_total = m["hybrid_found"]
        total   = m["total"]
        row = (f"  {'Macro Avg Recall':25s}  {total:7d}  "
               f"{m['hybrid_recall']:8.3f}  {h_total:6d}")
        if has_bert and m.get("bert_recall") is not None:
            row += f"  {m['bert_recall']:8.3f}"
        print(row)
    print(f"{'='*72}")


def build_report(agg: dict, has_bert: bool, n_samples: int) -> str:
    lines = [
        "=" * 72,
        "  LAPORAN EVALUASI NER: BERT vs HYBRID (Search-Based)",
        "=" * 72,
        f"  Dataset       : {TESTDATA_PATH}",
        f"  Samples       : {n_samples}",
        "",
        "  PARADIGMA EVALUASI:",
        "  ─────────────────────────────────────────────────────────────",
        "  Ground Truth (GT) = label + teks entitas yang ada di CV",
        "",
        "  Hybrid: diberikan GT text sebagai 'query dari HRD',",
        "          apakah sistem menemukan teks itu di CV?",
        "          → Simulasi alur nyata: HRD input requirement →",
        "            sistem search apakah ada di CV pelamar",
        "          → Exact search + normalized + ngram + per-keyword",
        "",
        "  BERT:   model prediksi entitas dari teks CV,",
        "          lalu cek apakah prediksi cocok dengan GT.",
        "",
        "  Metric utama: RECALL",
        "    = (berapa GT instance berhasil ditemukan) / (total GT)",
        "  ─────────────────────────────────────────────────────────────",
        "",
    ]

    hdr = f"  {'Entity':25s}  {'Support':>7s}  {'Hybrid Recall':>13s}  {'Found':>6s}"
    if has_bert:
        hdr += f"  {'BERT Recall':>11s}  {'Winner':>8s}"
    lines.append(hdr)
    lines.append("  " + "─" * (60 if not has_bert else 80))

    for label in LABEL_ORDER:
        if label not in agg:
            continue
        m = agg[label]
        h_r = m["hybrid_recall"]
        icon = "✓" if h_r >= 0.7 else "~" if h_r >= 0.4 else "✗"
        row = (f"  {icon} {label:23s}  {m['total']:7d}  "
               f"{h_r:13.3f}  {m['hybrid_found']:6d}")
        if has_bert and m.get("bert_recall") is not None:
            b_r = m["bert_recall"]
            diff = h_r - b_r
            winner = "Tie" if abs(diff) < 0.05 else ("Hybrid" if diff > 0 else "BERT")
            row += f"  {b_r:11.3f}  {winner:>8s}"
        lines.append(row)

    lines.append("  " + "─" * (60 if not has_bert else 80))

    if "__MACRO_AVG__" in agg:
        m = agg["__MACRO_AVG__"]
        row = (f"  {'Macro Avg':25s}  {m['total']:7d}  "
               f"{m['hybrid_recall']:13.3f}  {m['hybrid_found']:6d}")
        if has_bert and m.get("bert_recall") is not None:
            row += f"  {m['bert_recall']:11.3f}"
        lines.append(row)

    lines += [
        "",
        "=" * 72,
        "INTERPRETASI:",
        "─" * 50,
        "",
        "Recall tinggi (> 0.7)  : sistem berhasil menemukan sebagian besar GT",
        "Recall sedang (0.4-0.7): ada gap — perlu perbaikan coverage",
        "Recall rendah (< 0.4)  : sistem sering tidak menemukan entitas GT",
        "",
        "KENAPA HYBRID COCOK UNTUK KASUS INI:",
        "─" * 50,
        "  1. HRD sudah tahu apa yang dicari (requirement spesifik)",
        "     → Tidak perlu 'menebak' entitas dari teks bebas",
        "     → Cukup verifikasi keberadaan keyword di CV pelamar",
        "",
        "  2. CV memiliki struktur semi-terstruktur",
        "     → Skill section, education section, experience section",
        "     → Pattern lebih prediktabel daripada teks bebas",
        "",
        "  3. BERT NER lebih cocok untuk:",
        "     → Teks bebas tanpa struktur",
        "     → Kasus di mana query TIDAK diketahui sebelumnya",
        "     → Named Entity Recognition generik",
        "",
        "  4. Kekurangan Hybrid:",
        "     → GT Skills berupa section panjang, bukan keyword tunggal",
        "       → Diatasi dengan per-keyword parsing",
        "     → Perlu penanganan khusus untuk variasi penulisan",
        "       (mis. 'Accenture' vs 'accenture' → sudah ditangani normalize)",
        "",
        "=" * 72,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 7. DETAIL PER SAMPLE (verbose mode)
# ─────────────────────────────────────────────────────────────

def print_sample_details(all_results: list[dict], data: list[dict], n: int = 3):
    """Print detail evaluasi untuk N sample pertama."""
    print(f"\n{'='*72}")
    print(f"  DETAIL EVALUASI ({n} SAMPLE PERTAMA)")
    print(f"{'='*72}")

    for i, (sample_res, sample) in enumerate(zip(all_results[:n], data[:n])):
        name_ann = next(
            (pt["text"].strip() for ann in sample.get("annotation", [])
             for lbl in ann.get("label", []) if lbl == "Name"
             for pt in ann.get("points", [])), f"Sample {i+1}"
        )
        print(f"\n  Sample {i+1}: {name_ann}")
        print(f"  {'-'*60}")

        for label in LABEL_ORDER:
            if label not in sample_res:
                continue
            m = sample_res[label]
            print(f"\n  [{label}] — {m['hybrid_found']}/{m['total']} found")
            for d in m["details"]:
                icon = "✓" if d["hybrid_found"] else "✗"
                print(f"    {icon} GT: {repr(d['gt_text'][:60])}")
                if not d["hybrid_found"]:
                    print(f"      → TIDAK DITEMUKAN (method: {d['hybrid_method']})")
                else:
                    print(f"      → Found via: {d['hybrid_method']}")


# ─────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluasi Hybrid Search vs BERT NER pada testdata.json"
    )
    parser.add_argument("--testdata",    default=TESTDATA_PATH)
    parser.add_argument("--bert_model",  default=BERT_MODEL_PATH)
    parser.add_argument("--output_json", default=OUTPUT_JSON)
    parser.add_argument("--output_txt",  default=OUTPUT_TXT)
    parser.add_argument("--no_bert",     action="store_true",
                        help="Skip evaluasi BERT")
    parser.add_argument("--verbose",     action="store_true",
                        help="Tampilkan detail per sample")
    parser.add_argument("--verbose_n",   type=int, default=3,
                        help="Berapa sample yang ditampilkan detail (default: 3)")
    args = parser.parse_args()

    # ── Load data
    data = load_testdata(args.testdata)

    # ── Load BERT
    bert_pipe     = None
    bert_available = False
    if not args.no_bert:
        bert_pipe, bert_available = load_bert_model(args.bert_model)
        if not bert_available:
            print("[INFO] Evaluasi BERT dilewati (model tidak tersedia).")

    # ── Evaluasi
    print("\n[INFO] Mulai evaluasi...")
    print(f"[INFO] Paradigma: GT text → search di CV (Hybrid) / BERT predict")
    all_results = []

    for i, sample in enumerate(data):
        cv_text = sample.get("content", "")
        gt      = extract_ground_truth(sample)

        bert_grouped = bert_predict(bert_pipe, cv_text) if bert_available else None
        result       = evaluate_sample(cv_text, gt, bert_grouped)
        all_results.append(result)

        if (i + 1) % 5 == 0 or (i + 1) == len(data):
            print(f"  [{i+1}/{len(data)}] diproses")

    # ── Agregat
    agg = aggregate(all_results, bert_available)

    # ── Tampilkan
    print_table(agg, bert_available)

    if args.verbose:
        print_sample_details(all_results, data, args.verbose_n)

    # ── Report
    report = build_report(agg, bert_available, len(data))
    print("\n" + report)

    # ── Simpan
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({
            "paradigm": "GT text → search in CV (recall-focused)",
            "n_samples": len(data),
            "aggregate": agg,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] JSON  → {args.output_json}")

    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[INFO] TXT   → {args.output_txt}")


if __name__ == "__main__":
    main()