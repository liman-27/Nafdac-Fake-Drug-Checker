"""
STEP 3: Feature engineering.

A machine learning model can't read "Amoxicilin" and just know it's wrong -
it needs NUMBERS that capture the *pattern* of what looks suspicious.

For every product entry I checked, I computed 5 signals by comparing
it against the genuine NAFDAC database:

1. name_similarity   - how close is this name to the CLOSEST genuine name?
                        (1.0 = exact match, 0.0 = nothing like it)
2. nrn_exact_match    - is this exact NRN code in our genuine database? (1/0)
3. nrn_format_valid   - does the NRN follow NAFDAC's pattern, e.g. "A6-100070"? (1/0)
4. nrn_similarity     - how close is the NRN to the closest genuine NRN?
5. name_nrn_mismatch  - does this name+NRN combo exist together for real?
                        (catches: real name + wrong NRN, or vice versa)

This is the "secret sauce" - a typo'd name will have HIGH name_similarity
(close but not perfect) while a fabricated product will have LOW similarity
to everything. The model learns to tell these apart.
"""
import csv
import re
import difflib

def load_genuine_db(path="data/nafdac_products_clean.csv"):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    names = [r["product_name"] for r in rows]
    nrns = [r["nrn"] for r in rows if r["nrn"]]
    name_to_nrn = {r["product_name"]: r["nrn"] for r in rows}
    return names, nrns, name_to_nrn

NRN_PATTERN = re.compile(r"^[A-Za-z0-9]{1,3}-\d{2,7}$")

def closest_match_score(query, candidates):
    """difflib gives a 0-1 similarity ratio - like a simple fuzzy-match score."""
    if not candidates:
        return 0.0, None
    best_score, best_match = 0.0, None
    for c in candidates:
        score = difflib.SequenceMatcher(None, query.lower(), c.lower()).ratio()
        if score > best_score:
            best_score, best_match = score, c
    return best_score, best_match

def extract_features(product_name, nrn, names_db, nrns_db, name_to_nrn):
    name_sim, closest_name = closest_match_score(product_name, names_db)
    nrn_exact = 1 if nrn in nrns_db else 0
    nrn_valid_format = 1 if nrn and NRN_PATTERN.match(nrn) else 0
    nrn_sim, _ = closest_match_score(nrn or "", nrns_db)

    # Does the closest-matching name actually pair with THIS nrn in real life?
    expected_nrn = name_to_nrn.get(closest_name, "")
    name_nrn_mismatch = 1 if (name_sim > 0.85 and nrn != expected_nrn) else 0

    return {
        "name_similarity": round(name_sim, 4),
        "nrn_exact_match": nrn_exact,
        "nrn_format_valid": nrn_valid_format,
        "nrn_similarity": round(nrn_sim, 4),
        "name_nrn_mismatch": name_nrn_mismatch,
    }


if __name__ == "__main__":
    names_db, nrns_db, name_to_nrn = load_genuine_db()

    training_rows = list(csv.DictReader(open("data/training_data.csv", encoding="utf-8")))

    feature_rows = []
    for row in training_rows:
        feats = extract_features(row["product_name"], row["nrn"], names_db, nrns_db, name_to_nrn)
        feats["label"] = row["label"]
        feats["product_name"] = row["product_name"]
        feature_rows.append(feats)

    out_path = "data/features.csv"
    fieldnames = ["product_name", "name_similarity", "nrn_exact_match",
                  "nrn_format_valid", "nrn_similarity", "name_nrn_mismatch", "label"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)

    print(f"Features computed for {len(feature_rows)} rows -> {out_path}")
    print("\nSample (genuine vs suspicious side by side):")
    for r in feature_rows[:4]:
        print(f"  label={r['label']}  name_sim={r['name_similarity']:.2f}  "
              f"nrn_exact={r['nrn_exact_match']}  nrn_valid={r['nrn_format_valid']}  "
              f"| {r['product_name'][:40]}")
