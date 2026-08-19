"""
STEP 5: The Fake Drug Text/Barcode Checker app.

This ties everything together into the tool a real user (e.g. a pharmacist
or a patient) would actually use: type in a product name and its NRN number,
get back a verdict AND a plain-English explanation of why.

Run it with:  python3 05_app.py
"""
import joblib
import pandas as pd
import importlib.util

# Python module names can't start with a digit, so we load 03_features.py
# by file path instead of a normal import statement.
_spec = importlib.util.spec_from_file_location("features_mod", "03_features.py")
_features_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_features_mod)
load_genuine_db = _features_mod.load_genuine_db
extract_features = _features_mod.extract_features
closest_match_score = _features_mod.closest_match_score

FEATURE_COLS = ["name_similarity", "nrn_exact_match", "nrn_format_valid",
                 "nrn_similarity", "name_nrn_mismatch"]

def load_everything():
    model = joblib.load("data/fake_drug_model.joblib")
    names_db, nrns_db, name_to_nrn = load_genuine_db()
    return model, names_db, nrns_db, name_to_nrn

def check_product(product_name, nrn, model, names_db, nrns_db, name_to_nrn):
    feats = extract_features(product_name, nrn, names_db, nrns_db, name_to_nrn)
    X = pd.DataFrame([feats])[FEATURE_COLS]

    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0][prediction]

    verdict = "GENUINE" if prediction == 1 else "SUSPICIOUS"

    # Build a plain-English explanation - this is what makes it trustworthy,
    # not just a black box saying "fake" with no reason
    closest_name_score, closest_name = closest_match_score(product_name, names_db)
    reasons = []
    if feats["nrn_exact_match"] == 1:
        reasons.append(f"NRN '{nrn}' matches a registered NAFDAC product exactly.")
    else:
        reasons.append(f"NRN '{nrn}' was NOT found in the NAFDAC registry.")

    if closest_name_score > 0.99:
        reasons.append(f"Product name matches '{closest_name}' exactly.")
    elif closest_name_score > 0.80:
        reasons.append(
            f"Product name is suspiciously CLOSE to a real product "
            f"('{closest_name}', {closest_name_score:.0%} similar) but not an exact match "
            f"- possible typo-squatting."
        )
    else:
        reasons.append(
            f"Product name doesn't closely resemble any registered NAFDAC product "
            f"(closest match: '{closest_name}', only {closest_name_score:.0%} similar)."
        )

    if feats["name_nrn_mismatch"] == 1:
        reasons.append("The name and NRN don't belong together in the real registry.")

    return {
        "verdict": verdict,
        "confidence": round(confidence * 100, 1),
        "reasons": reasons,
    }


if __name__ == "__main__":
    model, names_db, nrns_db, name_to_nrn = load_everything()

    print("=" * 55)
    print(" NAFDAC Fake Drug / Barcode Checker (MVP)")
    print("=" * 55)
    print("Type 'quit' to exit.\n")

    # A few demo checks to show it working end-to-end
    demo_cases = [
        ("Simulect", "A6-0405"),                # genuine, exact
        ("Simulcet", "A6-0405"),                 # typo'd name
        ("Ocrevus", "A6-999999"),                # real name, fake NRN
        ("Super Cure Tablet", "A9-123456"),      # fully fabricated
    ]

    print("--- Demo run ---")
    for name, nrn in demo_cases:
        result = check_product(name, nrn, model, names_db, nrns_db, name_to_nrn)
        print(f"\nChecking: '{name}'  |  NRN: {nrn}")
        print(f"  -> Verdict: {result['verdict']}  (confidence: {result['confidence']}%)")
        for r in result["reasons"]:
            print(f"     - {r}")

    print("\n" + "=" * 55)
    print("--- Interactive mode ---")
    while True:
        name = input("\nProduct name (or 'quit'): ").strip()
        if name.lower() == "quit":
            break
        nrn = input("NRN code on the pack: ").strip()
        result = check_product(name, nrn, model, names_db, nrns_db, name_to_nrn)
        print(f"\n  -> Verdict: {result['verdict']}  (confidence: {result['confidence']}%)")
        for r in result["reasons"]:
            print(f"     - {r}")
