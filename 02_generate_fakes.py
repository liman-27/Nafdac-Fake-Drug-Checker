"""
STEP 2: Generate synthetic "suspicious" entries to pair with genuine data for training.

I don't have real counterfeit examples, so I used the 3 most common
real-world counterfeiting patterns to simulate fakes for training, which are:
  1. Typo/character-swap, attacks on the product name
  2. Tampered NRN (registration number) codes, swapped or changed completely
  3. Completely fabricated product names

This gives the classifier both classes (genuine=1, suspicious=0) to learn from.
"""
import csv
import random
import re

random.seed(42)  # makes results reproducible - same output every run

def typo_name(name):
    """Simulate a typo: swap two adjacent letters, or drop one letter."""
    if len(name) < 4:
        return name + "x"
    i = random.randint(1, len(name) - 2)
    chars = list(name)
    action = random.choice(["swap", "drop", "double"])
    if action == "swap":
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif action == "drop":
        del chars[i]
    else:
        chars.insert(i, chars[i])
    return "".join(chars)

def tamper_nrn(nrn):
    """Simulate a tampered NRN: change one digit."""
    if not nrn:
        return "A0-000000"
    chars = list(nrn)
    digit_positions = [i for i, c in enumerate(chars) if c.isdigit()]
    if digit_positions:
        pos = random.choice(digit_positions)
        chars[pos] = random.choice("0123456789")
    return "".join(chars)

FAKE_PRODUCT_NAMES = [
    "Super Cure Tablet", "MiracleVax Injection", "PowerHeal Capsule",
    "QuickFix Antibiotic", "MaxStrength Syrup", "InstaHeal Suspension",
    "ForteMed Tablet", "BioShield Vaccine", "PureLife Injection",
    "RapidCure Solution", "VitaBoost Capsule", "TrustMed Tablet",
    "GenCure Injection", "HealFast Syrup", "MedPlus Extra Tablet",
]

def generate_fakes(genuine_rows, n_per_genuine=1):
    fakes = []
    for row in genuine_rows:
        # Pattern 1: typo in a real product's name, kept with a fake-ish NRN
        fakes.append({
            "product_name": typo_name(row["product_name"]),
            "nrn": row["nrn"],  # NRN copied correctly - name is the giveaway
            "label": 0,
            "fake_type": "name_typo",
        })
        # Pattern 2: correct name, tampered NRN
        fakes.append({
            "product_name": row["product_name"],
            "nrn": tamper_nrn(row["nrn"]),
            "label": 0,
            "fake_type": "nrn_tamper",
        })

    # Pattern 3: fully fabricated products with plausible-looking fake NRNs
    for name in FAKE_PRODUCT_NAMES:
        fake_nrn = f"A{random.randint(1,9)}-{random.randint(100000,999999)}"
        fakes.append({
            "product_name": name,
            "nrn": fake_nrn,
            "label": 0,
            "fake_type": "fabricated",
        })
    return fakes


if __name__ == "__main__":
    genuine_rows = list(csv.DictReader(open("data/nafdac_products_clean.csv", encoding="utf-8")))

    fakes = generate_fakes(genuine_rows)

    # Build the full labeled training set: genuine (label=1) + fake (label=0)
    training_rows = []
    for row in genuine_rows:
        training_rows.append({
            "product_name": row["product_name"],
            "nrn": row["nrn"],
            "label": 1,
            "fake_type": "genuine",
        })
    training_rows += fakes

    random.shuffle(training_rows)

    out_path = "data/training_data.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["product_name", "nrn", "label", "fake_type"])
        writer.writeheader()
        writer.writerows(training_rows)

    n_genuine = sum(1 for r in training_rows if r["label"] == 1)
    n_fake = sum(1 for r in training_rows if r["label"] == 0)
    print(f"Training set built -> {out_path}")
    print(f"  Genuine (label=1): {n_genuine}")
    print(f"  Suspicious (label=0): {n_fake}")
    print(f"  Total: {len(training_rows)}")
