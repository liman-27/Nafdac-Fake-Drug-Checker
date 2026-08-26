"""
STEP 1b: 
Break down the Drugs category. NAFDAC provided the Drugs category in a DIFFERENT raw
format than they did for Vaccines - plain text with no [name](url) brackets, just one long run-on
string of "Name FORM Ingredient Strength NRN: xxx" blocks back to back.

The trick: NRN codes ("NRN: A4-100160") are the one reliable anchor. I split the whole blob on that pattern,
everything between an NRN and the next NRN is part of the entry that ends with that second NRN.

"""
import re
import csv

NRN_TOKEN = re.compile(r"NRN:\s*([A-Za-z0-9]{1,4}-\d{2,7})")

def parse_plaintext_drugs(filepath, category_name="Drugs"):
    text = open(filepath, encoding="utf-8").read()

    # Find every "...stuff... NRN: CODE" chunk
    matches = list(NRN_TOKEN.finditer(text))
    rows = []
    start = 0
    for m in matches:
        chunk = text[start:m.end()].strip()
        nrn = m.group(1)
        # The short product name is the text before the '##' / '**' / '#'
        # marker NAFDAC uses, same trick as the bracketed format
        name_match = re.match(r"^(.*?)(\*\*|##|#)", chunk)
        if name_match:
            short_name = name_match.group(1).strip()
        else:
            short_name = " ".join(chunk.split()[:6])
        rows.append({
            "product_name": short_name,
            "full_listing_text": chunk,
            "nrn": nrn,
            "category": category_name,
            "source_url": "",
        })
        start = m.end()
    return rows


if __name__ == "__main__":
    drug_rows = parse_plaintext_drugs("data/raw_drugs_plaintext.txt")
    print(f"Parsed {len(drug_rows)} genuine Drugs entries")
    for r in drug_rows[:5]:
        print(f" - {r['product_name']}  |  NRN: {r['nrn']}")

    # Merge with the existing vaccines dataset into one combined file
    existing = list(csv.DictReader(open("data/nafdac_products_clean.csv", encoding="utf-8")))
    combined = existing + drug_rows

    with open("data/nafdac_products_clean.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "product_name", "full_listing_text", "nrn", "category", "source_url"
        ])
        writer.writeheader()
        writer.writerows(combined)

    print(f"\nCombined dataset now has {len(combined)} genuine products -> data/nafdac_products_clean.csv")
