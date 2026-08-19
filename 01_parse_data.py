"""
STEP 1: Parse raw scraped NAFDAC Greenbook text into a clean, structured CSV.

Why this step matters:
Web-scraped data is messy — it comes as long strings mixing product name,
dosage form, active ingredient, strength, and registration number (NRN) all
together. Before we can do ANY machine learning, we need this in neat rows
and columns (like a spreadsheet). This is called "data cleaning" and it's
usually 70% of the work in a real ML project.
"""
import re
import csv

def parse_category_file(filepath, category_name):
    """
    Each line in our raw file looks like:
    [Product Name** Form Ingredient Strength NRN: A6-100070](https://.../details/6931)

    We use a "regular expression" (regex) - a pattern-matching tool - to pull
    out the pieces we care about: the display text, and the NRN code.
    """
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Match: [ ...text... ](url)
            m = re.match(r"^\[(.*?)\]\((https?://\S+)\)$", line)
            if not m:
                continue
            display_text, url = m.group(1), m.group(2)

            # Pull out the NRN registration number, e.g. "NRN: A6-100070"
            nrn_match = re.search(r"NRN:\s*([A-Za-z0-9\-]+)", display_text)
            nrn = nrn_match.group(1) if nrn_match else None

            # The product's short name is usually everything before the
            # first '**', '##', or '#' marker NAFDAC uses in their listing
            name_match = re.match(r"^(.*?)(\*\*|##|#)", display_text)
            if name_match:
                short_name = name_match.group(1).strip()
            else:
                # Fallback: take text up to " NA " or the first number-heavy
                # dosage chunk - just use first 8 words as a rough cut
                short_name = " ".join(display_text.split()[:8])

            rows.append({
                "product_name": short_name,
                "full_listing_text": display_text,
                "nrn": nrn,
                "category": category_name,
                "source_url": url,
            })
    return rows


if __name__ == "__main__":
    all_rows = []
    all_rows += parse_category_file("data/raw_vaccines.txt", "Vaccines and Biologics")

    out_path = "data/nafdac_products_clean.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "product_name", "full_listing_text", "nrn", "category", "source_url"
        ])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Parsed {len(all_rows)} genuine NAFDAC products -> {out_path}")
    print("\nSample rows:")
    for r in all_rows[:5]:
        print(f" - {r['product_name']}  |  NRN: {r['nrn']}")
