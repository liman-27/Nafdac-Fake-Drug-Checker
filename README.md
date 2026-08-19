<div align="center">

# AI-06: Fake Drug Text/Barcode Checker
**Fighting counterfeit medicine in Nigeria with machine learning and NAFDAC's official Greenbook registry**
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/status-MVP-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

</div>

## Overview
Counterfeit, NAFDAC-numbered drugs are a real and dangerous problem across
Nigeria — patients frequently have no reliable way to tell a genuine
registered medicine from a fake one just by reading the pack. it isn't case sensitive
and can detect typo'd or fabricated product.

**This project is a classifier + lookup tool** that takes a product name and
its NRN (NAFDAC Registration Number) and returns a verdict — **GENUINE** or
**SUSPICIOUS** — along with a plain-English explanation of *why*, not just a
black-box label.

> Built as part of the **3MTT AI & Machine Learning Fellowship Graduation requirement**, brief
> **AI-06: Fake Drug Text/Barcode Checker**.

## Table of Contents
- [How It Works](#-how-it-works)
- [Why Synthetic Fakes?](#-why-synthetic-fakes)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Scaling to the Full Dataset](#-scaling-to-the-full-dataset)
- [Data Source](#-data-source)
- [Limitations & Roadmap](#-limitations--roadmap)
- [Author](#-author)

## How It Works
```
                NAFDAC Greenbook (greenbook.nafdac.gov.ng)
                              │
                     (web scraping)
                              ▼
              ┌───────────────────────────┐
              │  01_parse_data.py         │  Clean genuine products
              │  01b_parse_drugs.py       │  (name, NRN, category)
              └──────────────┬────────────┘
                              ▼
              ┌───────────────────────────┐
              │  02_generate_fakes.py     │  Synthetic "suspicious"
              │                           │  examples (typos, tampered
              │                           │  NRNs, fabrications)
              └──────────────┬────────────┘
                              ▼
              ┌───────────────────────────┐
              │  03_features.py           │  Numeric features:
              │                           │  name similarity, NRN match,
              │                           │  NRN format, mismatch flag
              └──────────────┬────────────┘
                              ▼
              ┌───────────────────────────┐
              │  04_train_model.py        │  Random Forest classifier
              │                           │  trained + evaluated
              └──────────────┬────────────┘
                              ▼
              ┌───────────────────────────┐
              │  05_app.py                │  Interactive checker app
              │                           │  (the actual MVP tool)
              └───────────────────────────┘
```
## Why Synthetic Fakes Drugs?
NAFDAC's public Greenbook only lists **genuine, approved products** — there
is no public dataset of confirmed counterfeit entries to train on. To work
around this, the model is trained on synthetic "suspicious" examples built
from three real-world counterfeiting patterns:

| Pattern | Example |
|---|---|
| **Typo/character-swap names** | `Simulect` → `Simulcet` |
| **Tampered NRN codes** | `A6-0405` → `A6-0455` |
| **Fully fabricated products** | `"Super Cure Tablet"` with a made-up NRN |

This is a standard technique called **negative sampling**, used whenever a
real-world problem only has confirmed examples of one class.

## Results
Evaluated on a held-out 20% test set the model never saw during training:

| Metric | Score | What it means |
|---|---|---|
| **Accuracy** | 97.2% | Overall correctness |
| **Recall (genuine)** | **100%** | Never wrongly flags a real drug as fake |
| **Precision (genuine)** | 92.2% | Of predicted-genuine, how many really are |
| **F1 Score** | 95.9% | Balance of precision & recall |

**100% recall on genuine products is the number that matters most here** —
for a tool like this, a false "suspicious" alarm on a real medicine is far
more dangerous than being cautious about a fake, since it could scare a
patient away from treatment they actually need.

**Top features the model relies on:**
1. `name_similarity` (58.5%) — how close the name is to a real registered product
2. `name_nrn_mismatch` (22.5%) — does this name+NRN pairing exist for real
3. `nrn_exact_match` (10%)
4. `nrn_similarity` (9%)

## Project Structure
```
nafdac_checker/
├── data/
│   ├── raw_vaccines.txt              # Raw scraped text (Vaccines/Biologics)
│   ├── raw_drugs_plaintext.txt       # Raw scraped text (Drugs)
│   ├── nafdac_products_clean.csv     # 235 genuine products, cleaned
│   ├── training_data.csv             # Genuine + synthetic-fake, labeled
│   ├── features.csv                  # Numeric features for training
│   └── fake_drug_model.joblib        # Trained model, ready to load
├── 01_parse_data.py                  # Parse Vaccines/Biologics
├── 01b_parse_drugs.py                # Parse Drugs (different raw format)
├── 02_generate_fakes.py              # Generate synthetic suspicious examples
├── 03_features.py                    # Feature engineering
├── 04_train_model.py                 # Train + evaluate the classifier
├── 05_app.py                         # Interactive checker app
├── scrape_full_nafdac.py             # Full scraper for all 6 NAFDAC categories
├── build_notebook.py                 # Assembles the .py scripts into the notebook
├── Fake_Drug_Checker.ipynb           # Full pipeline in one notebook
└── README.md
```

## Getting Started
**Requirements:** Python 3.10+
```bash
cd nafdac-fake-drug-checker
pip install pandas scikit-learn joblib requests beautifulsoup4
```

Run the pipeline step by step:
```bash
python 01_parse_data.py
python 01b_parse_drugs.py
python 02_generate_fakes.py
python 03_features.py
python 04_train_model.py
python 05_app.py
```
Or open **`Fake_Drug_Checker.ipynb`** in Jupyter/VS Code and click **Run All**
to see the entire pipeline — data cleaning, training, and evaluation — in
one place.

## Usage
`05_app.py` runs a short demo automatically, then drops into interactive mode:
```
Checking: 'Simulect'  |  NRN: A6-0405
  -> Verdict: GENUINE  (confidence: 98.0%)
     - NRN 'A6-0405' matches a registered NAFDAC product exactly.
     - Product name matches 'Simulect' exactly.

Checking: 'Simulcet'  |  NRN: A6-0405
  -> Verdict: SUSPICIOUS  (confidence: 91.5%)
     - NRN 'A6-0405' matches a registered NAFDAC product exactly.
     - Product name is suspiciously CLOSE to a real product
       ('Simulect', 93% similar) but not an exact match
       - possible typo-squatting.
```
## Data Source
Scraped from the official NAFDAC Greenbook:
**[greenbook.nafdac.gov.ng](https://greenbook.nafdac.gov.ng)**

## Limitations & Roadmap
-**Synthetic fakes, not real ones** — production version should validate
      against real NAFDAC/WHO counterfeit alert bulletins
-**Partial category coverage** in this build — run `scrape_full_nafdac.py`
      for the full ~1,900+ drug list
-**No barcode/GS1 scanning yet** — text lookup only; `pyzbar` would add
      phone-camera barcode support
-**No UI yet** — a Streamlit front-end would make this demoable to
      non-technical stakeholders (pharmacists, patients, regulators)

## Author
**Alhassan Aliyu Liman (Aliyu)**
3MTT AI & ML Fellow

<div align="center">

*Built to help protect patients from counterfeit medicine, one lookup at a time.*
</div>