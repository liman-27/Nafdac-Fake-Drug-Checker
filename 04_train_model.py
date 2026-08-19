"""
STEP 4: Train a classifier and evaluate it properly.

We use a Random Forest - it builds many small decision trees (e.g. "IF
name_similarity < 0.95 AND nrn_exact_match == 0 THEN suspicious") and
averages their votes. It's a great choice for beginners because:
  - It handles our small feature set well
  - It's hard to badly misconfigure
  - It tells us which features actually mattered (interpretable)

CRITICAL RULE: we NEVER evaluate a model on the same data it trained on -
that's like grading a student's exam using the answer key they copied from.
We split into train/test sets so the model is judged on examples it has
never seen.
"""
import csv
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

FEATURE_COLS = ["name_similarity", "nrn_exact_match", "nrn_format_valid",
                 "nrn_similarity", "name_nrn_mismatch"]

df = pd.read_csv("data/features.csv")
X = df[FEATURE_COLS]
y = df["label"]

# 80% train, 20% test. stratify=y keeps the genuine/fake ratio balanced
# in both splits so the test set is a fair, representative sample.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,   # 200 decision trees voting together
    max_depth=6,        # keeps trees simple - avoids memorizing noise
    random_state=42,
)
model.fit(X_train, y_train)

# --- Evaluate on the held-out TEST set (data the model never saw) ---
y_pred = model.predict(X_test)

print("=" * 55)
print("EVALUATION RESULTS (on unseen test data)")
print("=" * 55)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}  "
      "(of predicted-genuine, how many really are genuine)")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}  "
      "(of actually-genuine, how many we correctly caught)")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Predicted Fake   Predicted Genuine")
print(f"  Actual Fake         {cm[0][0]:<15} {cm[0][1]}")
print(f"  Actual Genuine      {cm[1][0]:<15} {cm[1][1]}")

print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=["Suspicious", "Genuine"]))

print("Feature importance (what the model relies on most):")
importances = sorted(zip(FEATURE_COLS, model.feature_importances_),
                      key=lambda x: -x[1])
for feat, imp in importances:
    bar = "#" * int(imp * 50)
    print(f"  {feat:<20} {imp:.3f} {bar}")

# Save the trained model to disk so the app can load it instantly
joblib.dump(model, "data/fake_drug_model.joblib")
print("\nModel saved -> data/fake_drug_model.joblib")
