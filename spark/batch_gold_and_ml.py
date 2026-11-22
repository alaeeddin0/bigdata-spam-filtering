import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

BASE_DIR = "/opt/airflow/data"
SILVER_PATH = f"{BASE_DIR}/silver/messages_clean.parquet"
GOLD_DIR = f"{BASE_DIR}/gold"

os.makedirs(GOLD_DIR, exist_ok=True)

# 1. Lecture de la Zone Silver
df = pd.read_parquet(SILVER_PATH)
df["date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)

# 2. Calcul des agrégats pour la Zone Gold (Volume par jour)
daily_stats = df.groupby("date").agg(
    total_messages=("message_id", "count"),
    spam_count=("is_spam", "sum")
).reset_index()
daily_stats["ham_count"] = daily_stats["total_messages"] - daily_stats["spam_count"]
daily_stats["spam_rate"] = round((daily_stats["spam_count"] / daily_stats["total_messages"]) * 100, 2)
daily_stats.to_parquet(f"{GOLD_DIR}/daily_spam_stats.parquet", index=False)

# 3. Calcul des expéditeurs suspects (Zone Gold)
suspects = df[df["is_spam"] == 1].groupby("sender").agg(
    spam_sent=("message_id", "count")
).reset_index().sort_values(by="spam_sent", ascending=False)
suspects.to_parquet(f"{GOLD_DIR}/top_suspect_senders.parquet", index=False)

# 4. Réentraînement et évaluation du modèle ML
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["is_spam"], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
    "precision": round(precision_score(y_test, y_pred) * 100, 2),
    "recall": round(recall_score(y_test, y_pred) * 100, 2),
    "samples_trained": len(X_train)
}

with open(f"{GOLD_DIR}/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Zone Gold et métriques ML mises à jour avec succès :", metrics)