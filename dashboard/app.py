import os
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IL-15 | Monitoring Anti-Spam", page_icon="🛡️", layout="wide")
st.title("🛡️ Tableau de Bord — Filtrage de Spam à Grande Échelle (IL-15)")

daily_path = "/app/data/gold/daily_spam_stats.parquet"
suspects_path = "/app/data/gold/top_suspect_senders.parquet"
metrics_path = "/app/data/gold/model_metrics.json"

if not os.path.exists(daily_path):
    st.warning("⚠️ Aucune donnée trouvée dans la Zone Gold. Lancez d'abord le DAG dans Airflow !")
    st.stop()

df_daily = pd.read_parquet(daily_path)
df_suspects = pd.read_parquet(suspects_path)
with open(metrics_path, "r") as f:
    metrics = json.load(f)

# KPI Principaux
col1, col2, col3, col4 = st.columns(4)
total_msg = int(df_daily["total_messages"].sum())
total_spam = int(df_daily["spam_count"].sum())
global_rate = round((total_spam / total_msg) * 100, 1) if total_msg > 0 else 0

col1.metric("Messages Analysés", f"{total_msg:,}")
col2.metric("Spams Bloqués", f"{total_spam:,}")
col3.metric("Taux de Spam Global", f"{global_rate} %")
col4.metric("Précision Modèle ML", f"{metrics['precision']} %", f"Rappel : {metrics['recall']} %")

st.divider()

# Graphiques
left, right = st.columns([2, 1])

with left:
    st.subheader("📈 Évolution quotidienne : Spams vs Messages Légitimes")
    chart_data = df_daily.set_index("date")[["spam_count", "ham_count"]]
    st.bar_chart(chart_data)

with right:
    st.subheader("🚨 Top Expéditeurs Suspects")
    st.dataframe(df_suspects, use_container_width=True, hide_index=True)