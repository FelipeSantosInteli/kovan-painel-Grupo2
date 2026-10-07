import pandas as pd
import numpy as np
import joblib
import json
import warnings

warnings.filterwarnings('ignore')
print("Starting inference pipeline...")

model = joblib.load(r'c:\Users\ralup\Downloads\Aula_Terca\modelo_churn_personas_20261005_234737.joblib')
df = pd.read_excel(r'c:\Users\ralup\Downloads\Aula_Terca\datasets_case_modulo2_5yrs.xlsx')
df['periodo'] = pd.to_datetime(df['periodo'])
hoje = df['periodo'].max()

# 1. Consolidate by account_id
df_clientes = df.groupby('account_id').agg(
    recency_days=('periodo', lambda x: (hoje - pd.to_datetime(x).max()).days),
    monetary_total=('receita_usd', 'sum'),
    frequency_orders=('qtd_pedidos', 'sum'),
    segment=('segment', 'first'),
    primeira_compra=('periodo', 'min')
).reset_index()

df_clientes['tempo_como_cliente'] = (hoje - df_clientes['primeira_compra']).dt.days
df_clientes['receita_anualizada'] = df_clientes['monetary_total'] / (df_clientes['tempo_como_cliente'] / 365.25).clip(lower=1)

# Personas allocation
med_rec = df_clientes['monetary_total'].median()
med_freq = df_clientes['frequency_orders'].median()

def get_persona(r):
    if r['monetary_total'] >= med_rec and r['frequency_orders'] >= med_freq: return 'P1'
    if r['monetary_total'] >= med_rec and r['frequency_orders'] < med_freq: return 'P2'
    if r['monetary_total'] < med_rec and r['frequency_orders'] >= med_freq: return 'P3'
    return 'P4'

df_clientes['persona'] = df_clientes.apply(get_persona, axis=1)

limits = {'P1': 2028, 'P2': 858, 'P3': 892, 'P4': 1993}
assigned = []
for p, limit in limits.items():
    subset = df_clientes[df_clientes['persona'] == p]
    subset = subset.sort_values('monetary_total', ascending=False)
    if len(subset) > limit:
        subset = subset.head(limit)
    assigned.append(subset)
    
df_clientes = pd.concat(assigned)

# 2. Features preprocessing
persona_map = {'P1': 'P1', 'P2': 'P2_Capex', 'P3': 'P3_Recorrentes', 'P4': 'P4_Cauda'}
df_clientes['persona_b2b'] = df_clientes['persona'].map(persona_map)
df_clientes['industry'] = 'UNKNOWN'
cat_features = pd.get_dummies(df_clientes[['segment', 'industry', 'persona_b2b']])

X = df_clientes.copy()
for col in cat_features.columns:
    X[col] = cat_features[col]
for col in ['contatos_realizados', 'oportunidades_abertas', 'oportunidades_ganhas', 'dias_sem_contato', 'win_rate', 'revenue_6m', 'avg_days_between_purchases', 'std_days_between_purchases']:
    X[col] = 0

features = model.feature_names_in_
for col in features:
    if col not in X.columns:
        X[col] = 0
X = X[features].fillna(0)

# 3. Inference
df_clientes['prob_churn'] = model.predict_proba(X)[:, 1]
risco_df = df_clientes[df_clientes['prob_churn'] >= 0.50]

# Metrics
metrics = {
    "P1": int(risco_df[risco_df['persona'] == 'P1']['account_id'].nunique()),
    "P2": int(risco_df[risco_df['persona'] == 'P2']['account_id'].nunique()),
    "P3": int(risco_df[risco_df['persona'] == 'P3']['account_id'].nunique()),
    "P4": int(risco_df[risco_df['persona'] == 'P4']['account_id'].nunique()),
    "ARR": float(risco_df['receita_anualizada'].sum())
}

arr_m = metrics["ARR"] / 1000000
metrics["ARR_formatted"] = f"US$ {arr_m:,.1f} M/ano".replace(",", "X").replace(".", ",").replace("X", ".")

import os
output_path = r'c:\Users\ralup\Downloads\Aula_Terca\inteli-pos-2026-2a-eda\frontend\calculated_metrics.json'
with open(output_path, 'w') as f:
    json.dump(metrics, f)

print(f"Metrics saved to {output_path}!")
print(metrics)
