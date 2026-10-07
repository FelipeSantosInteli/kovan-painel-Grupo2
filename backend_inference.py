import os
import re
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
print("Starting inference pipeline...")

# Diretório base: raiz deste projeto
BASE_DIR = Path(__file__).resolve().parent

class JSChurnModel:
    """Carrega e executa o modelo de churn baseado em JavaScript (modelo_churn_dinamico.js).
    
    Extrai a lista de features e transpila a função `score(input)` de árvore de decisão /
    ensemble para Python nativo, garantindo alta performance sem dependência de runtimes externos.
    """
    def __init__(self, js_path: str | Path):
        self.js_path = Path(js_path)
        self._load()

    def _load(self):
        if not self.js_path.exists():
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {self.js_path}")

        with open(self.js_path, "r", encoding="utf-8") as f:
            js_content = f.read()

        # Extrai nomes das features a partir dos comentários no cabeçalho do script JS
        feature_lines = re.findall(r'input\[(\d+)\]\s*=\s*(.+)', js_content)
        if feature_lines:
            self.feature_names_in_ = np.array([
                m[1].strip() for m in sorted(feature_lines, key=lambda x: int(x[0]))
            ])
        else:
            self.feature_names_in_ = np.array([
                'frequency_orders', 'monetary_total', 'recency_days',
                'avg_days_between_purchases', 'std_days_between_purchases', 'revenue_6m',
                'tempo_como_cliente', 'contatos_realizados', 'oportunidades_abertas',
                'oportunidades_ganhas', 'dias_sem_contato', 'win_rate',
                'receita_anualizada', 'segment_LARGE ENTERPRISE', 'segment_MID MARKET',
                'segment_PUBLIC SECTOR', 'segment_SMALL MARKET', 'segment_STRATEGIC ACCOUNT',
                'industry_COMMUNICATIONS, MEDIA AND SERVICES', 'industry_EDUCATION',
                'industry_GOVERNMENT', 'industry_HEALTHCARE AND LIFE SCIENCES',
                'industry_INSURANCE', 'industry_MANUFACTURING AND NATURAL RESOURCES',
                'industry_OIL AND GAS', 'industry_POWER AND UTILITIES', 'industry_RETAIL',
                'industry_TRANSPORTATION', 'industry_UNCLASSIFIED', 'industry_UNKNOWN',
                'industry_WHOLESALE TRADE', 'persona_b2b_P2_Capex',
                'persona_b2b_P3_Recorrentes', 'persona_b2b_P4_Cauda'
            ])

        # Transpila a função score(input) do JS para Python nativo
        py_lines = ['def score(input):']
        indent = 1
        for raw_line in js_content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(('/*', '*', '//', 'function score')):
                continue
            if line.startswith('var var') and ';' in line and '=' not in line:
                continue
            if line.startswith('if (') and line.endswith('{'):
                cond = line[3:].rstrip('{').strip()
                if cond.startswith('(') and cond.endswith(')'):
                    cond = cond[1:-1]
                py_lines.append('    ' * indent + f'if {cond}:')
                indent += 1
            elif line.startswith('} else {'):
                indent -= 1
                py_lines.append('    ' * indent + 'else:')
                indent += 1
            elif line.startswith('} else if (') and line.endswith('{'):
                indent -= 1
                cond = line[10:].rstrip('{').strip()
                if cond.startswith('(') and cond.endswith(')'):
                    cond = cond[1:-1]
                py_lines.append('    ' * indent + f'elif {cond}:')
                indent += 1
            elif line == '}':
                indent = max(1, indent - 1)
            elif '=' in line and not line.startswith('return'):
                stmt = line.rstrip(';').replace('var ', '')
                py_lines.append('    ' * indent + stmt)
            elif 'return mulVectorNumber' in line:
                tree_vars = [f'var{i}' for i in range(100)]
                py_lines.append('    p0 = sum(v[0] for v in [' + ', '.join(tree_vars) + ']) * 0.01')
                py_lines.append('    p1 = sum(v[1] for v in [' + ', '.join(tree_vars) + ']) * 0.01')
                py_lines.append('    return [p0, p1]')
                break

        exec_scope = {}
        exec('\n'.join(py_lines), exec_scope)
        self._score_fn = exec_scope['score']

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_names_in_ if c not in X.columns]
            if missing:
                for c in missing:
                    X[c] = 0
            matrix = X[self.feature_names_in_].fillna(0).to_numpy(dtype=float)
        else:
            matrix = np.asarray(X, dtype=float)

        probs = [self._score_fn(row) for row in matrix]
        return np.array(probs)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

# Carrega a variável 'model' com o modelo JavaScript da raiz do projeto
model = JSChurnModel(BASE_DIR / 'modelo_churn_dinamico.js')

def run_pipeline():
    # Localização da planilha de dados na raiz do projeto
    dataset_path = BASE_DIR / 'datasets_case_modulo2_5yrs.xlsx'
    if not dataset_path.exists():
        alt_path = BASE_DIR / 'datasets_case_modulo2.xlsx'
        if alt_path.exists():
            dataset_path = alt_path

    if not dataset_path.exists():
        print(f"Planilha de dados não encontrada em '{dataset_path}'.")
        print("Para gerar novas métricas, coloque 'datasets_case_modulo2_5yrs.xlsx' ou 'datasets_case_modulo2.xlsx' na pasta raiz do projeto.")
        metrics_file = BASE_DIR / 'calculated_metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r', encoding='utf-8') as f:
                print(f"Métricas atuais em '{metrics_file}':\n", f.read())
        return

    print(f"Carregando dados de: {dataset_path}...")
    df = pd.read_excel(dataset_path)
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

    # Salva arquivo de métricas diretamente na raiz do projeto
    output_path = BASE_DIR / 'calculated_metrics.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Metrics saved to {output_path}!")
    print(metrics)

if __name__ == '__main__':
    run_pipeline()
