# -*- coding: utf-8 -*-
"""Saída de emergência: os números das Aulas 03 e 04, lidos do dataset oficial.

Este script existe para que uma mesa com o ambiente travado continue a aula. Ele
não substitui a análise do grupo: o artefato é o código que vocês escreverem.

O arquivo `datasets_case_modulo2.xlsx` não está neste repositório. Coloque-o em
`dados/` antes de executar.

Uso: python analise_referencia.py
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parent
XLSX = RAIZ / "dados" / "datasets_case_modulo2.xlsx"

# O painel termina em 2026-03. A recência de uma conta é contada a partir daí,
# e é essa âncora que torna comparáveis os cortes de inatividade.
FIM_DO_PAINEL = "2026-03"

ABAS = {
    "painel": "Dataset 1",
    "mix": "Dataset 2",
    "engajamento": "Dataset 3",
    "cadastro": "Dataset 4",
    "pedidos": "raw data",
}


class DatasetAusente(FileNotFoundError):
    """O xlsx oficial não está em dados/. Não é erro de código."""


@lru_cache(maxsize=1)
def carregar() -> dict[str, pd.DataFrame]:
    """Lê as cinco abas. Nenhuma limpeza acontece aqui: o dado cru é o material
    da aula, e limpar antes esconderia as advertências que a turma precisa achar."""
    if not XLSX.exists():
        raise DatasetAusente(
            f"{XLSX} não encontrado. O dataset oficial não é versionado (ADR-005): "
            "peça o arquivo pelo canal da turma e coloque-o em dados/."
        )
    xl = pd.ExcelFile(XLSX)
    return {nome: xl.parse(aba) for nome, aba in ABAS.items()}


def _meses_desde(periodo: str, referencia: str = FIM_DO_PAINEL) -> int:
    """Distância em meses entre dois rótulos 'YYYY-MM'."""
    ay, am = (int(p) for p in periodo.split("-"))
    by, bm = (int(p) for p in referencia.split("-"))
    return (by - ay) * 12 + (bm - am)


@lru_cache(maxsize=1)
def contas() -> pd.DataFrame:
    """Uma linha por conta: o grão em que o rótulo vive.

    O `churn_label` do painel é constante dentro da conta, então agregar por
    `max` não perde informação. Conferido em `perfil_do_rotulo`.
    """
    d = carregar()
    painel, cadastro = d["painel"], d["cadastro"]
    c = painel.groupby("account_id").agg(
        churn=("churn_label", "max"),
        segmento=("segmento_lenovo", "first"),
        regiao=("regiao", "first"),
        receita=("receita_usd", "sum"),
        pedidos=("qtd_pedidos", "sum"),
        meses_ativos=("periodo", "nunique"),
        ultimo_mes=("periodo", "max"),
        primeiro_mes=("periodo", "min"),
    )
    c["inatividade"] = c.ultimo_mes.map(_meses_desde)
    # O cadastro tem account_id repetido com setor e região divergentes: a
    # primeira linha é uma escolha declarada, não um fato do dado.
    cad = cadastro.drop_duplicates("account_id").set_index("account_id")
    return c.join(cad[["setor", "tempo_como_cliente", "porte"]])


# ---------------------------------------------------------------------------
# Perfil e qualidade
# ---------------------------------------------------------------------------

def formato_das_abas() -> dict[str, tuple[int, int]]:
    return {nome: df.shape for nome, df in carregar().items()}


def qualidade() -> dict[str, int | float]:
    d = carregar()
    painel, mix, engaj, cad = d["painel"], d["mix"], d["engajamento"], d["cadastro"]
    contas_painel = set(painel.account_id)
    return {
        "contas": painel.account_id.nunique(),
        "meses": painel.periodo.nunique(),
        "primeiro_mes": painel.periodo.min(),
        "ultimo_mes": painel.periodo.max(),
        "receita_negativa": int((painel.receita_usd < 0).sum()),
        "receita_zero": int((painel.receita_usd == 0).sum()),
        "cadastro_duplicado": int(cad.account_id.duplicated().sum()),
        "mix_pct_nulo": int(mix.pct_receita.isna().sum()),
        "engaj_contas": engaj.account_id.nunique(),
        "engaj_fora_do_painel": len(set(engaj.account_id) - contas_painel),
        "engaj_dentro_do_painel": len(set(engaj.account_id) & contas_painel),
        "engaj_cobertura": len(set(engaj.account_id) & contas_painel) / len(contas_painel),
        "engaj_periodos": engaj.periodo.nunique(),
    }


# ---------------------------------------------------------------------------
# Univariada
# ---------------------------------------------------------------------------

def receita_univariada() -> dict[str, float]:
    r = contas().receita
    return {
        "media": float(r.mean()),
        "mediana": float(r.median()),
        "desvio": float(r.std()),
        "minimo": float(r.min()),
        "maximo": float(r.max()),
        "assimetria": float(r.skew()),
        "curtose": float(r.kurtosis()),
        "assimetria_log10": float(np.log10(r.clip(lower=1)).skew()),
    }


def pareto(fracoes=(0.01, 0.05, 0.10, 0.20)) -> dict[float, float]:
    """Participação na receita das contas do topo. A cauda é o assunto do dia:
    com assimetria de 44, a média por conta não descreve conta nenhuma."""
    r = contas().receita.sort_values(ascending=False)
    total = r.sum()
    return {f: float(r.head(int(len(r) * f)).sum() / total) for f in fracoes}


# ---------------------------------------------------------------------------
# Bivariada
# ---------------------------------------------------------------------------

def prevalencia_por(coluna: str) -> pd.DataFrame:
    c = contas()
    t = c.groupby(coluna).agg(contas=("churn", "size"), perdidas=("churn", "sum"),
                              receita=("receita", "sum"))
    t["prevalencia"] = t.perdidas / t.contas
    t["participacao_receita"] = t.receita / t.receita.sum()
    return t.sort_values("prevalencia", ascending=False)


def contingencia_rotulo() -> pd.DataFrame:
    """Último mês com receita contra o rótulo. É a tabela que entrega o corte."""
    c = contas()
    return pd.crosstab(c.ultimo_mes, c.churn)


def perfil_do_rotulo() -> dict[str, object]:
    d = carregar()
    por_conta = d["painel"].groupby("account_id").churn_label.nunique()
    c = contas()
    perdidas, ativas = c[c.churn == 1], c[c.churn == 0]
    return {
        "rotulo_constante_na_conta": bool((por_conta == 1).all()),
        "contas_perdidas": int(c.churn.sum()),
        "prevalencia": float(c.churn.mean()),
        "ultimo_mes_maximo_das_perdidas": perdidas.ultimo_mes.max(),
        "ultimo_mes_minimo_das_ativas": ativas.ultimo_mes.min(),
        "inatividade_minima_das_perdidas": int(perdidas.inatividade.min()),
        "inatividade_maxima_das_ativas": int(ativas.inatividade.max()),
        "mes_de_sobreposicao": "2025-02",
    }


def limiar_de_inatividade(cortes=(6, 9, 12, 13)) -> dict[int, dict[str, float]]:
    """O que cada corte de inatividade marcaria, contra o rótulo entregue.

    É o exercício do dia: o rótulo oficial não é um fato do sistema, é um corte
    que alguém escolheu. Trocar o corte troca o tamanho da fila e troca quantas
    das contas hoje marcadas continuam marcadas.
    """
    c = contas()
    oficial = c.churn == 1
    saida = {}
    for corte in cortes:
        marcado = c.inatividade >= corte
        saida[corte] = {
            "fila": int(marcado.sum()),
            "captura_do_rotulo_oficial": float((marcado & oficial).sum() / oficial.sum()),
            "receita_da_fila": float(c.receita[marcado].sum()),
        }
    return saida


def corte_diario_do_rotulo() -> dict[str, object]:
    """A data exata em que o rótulo separa, recuperada da aba de pedidos.

    O painel mensal leva a turma até "inatividade de 13 meses" e para ali, com
    382 contas que o corte marca e o rótulo não. A separação é perfeita no dia:
    a regra foi construída sobre a data do pedido, que o agregado mensal não
    expõe. É o argumento de por que a aba `raw data` entra na aula.
    """
    d = carregar()
    ultima = d["pedidos"].groupby("account_id")["Order Date"].max()
    c = contas().join(ultima.rename("ultima_compra"))
    perdidas, ativas = c[c.churn == 1], c[c.churn == 0]
    limite_perdidas = perdidas.ultima_compra.max()
    limite_ativas = ativas.ultima_compra.min()
    return {
        "ultima_compra_maxima_das_perdidas": limite_perdidas.strftime("%Y-%m-%d"),
        "ultima_compra_minima_das_ativas": limite_ativas.strftime("%Y-%m-%d"),
        "separacao_perfeita": bool(limite_perdidas < limite_ativas),
        "contas_no_mes_de_sobreposicao": int((c.ultimo_mes == "2025-02").sum()),
        "sobra_do_corte_mensal": int(((c.inatividade >= 13) & (c.churn == 0)).sum()),
    }


def tempo_de_casa_por_rotulo() -> dict[str, int | float]:
    c = contas()
    novas = c[c.tempo_como_cliente < 17]
    return {
        "piso_de_tempo_das_perdidas": int(c.tempo_como_cliente[c.churn == 1].min()),
        "contas_com_menos_de_17_meses": int(len(novas)),
        "perdidas_entre_elas": int(novas.churn.sum()),
    }


# ---------------------------------------------------------------------------
# Aula 04: elegibilidade e testes de hipótese
# ---------------------------------------------------------------------------

# Última primeira-compra que ainda deixa treze meses de painel pela frente.
ULTIMO_MES_ELEGIVEL = "2025-02"


def _wilson(k: int, n: int) -> tuple[float, float, float]:
    lo, hi = stats.binomtest(int(k), int(n)).proportion_ci(method="wilson")
    return k / n, float(lo), float(hi)


@lru_cache(maxsize=1)
def contas_enriquecidas() -> pd.DataFrame:
    d = carregar()
    c = contas().copy()
    ped = d["pedidos"].copy()
    ped["Order Date"] = pd.to_datetime(ped["Order Date"])
    por_conta = ped.sort_values("Order Date").groupby("account_id")["Order Date"]
    c["dias_de_compra"] = por_conta.nunique()
    c["intervalo_mediano"] = por_conta.apply(
        lambda s: s.drop_duplicates().diff().dt.days.median())
    mix = d["mix"]
    c["marcas"] = mix.groupby("account_id").Brand.nunique()
    c["marca_dominante"] = (mix.sort_values("pct_receita", ascending=False)
                            .drop_duplicates("account_id").set_index("account_id").Brand)
    c["elegivel"] = c.primeiro_mes <= ULTIMO_MES_ELEGIVEL
    c["faixa_marcas"] = c.marcas.clip(upper=4).astype(int).astype(str).replace("4", "4+")
    c["faixa_dias"] = pd.cut(c.dias_de_compra, [0, 1, 2, 5, 10**6],
                             labels=["1", "2", "3 a 5", "6+"]).astype(str)
    return c


def elegiveis() -> pd.DataFrame:
    c = contas_enriquecidas()
    return c[c.elegivel]


def populacoes() -> dict[str, dict]:
    c = contas_enriquecidas()
    saida = {}
    for nome, mascara in (("carteira", np.ones(len(c), dtype=bool)),
                          ("elegiveis", c.elegivel.to_numpy()),
                          ("nao_elegiveis", (~c.elegivel).to_numpy())):
        s = c[mascara]
        prev, lo, hi = _wilson(int(s.churn.sum()), len(s))
        saida[nome] = {"contas": len(s), "perdidas": int(s.churn.sum()),
                       "prevalencia": prev, "ic_inferior": lo, "ic_superior": hi}
    return saida


def perfil_por_segmento() -> pd.DataFrame:
    e = elegiveis()
    t = e.groupby("segmento").agg(
        contas=("churn", "size"), perdidas=("churn", "sum"), prevalencia=("churn", "mean"),
        receita_mediana=("receita", "median"), dias_mediano=("dias_de_compra", "median"),
        marcas_mediana=("marcas", "median"), intervalo_mediano=("intervalo_mediano", "median"),
        marca_dominante=("marca_dominante", lambda s: s.mode().iloc[0]),
    )
    return t.sort_values("prevalencia", ascending=False)


def tabela_com_ic(coluna: str, df: pd.DataFrame | None = None):
    """Contingência com IC de Wilson por categoria e qui-quadrado da tabela."""
    df = elegiveis() if df is None else df
    t = df.groupby(coluna, observed=True).churn.agg(contas="size", perdidas="sum")
    ics = [_wilson(int(k), int(n)) for n, k in zip(t.contas, t.perdidas)]
    t["prevalencia"] = [p for p, _, _ in ics]
    t["ic_inferior"] = [lo for _, lo, _ in ics]
    t["ic_superior"] = [hi for _, _, hi in ics]
    qui2, p, gl, _ = stats.chi2_contingency(pd.crosstab(df[coluna], df.churn))
    return t, float(qui2), int(gl), float(p)


def estratificar(coluna_teste: str, valor_a, valor_b, estrato: str) -> pd.DataFrame:
    """Prevalência do grupo A contra o grupo B dentro de cada estrato.

    Para `marcas`, `valor_b` é um piso: 3 significa três ou mais marcas. Para
    `regiao`, `valor_b == "outros"` significa todas as regiões exceto `valor_a`.
    Estrato com menos de 30 contas em qualquer dos grupos fica de fora.
    """
    e = elegiveis()
    if coluna_teste == "marcas":
        grupo_a, grupo_b = e.marcas == valor_a, e.marcas >= valor_b
    elif coluna_teste == "regiao" and valor_b == "outros":
        grupo_a, grupo_b = e.regiao == valor_a, e.regiao != valor_a
    else:
        grupo_a, grupo_b = e[coluna_teste] == valor_a, e[coluna_teste] == valor_b
    linhas = {}
    for nome, g in e.groupby(estrato, observed=True):
        a, b = g[grupo_a.loc[g.index]], g[grupo_b.loc[g.index]]
        if len(a) < 30 or len(b) < 30:
            continue
        pa, loa, hia = _wilson(int(a.churn.sum()), len(a))
        pb, lob, hib = _wilson(int(b.churn.sum()), len(b))
        linhas[nome] = {"n_a": len(a), "prev_a": pa, "lo_a": loa, "hi_a": hia,
                        "n_b": len(b), "prev_b": pb, "lo_b": lob, "hi_b": hib}
    return pd.DataFrame.from_dict(linhas, orient="index")


def aula04() -> None:
    print("\n=== Aula 04: populações ===")
    for nome, v in populacoes().items():
        print(f"  {nome:14} {v['contas']:5} contas  {v['perdidas']:5} perdidas  "
              f"{v['prevalencia']:.3f} [{v['ic_inferior']:.3f}, {v['ic_superior']:.3f}]")
    print("\n=== Aula 04: perfil por segmento (elegíveis) ===")
    print(perfil_por_segmento().round(3).to_string())
    for col in ("faixa_marcas", "faixa_dias"):
        t, q, gl, p = tabela_com_ic(col)
        print(f"\n=== Aula 04: {col}: qui2={q:.1f} gl={gl} p={p:.1e} ===")
        print(t.round(3).to_string())
    print("\n=== Aula 04: marcas 1 contra 3+, por faixa de dias de compra ===")
    print(estratificar("marcas", 1, 3, "faixa_dias").round(3).to_string())


def main() -> None:
    q, r, p = qualidade(), receita_univariada(), perfil_do_rotulo()
    print("=== formato das abas ===")
    for nome, (linhas, colunas) in formato_das_abas().items():
        print(f"  {nome:12} {linhas:>7} linhas x {colunas:>2} colunas")
    print("\n=== qualidade ===")
    for k, v in q.items():
        print(f"  {k:26} {v}")
    print("\n=== receita por conta ===")
    for k, v in r.items():
        print(f"  {k:18} {v:,.4f}")
    print("\n=== pareto ===")
    for f, v in pareto().items():
        print(f"  top {f:.0%}: {v:.4%}")
    print("\n=== rotulo ===")
    for k, v in p.items():
        print(f"  {k:34} {v}")
    print("\n=== tempo de casa ===")
    for k, v in tempo_de_casa_por_rotulo().items():
        print(f"  {k:30} {v}")
    print("\n=== prevalencia por segmento ===")
    print(prevalencia_por("segmento").to_string())
    print("\n=== prevalencia por setor ===")
    print(prevalencia_por("setor").to_string())
    print("\n=== contingencia ultimo mes x rotulo ===")
    print(contingencia_rotulo().to_string())
    print("\n=== corte diario ===")
    for k, v in corte_diario_do_rotulo().items():
        print(f"  {k:36} {v}")
    print("\n=== cortes de inatividade ===")
    for corte, v in limiar_de_inatividade().items():
        print(f"  >= {corte:>2} meses: fila {v['fila']:>5}  "
              f"captura {v['captura_do_rotulo_oficial']:.4%}  "
              f"receita {v['receita_da_fila']:,.0f}")
    aula04()


if __name__ == "__main__":
    main()
