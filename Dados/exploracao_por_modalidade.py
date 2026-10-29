#!/usr/bin/env python3
"""
Modalidade de ensino × dependência administrativa — taxas de rendimento (INEP, 2011–2025)

Lê a base (bd5.csv ou bd5.zip) em blocos, guarda só as linhas e colunas da etapa
escolhida (padrão: Ensino Médio) e gera um relatório com:

  0. Leitura: quantas linhas, escolas-ano e escolas entraram na análise
  1. Proporção de matrículas integrais (p) por escola-ano, em faixas
  2. Modalidade × dependência por escola-ano e % de integral por ano
  3. Mudanças de modalidade por escola: quantas mudaram e de que forma
  4. Número de escolas em cada célula (dependência × modalidade)
  5. Taxas acumuladas por célula: n, Q1, mediana, Q3, % zero, % outliers (Tukey)
  6. Sensibilidade: medianas por célula em cinco versões da análise

Duas regras de modalidade são calculadas a partir das matrículas:
  Regra A (a da base, ENS_PRED): Regular p < 48%, Misto 48% ≤ p ≤ 52%, Integral p > 52%
  Regra B (com tolerância):      Regular p ≤ tol, Integral p ≥ 1 − tol, Misto no meio
onde p = matrículas integrais ÷ (regulares + integrais). As taxas usam a regra
escolhida em --regra (padrão: B).

Uso (na pasta Dados/):
    python analise_modalidade.py bd5/bd5.csv
    python analise_modalidade.py bd5.zip --saida saida_modalidade --tol 0.05 --regra B

Requer pandas e numpy; matplotlib é opcional (figuras).
Saídas em --saida: relatorio.txt, tabelas .csv (separador ;, decimal ,) e figuras .png.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TAXAS = ["apr", "rep", "abn"]
NOME_TAXA = {"apr": "Aprovação", "rep": "Reprovação", "abn": "Abandono"}
ORDEM_TAXA = ["Aprovação", "Reprovação", "Abandono"]
ORDEM_DEP = ["Federal", "Estadual/Municipal", "Privada"]
ORDEM_MOD = ["Regular", "Misto", "Integral"]
MAPA_DEP = {
    "federal": "Federal",
    "estadual": "Estadual/Municipal",
    "municipal": "Estadual/Municipal",
    "privada": "Privada",
}
ANOS_PANDEMIA = (2020, 2021)
MIN_N = 30  # células com menos escolas recebem aviso

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


# --------------------------------------------------------------------------- #
# utilidades
# --------------------------------------------------------------------------- #
def n(x):
    """Inteiro com separador de milhar brasileiro."""
    return f"{int(x):,}".replace(",", ".")


def pct(parte, total):
    return f"{100 * parte / total:.1f}%" if total else "—"


class Relatorio:
    """Imprime na tela e guarda o texto para salvar em relatorio.txt."""

    def __init__(self):
        self.linhas = []

    def __call__(self, texto=""):
        print(texto)
        self.linhas.append(str(texto))

    def secao(self, titulo):
        self("")
        self("=" * 78)
        self(titulo)
        self("=" * 78)

    def tabela(self, df):
        self(df.to_string())

    def salvar(self, caminho):
        caminho.write_text("\n".join(self.linhas) + "\n", encoding="utf-8")


def salvar_csv(df, pasta, nome, index=False):
    df.to_csv(pasta / nome, sep=";", decimal=",", encoding="utf-8-sig", index=index)


# --------------------------------------------------------------------------- #
# leitura em blocos
# --------------------------------------------------------------------------- #
def colunas_etapa(etapa):
    return {
        "mat_reg": f"QT_MAT_{etapa}_REG",
        "mat_int": f"QT_MAT_{etapa}_INT",
        "apr": f"QT_APR_{etapa}_TOTAL",
        "rep": f"QT_REP_{etapa}_TOTAL",
        "abn": f"QT_ABN_{etapa}_TOTAL",
        "pred": f"ENS_PRED_{etapa}",
    }


def ler_base(caminho, etapa, sep, encoding, chunksize):
    """Lê o CSV em blocos e mantém só escolas-ano com matrícula > 0 e rendimento informado."""
    c = colunas_etapa(etapa)
    id_cols = ["ANO", "CO_ESCOLA", "UF", "REGIAO", "LOCALIZACAO", "DEP_ADM"]
    num_cols = [c["mat_reg"], c["mat_int"], c["apr"], c["rep"], c["abn"]]

    cabecalho = pd.read_csv(
        caminho, sep=sep, encoding=encoding, nrows=0
    ).columns.str.strip()
    faltando = [x for x in id_cols + num_cols if x not in cabecalho]
    if faltando:
        sys.exit(
            f"Colunas não encontradas: {faltando}\n"
            f"Colunas lidas: {list(cabecalho)}\nConfira --sep e --encoding."
        )
    tem_pred = c["pred"] in cabecalho
    alvo = set(id_cols + num_cols + ([c["pred"]] if tem_pred else []))
    renomear = {v: k for k, v in c.items()}

    contagem = {"linhas": 0, "mat_pos": 0, "mat_pos_sem_rend": 0}
    partes = []
    leitor = pd.read_csv(
        caminho,
        sep=sep,
        encoding=encoding,
        chunksize=chunksize,
        usecols=lambda col: col.strip() in alvo,
        low_memory=False,
    )
    for i, bloco in enumerate(leitor, 1):
        bloco.columns = bloco.columns.str.strip()
        bloco = bloco.rename(columns=renomear)
        for col in ["mat_reg", "mat_int", "apr", "rep", "abn"]:
            bloco[col] = pd.to_numeric(bloco[col], errors="coerce")
        mat = bloco["mat_reg"].fillna(0) + bloco["mat_int"].fillna(0)
        tem_rend = bloco[TAXAS].notna().any(axis=1)

        contagem["linhas"] += len(bloco)
        contagem["mat_pos"] += int((mat > 0).sum())
        contagem["mat_pos_sem_rend"] += int(((mat > 0) & ~tem_rend).sum())

        b = bloco.loc[(mat > 0) & tem_rend].copy()
        for col in ["mat_reg", "mat_int", "apr", "rep", "abn"]:
            b[col] = b[col].astype("float32")
        b["ANO"] = pd.to_numeric(b["ANO"]).astype("int16")
        b["CO_ESCOLA"] = pd.to_numeric(b["CO_ESCOLA"]).astype("int64")
        partes.append(b)
        print(
            f"\r  bloco {i}: {n(contagem['linhas'])} linhas lidas, "
            f"{n(sum(len(p) for p in partes))} mantidas",
            end="",
            file=sys.stderr,
        )
    print(file=sys.stderr)

    if not partes or sum(len(p) for p in partes) == 0:
        sys.exit(f"Nenhuma escola-ano com matrícula e rendimento na etapa {etapa}.")
    df = pd.concat(partes, ignore_index=True)
    for col in ["UF", "REGIAO", "LOCALIZACAO", "DEP_ADM"] + (
        ["pred"] if tem_pred else []
    ):
        df[col] = df[col].astype("string").str.strip().astype("category")
    return df, contagem, tem_pred


def derivar(df, tol):
    df["mat"] = df["mat_reg"].fillna(0).astype("float64") + df["mat_int"].fillna(
        0
    ).astype("float64")
    df["p_int"] = df["mat_int"].fillna(0).astype("float64") / df["mat"]
    dep = df["DEP_ADM"].astype("string").str.lower().map(MAPA_DEP)
    df["dep"] = pd.Categorical(dep, categories=ORDEM_DEP)
    p = df["p_int"]
    df["mod_A"] = pd.Categorical(
        np.select([p > 0.52, p >= 0.48], ["Integral", "Misto"], "Regular"),
        categories=ORDEM_MOD,
    )
    df["mod_B"] = pd.Categorical(
        np.select([p >= 1 - tol, p > tol], ["Integral", "Misto"], "Regular"),
        categories=ORDEM_MOD,
    )
    return df


# --------------------------------------------------------------------------- #
# cálculos
# --------------------------------------------------------------------------- #
def faixas_p(p, tol):
    cortes = [
        ("p = 0%", p == 0),
        (f"0% < p ≤ {tol:.0%}", (p > 0) & (p <= tol)),
        (f"{tol:.0%} < p < 48%", (p > tol) & (p < 0.48)),
        ("48% ≤ p ≤ 52%", (p >= 0.48) & (p <= 0.52)),
        (f"52% < p < {1 - tol:.0%}", (p > 0.52) & (p < 1 - tol)),
        (f"{1 - tol:.0%} ≤ p < 100%", (p >= 1 - tol) & (p < 1)),
        ("p = 100%", p == 1),
    ]
    return pd.Series({nome: int(m.sum()) for nome, m in cortes})


def mudancas(df, col, lim_pred):
    """Uma linha por escola: anos, trocas de modalidade, padrão e classificação."""
    s = df.sort_values(["CO_ESCOLA", "ANO"])
    esc = s["CO_ESCOLA"]
    cat = s[col].astype(str)
    anterior = cat.groupby(esc).shift()
    troca = anterior.notna() & (anterior != cat)

    r = pd.DataFrame(
        {
            "n_anos": s.groupby("CO_ESCOLA")["ANO"].size(),
            "n_trocas": troca.groupby(esc).sum().astype(int),
            "primeira": cat.groupby(esc).first(),
            "ultima": cat.groupby(esc).last(),
        }
    )
    cont = pd.crosstab(esc, cat)
    r["cat_modal"] = cont.idxmax(axis=1)
    r["frac_modal"] = cont.max(axis=1) / cont.sum(axis=1)
    r["anos_integral"] = cont["Integral"] if "Integral" in cont.columns else 0
    r["mudou_dep"] = s.groupby("CO_ESCOLA")["dep"].nunique() > 1
    r["dep"] = s.groupby("CO_ESCOLA")["dep"].last()

    r["padrao"] = np.where(
        r["n_trocas"] == 0,
        "Estável: " + r["primeira"],
        np.where(
            r["n_trocas"] == 1,
            "1 troca: " + r["primeira"] + " → " + r["ultima"],
            "2 ou mais trocas",
        ),
    )
    r["mod_estavel"] = r["primeira"].where(r["n_trocas"] == 0)
    r["mod_pred"] = r["cat_modal"].where(r["frac_modal"] >= lim_pred)
    return r


def acumular(df, chaves):
    """Taxa acumulada = soma das quantidades ÷ soma das matrículas (só onde a quantidade existe)."""
    q = df[chaves + ["ANO", "mat", "mat_int"] + TAXAS].copy()
    for t in TAXAS:
        q[t] = q[t].astype("float64")
        q[f"den_{t}"] = q["mat"].where(q[t].notna())
    q["mat_int"] = q["mat_int"].fillna(0).astype("float64")
    g = q.groupby(chaves, observed=True, sort=False)
    acc = g[["mat", "mat_int"] + TAXAS + [f"den_{t}" for t in TAXAS]].sum()
    acc["n_anos"] = g["ANO"].nunique()
    acc["ano_mediano"] = g["ANO"].median()
    for t in TAXAS:
        acc[f"tx_{t}"] = acc[t] / acc[f"den_{t}"].replace(0, np.nan) * 100
    acc["mat_media"] = acc["mat"] / acc["n_anos"]
    acc["p_int_acum"] = acc["mat_int"] / acc["mat"]
    return acc.reset_index()


def caixa(s):
    s = s.dropna()
    if s.empty:
        return {
            "n": 0,
            "Q1": np.nan,
            "mediana": np.nan,
            "Q3": np.nan,
            "% zero": np.nan,
            "% outliers": np.nan,
        }
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    aiq = q3 - q1
    fora = (s < q1 - 1.5 * aiq) | (s > q3 + 1.5 * aiq)
    return {
        "n": len(s),
        "Q1": q1,
        "mediana": med,
        "Q3": q3,
        "% zero": (s == 0).mean() * 100,
        "% outliers": fora.mean() * 100,
    }


def caixas(acc, grupos):
    linhas = []
    for chave, sub in acc.groupby(grupos, observed=True):
        chave = chave if isinstance(chave, tuple) else (chave,)
        for t in TAXAS:
            linhas.append(
                {
                    **dict(zip(grupos, chave)),
                    "taxa": NOME_TAXA[t],
                    **caixa(sub[f"tx_{t}"]),
                }
            )
    out = pd.DataFrame(linhas)
    ordens = {"dep": ORDEM_DEP, "mod": ORDEM_MOD}
    for g in grupos:
        if g in ordens:
            out[g] = pd.Categorical(out[g], categories=ordens[g])
    out["taxa"] = pd.Categorical(out["taxa"], categories=ORDEM_TAXA)
    out["aviso"] = np.where(out["n"] < MIN_N, f"n < {MIN_N}", "")
    return out.sort_values(grupos + ["taxa"]).reset_index(drop=True)


def legivel(cx, grupos):
    """mediana [Q1–Q3] por célula, uma coluna por taxa."""
    cx = cx.copy()
    cx["resumo"] = [
        f"{m:5.1f} [{a:.1f}–{b:.1f}]" if k else "—"
        for m, a, b, k in zip(cx["mediana"], cx["Q1"], cx["Q3"], cx["n"])
    ]
    w = cx.pivot_table(
        index=grupos, columns="taxa", values="resumo", aggfunc="first", observed=True
    )
    w = w[[t for t in ORDEM_TAXA if t in w.columns]]
    w.insert(0, "escolas", cx.groupby(grupos, observed=True)["n"].max())
    avisos = cx.groupby(grupos, observed=True)["aviso"].max()
    w["aviso"] = avisos
    return w


def ordenar_categorias(acc, col_mod):
    acc["dep"] = pd.Categorical(acc["dep"], categories=ORDEM_DEP)
    acc[col_mod] = pd.Categorical(acc[col_mod], categories=ORDEM_MOD)
    return acc


# --------------------------------------------------------------------------- #
# figuras (opcionais)
# --------------------------------------------------------------------------- #
def figuras(pasta, faixas, acc_principal, col_mod, integral_ano, etapa, regra):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return "matplotlib não encontrado: figuras não geradas."

    cores = {"Regular": "#4C78A8", "Misto": "#F2A541", "Integral": "#54A24B"}

    # 1. faixas de p
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(faixas)), faixas.values, color="#4C78A8")
    ax.set_xticks(range(len(faixas)))
    ax.set_xticklabels(faixas.index, rotation=20, ha="right")
    ax.set_ylabel("escolas-ano")
    ax.set_title(f"{etapa}: proporção de matrículas integrais (p) por escola-ano")
    total = faixas.sum()
    for i, v in enumerate(faixas.values):
        ax.text(i, v, f"{100 * v / total:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(pasta / "fig1_faixas_proporcao_integral.png", dpi=150)
    plt.close(fig)

    # 2. boxplots cruzados (análise principal)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    larg = 0.25
    for ax, t in zip(axes, TAXAS):
        for i, dep in enumerate(ORDEM_DEP):
            for j, mod in enumerate(ORDEM_MOD):
                s = acc_principal.loc[
                    (acc_principal["dep"] == dep) & (acc_principal[col_mod] == mod),
                    f"tx_{t}",
                ].dropna()
                if s.empty:
                    continue
                pos = i + (j - 1) * larg
                bp = ax.boxplot(
                    s,
                    positions=[pos],
                    widths=larg * 0.85,
                    patch_artist=True,
                    showfliers=False,
                )
                bp["boxes"][0].set_facecolor(cores[mod])
                bp["boxes"][0].set_alpha(0.8)
                bp["medians"][0].set_color("black")
                ax.text(
                    pos,
                    1.01,
                    f"n={n(len(s))}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )
        ax.set_xticks(range(len(ORDEM_DEP)))
        ax.set_xticklabels(ORDEM_DEP, fontsize=9)
        ax.set_title(NOME_TAXA[t], pad=28)
        ax.set_ylabel("%")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(
        handles=[Patch(color=cores[m], label=m) for m in ORDEM_MOD],
        loc="lower left",
        fontsize=8,
    )
    fig.suptitle(
        f"{etapa}: taxas acumuladas por escola — escolas que não mudaram de "
        f"modalidade (regra {regra}; outliers omitidos)",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(
        pasta / "fig2_boxplots_dependencia_modalidade.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # 3. % de escolas-ano integrais por ano (contexto)
    fig, ax = plt.subplots(figsize=(9, 4))
    for dep in integral_ano.columns:
        ax.plot(integral_ano.index, integral_ano[dep], marker="o", label=dep)
    ax.set_ylabel("% das escolas-ano")
    ax.set_title(
        f"{etapa}: escolas classificadas como integrais, por ano (regra {regra})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(pasta / "fig3_integral_por_ano.png", dpi=150)
    plt.close(fig)
    return (
        "Figuras salvas: fig1_faixas_proporcao_integral.png, "
        "fig2_boxplots_dependencia_modalidade.png, fig3_integral_por_ano.png"
    )


# --------------------------------------------------------------------------- #
# principal
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("arquivo", help="bd5.csv ou bd5.zip")
    ap.add_argument("--saida", default="saida_modalidade", help="pasta de saída")
    ap.add_argument("--etapa", default="EM", choices=["AI", "AF", "EM"])
    ap.add_argument(
        "--regra",
        default="B",
        choices=["A", "B"],
        help="regra de modalidade usada nas taxas (padrão: B)",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=0.05,
        help="tolerância da regra B (padrão: 0.05 = 5%%)",
    )
    ap.add_argument(
        "--lim-pred",
        type=float,
        default=0.8,
        help="fração mínima de anos na mesma modalidade para 'predominante'",
    )
    ap.add_argument(
        "--min-mat",
        type=float,
        default=30,
        help="corte de matrícula média anual na sensibilidade",
    )
    ap.add_argument("--sep", default=";")
    ap.add_argument("--encoding", default="utf-8-sig")
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--sem-figuras", action="store_true")
    args = ap.parse_args()

    pasta = Path(args.saida)
    pasta.mkdir(parents=True, exist_ok=True)
    rel = Relatorio()
    col_mod = f"mod_{args.regra}"
    etapa = args.etapa

    rel(f"Análise de modalidade × dependência — etapa {etapa}")
    rel(f"Arquivo: {args.arquivo}")
    rel(
        f"Regra das taxas: {args.regra}  |  tolerância da regra B: {args.tol:.0%}  |  "
        f"predominância: ≥ {args.lim_pred:.0%} dos anos  |  corte de porte: "
        f"{args.min_mat:g} matrículas/ano"
    )
    rel(
        "Regra A: Regular p < 48%, Misto 48–52%, Integral p > 52%  (a mesma de ENS_PRED)"
    )
    rel(
        f"Regra B: Regular p ≤ {args.tol:.0%}, Integral p ≥ {1 - args.tol:.0%}, Misto no meio"
    )
    rel("p = matrículas integrais ÷ (regulares + integrais), por escola-ano")

    df, contagem, tem_pred = ler_base(
        Path(args.arquivo), etapa, args.sep, args.encoding, args.chunksize
    )
    df = derivar(df, args.tol)

    # ---------------------------------------------------------------- 0
    rel.secao("0. LEITURA")
    rel(f"Linhas na base: {n(contagem['linhas'])}")
    rel(f"Escolas-ano com matrícula na etapa: {n(contagem['mat_pos'])}")
    rel(f"  sem rendimento informado (descartadas): {n(contagem['mat_pos_sem_rend'])}")
    rel(f"Escolas-ano analisadas: {n(len(df))}")
    rel(f"Escolas distintas: {n(df['CO_ESCOLA'].nunique())}")
    rel(f"Anos: {df['ANO'].min()}–{df['ANO'].max()}")
    sem_dep = df["dep"].isna()
    if sem_dep.any():
        rel(
            f"ATENÇÃO: {n(sem_dep.sum())} escolas-ano com DEP_ADM não reconhecida: "
            f"{sorted(df.loc[sem_dep, 'DEP_ADM'].astype(str).unique())[:10]}"
        )
    rel("")
    rel("Escolas-ano por dependência:")
    rel.tabela(df["dep"].value_counts().reindex(ORDEM_DEP).to_frame("escolas-ano"))

    # ---------------------------------------------------------------- 1
    rel.secao("1. PROPORÇÃO DE MATRÍCULAS INTEGRAIS (p), POR ESCOLA-ANO")
    faixas = faixas_p(df["p_int"], args.tol)
    t1 = pd.DataFrame(
        {"escolas-ano": faixas, "%": (100 * faixas / faixas.sum()).round(2)}
    )
    por_dep = pd.DataFrame(
        {
            dep: faixas_p(df.loc[df["dep"] == dep, "p_int"], args.tol)
            for dep in ORDEM_DEP
        }
    )
    por_dep_pct = (100 * por_dep / por_dep.sum()).round(1)
    rel.tabela(t1)
    rel("")
    rel("% por dependência (coluna soma 100):")
    rel.tabela(por_dep_pct)
    meio = faixas.iloc[1:-1].sum()
    rel("")
    rel(
        f"Escolas-ano estritamente entre 0% e 100%: {n(meio)} ({pct(meio, faixas.sum())})"
    )
    rel(
        f"Na faixa Misto da regra A (48–52%): {n(faixas['48% ≤ p ≤ 52%'])} "
        f"({pct(faixas['48% ≤ p ≤ 52%'], faixas.sum())})"
    )
    salvar_csv(
        t1.join(por_dep_pct.add_prefix("% ")), pasta, "1_faixas_p.csv", index=True
    )

    # ---------------------------------------------------------------- 2
    rel.secao("2. MODALIDADE × DEPENDÊNCIA, POR ESCOLA-ANO")
    for regra in ["A", "B"]:
        ct = pd.crosstab(
            df["dep"], df[f"mod_{regra}"], margins=True, margins_name="Total"
        )
        rel(f"Regra {regra} (escolas-ano):")
        rel.tabela(ct)
        rel("")
        salvar_csv(ct, pasta, f"2_escola_ano_dep_mod_regra{regra}.csv", index=True)
    if tem_pred:
        concord = (df["pred"].astype(str) == df["mod_A"].astype(str)).mean()
        rel(
            f"Conferência: ENS_PRED_{etapa} igual à regra A recalculada em "
            f"{100 * concord:.2f}% das escolas-ano"
        )
        if concord < 0.99:
            rel.tabela(pd.crosstab(df["pred"].astype(str), df["mod_A"], margins=True))
        rel("")
    integral_ano = (
        df.assign(integral=df[col_mod] == "Integral")
        .groupby(["ANO", "dep"], observed=True)["integral"]
        .mean()
        .unstack()
        * 100
    ).round(1)
    integral_ano = integral_ano[[d for d in ORDEM_DEP if d in integral_ano.columns]]
    rel(f"% das escolas-ano classificadas como Integral, por ano (regra {args.regra}):")
    rel.tabela(integral_ano)
    salvar_csv(integral_ano, pasta, "2_integral_por_ano.csv", index=True)

    # ---------------------------------------------------------------- 3
    rel.secao("3. MUDANÇAS DE MODALIDADE POR ESCOLA")
    mud = {r: mudancas(df, f"mod_{r}", args.lim_pred) for r in ["A", "B"]}
    m = mud[args.regra]
    tot = len(m)
    mais1 = m["n_anos"] >= 2
    rel(f"Regra {args.regra}")
    rel(f"Escolas: {n(tot)}  (aparecem em 2 ou mais anos: {n(mais1.sum())})")
    rel(
        f"Nunca mudaram de modalidade: {n((m['n_trocas'] == 0).sum())} "
        f"({pct((m['n_trocas'] == 0).sum(), tot)})"
    )
    rel(f"  dessas, presentes em só 1 ano: {n(((m['n_trocas'] == 0) & ~mais1).sum())}")
    rel(
        f"Mudaram ao menos uma vez: {n((m['n_trocas'] > 0).sum())} "
        f"({pct((m['n_trocas'] > 0).sum(), tot)}; "
        f"{pct((m['n_trocas'] > 0).sum(), mais1.sum())} das que aparecem em 2+ anos)"
    )
    rel(f"  exatamente 1 troca: {n((m['n_trocas'] == 1).sum())}")
    rel(f"  2 ou mais trocas: {n((m['n_trocas'] >= 2).sum())}")
    rel(
        f"Com a mesma modalidade em ≥ {args.lim_pred:.0%} dos anos: "
        f"{n(m['mod_pred'].notna().sum())} ({pct(m['mod_pred'].notna().sum(), tot)})"
    )
    rel(f"Mudaram de dependência no período: {n(m['mudou_dep'].sum())}")
    rel("")
    padroes = m["padrao"].value_counts().to_frame("escolas")
    padroes["%"] = (100 * padroes["escolas"] / tot).round(1)
    rel("Padrões:")
    rel.tabela(padroes)
    salvar_csv(padroes, pasta, f"3_padroes_mudanca_regra{args.regra}.csv", index=True)
    rel("")
    dist = m["n_trocas"].clip(upper=5).value_counts().sort_index()
    dist.index = [f"{i}" if i < 5 else "5+" for i in dist.index]
    rel("Número de trocas por escola:")
    rel.tabela(dist.to_frame("escolas"))
    rel("")
    por_dep = m.groupby("dep", observed=True).agg(
        escolas=("n_anos", "size"),
        mudaram=("n_trocas", lambda s: (s > 0).sum()),
        sempre_integral=("padrao", lambda s: (s == "Estável: Integral").sum()),
        ja_foram_integrais=("anos_integral", lambda s: (s > 0).sum()),
    )
    por_dep["% mudaram"] = (100 * por_dep["mudaram"] / por_dep["escolas"]).round(1)
    rel("Por dependência (classificação do último ano):")
    rel.tabela(por_dep)
    salvar_csv(
        por_dep, pasta, f"3_mudancas_por_dependencia_regra{args.regra}.csv", index=True
    )

    rel("")
    ev = m[m["anos_integral"] > 0]
    sempre = (ev["n_trocas"] == 0).sum()
    virou = ((ev["n_trocas"] == 1) & (ev["ultima"] == "Integral")).sum()
    deixou = ((ev["n_trocas"] == 1) & (ev["primeira"] == "Integral")).sum()
    oscilou = (ev["n_trocas"] >= 2).sum()
    rel(f"Escolas que foram integrais em pelo menos um ano: {n(len(ev))}")
    rel(
        f"  integrais em todos os anos:            {n(sempre)} ({pct(sempre, len(ev))})"
    )
    rel(
        f"    dessas, presentes em 2+ anos:        "
        f"{n(((ev['n_trocas'] == 0) & (ev['n_anos'] >= 2)).sum())}"
    )
    rel(f"  viraram integrais e ficaram (1 troca): {n(virou)} ({pct(virou, len(ev))})")
    rel(
        f"  deixaram de ser integrais (1 troca):   {n(deixou)} ({pct(deixou, len(ev))})"
    )
    rel(
        f"  oscilaram (2+ trocas):                 {n(oscilou)} ({pct(oscilou, len(ev))})"
    )
    if len(ev):
        if sempre / len(ev) >= 0.5:
            rel(
                "  → Indicação: a maioria das integrais é estável; usar só escolas estáveis "
                "na análise principal perde pouco."
            )
        else:
            rel(
                "  → Indicação: a maioria das integrais mudou de modalidade; usar só escolas "
                "estáveis descarta boa parte do integral. Considerar escola × período "
                "(seção 6) como análise principal."
            )
    rel("")
    outra = "A" if args.regra == "B" else "B"
    mo = mud[outra]
    rel(
        f"Comparação com a regra {outra}: mudaram {n((mo['n_trocas'] > 0).sum())} escolas "
        f"({pct((mo['n_trocas'] > 0).sum(), len(mo))}), estáveis "
        f"{n((mo['n_trocas'] == 0).sum())}."
    )
    rel(
        "  Se a regra A gera muito mais trocas, parte delas é oscilação em torno dos 48–52%."
    )

    # ---------------------------------------------------------------- 4
    rel.secao("4. ESCOLAS POR CÉLULA (dependência × modalidade)")
    versoes_contagem = {
        f"Regra {args.regra}, só escolas que nunca mudaram": m.dropna(
            subset=["mod_estavel"]
        ),
        f"Regra {args.regra}, modalidade em ≥ {args.lim_pred:.0%} dos anos": m.dropna(
            subset=["mod_pred"]
        ),
        f"Regra {outra}, só escolas que nunca mudaram": mo.dropna(
            subset=["mod_estavel"]
        ),
    }
    col_versao = {0: "mod_estavel", 1: "mod_pred", 2: "mod_estavel"}
    for k, (titulo, sub) in enumerate(versoes_contagem.items()):
        ct = pd.crosstab(
            pd.Categorical(sub["dep"], categories=ORDEM_DEP),
            pd.Categorical(sub[col_versao[k]], categories=ORDEM_MOD),
            margins=True,
            margins_name="Total",
            dropna=False,
        )
        ct.index.name, ct.columns.name = "dependência", "modalidade"
        rel(titulo + ":")
        rel.tabela(ct)
        rel("")
        salvar_csv(ct, pasta, f"4_contagem_celulas_{k + 1}.csv", index=True)
    pequenas = []
    ct0 = pd.crosstab(m["dep"], m["mod_estavel"])
    for dep in ORDEM_DEP:
        for mod in ORDEM_MOD:
            v = ct0.loc[dep, mod] if dep in ct0.index and mod in ct0.columns else 0
            if v < MIN_N:
                pequenas.append(f"{dep} × {mod} ({v})")
    if pequenas:
        rel(
            f"Células com menos de {MIN_N} escolas (versão principal): "
            + "; ".join(pequenas)
        )

    # ---------------------------------------------------------------- 5
    rel.secao(
        "5. TAXAS ACUMULADAS POR CÉLULA — escolas que nunca mudaram de modalidade"
    )
    estaveis = m.index[m["n_trocas"] == 0]
    base_est = df[df["CO_ESCOLA"].isin(estaveis)]
    atributos = (
        df.sort_values("ANO")
        .groupby("CO_ESCOLA")[["UF", "REGIAO", "LOCALIZACAO"]]
        .last()
    )

    def por_escola(dados, escolas_mod):
        acc = acumular(dados, ["CO_ESCOLA"])
        acc = acc.merge(escolas_mod, left_on="CO_ESCOLA", right_index=True, how="inner")
        return ordenar_categorias(acc, "mod")

    principal = por_escola(
        base_est,
        m[["dep", "mod_estavel"]].rename(columns={"mod_estavel": "mod"}).dropna(),
    )
    cx_principal = caixas(principal, ["dep", "mod"])
    rel("mediana [Q1–Q3], em %. 'escolas' = n da célula.")
    rel.tabela(legivel(cx_principal, ["dep", "mod"]))
    salvar_csv(cx_principal.round(2), pasta, "5_caixas_principal.csv")

    rel("")
    rel("Só dependência:")
    cx_dep = caixas(principal, ["dep"])
    rel.tabela(legivel(cx_dep, ["dep"]))
    rel("")
    rel("Só modalidade:")
    cx_mod = caixas(principal, ["mod"])
    rel.tabela(legivel(cx_mod, ["mod"]))
    salvar_csv(
        pd.concat(
            [cx_dep.assign(recorte="dependência"), cx_mod.assign(recorte="modalidade")]
        ).round(2),
        pasta,
        "5_caixas_marginais.csv",
    )

    rel("")
    comp = principal.groupby(["dep", "mod"], observed=True).agg(
        escolas=("CO_ESCOLA", "size"),
        mat_media_mediana=("mat_media", "median"),
        anos_mediana=("n_anos", "median"),
    )
    rural = (
        principal.merge(atributos, left_on="CO_ESCOLA", right_index=True)
        .assign(rural=lambda d: d["LOCALIZACAO"].astype(str).str.lower() == "rural")
        .groupby(["dep", "mod"], observed=True)["rural"]
        .mean()
        * 100
    )
    comp["% rural"] = rural.round(1)
    comp["mat_media_mediana"] = comp["mat_media_mediana"].round(0)
    rel("Composição das células (matrícula média anual e anos na base, medianas):")
    rel.tabela(comp)
    salvar_csv(comp, pasta, "5_composicao_celulas.csv", index=True)

    # ---------------------------------------------------------------- 6
    rel.secao("6. SENSIBILIDADE — medianas por célula em cinco versões")
    versoes = {}
    versoes["1 estáveis"] = principal
    sem_pand = por_escola(
        base_est[~base_est["ANO"].isin(ANOS_PANDEMIA)],
        m[["dep", "mod_estavel"]].rename(columns={"mod_estavel": "mod"}).dropna(),
    )
    versoes["2 estáveis, sem 2020–21"] = sem_pand
    versoes[f"3 estáveis, ≥{args.min_mat:g} mat/ano"] = principal[
        principal["mat_media"] >= args.min_mat
    ]
    pred = m[["dep", "mod_pred"]].rename(columns={"mod_pred": "mod"}).dropna()
    versoes[f"4 predominante ≥{args.lim_pred:.0%}"] = por_escola(
        df[df["CO_ESCOLA"].isin(pred.index)], pred
    )

    # escola × período: os anos de cada escola em cada modalidade viram uma unidade
    ep = acumular(df.rename(columns={col_mod: "mod"}), ["CO_ESCOLA", "mod"])
    dep_ep = (
        df.rename(columns={col_mod: "mod"})
        .sort_values("ANO")
        .groupby(["CO_ESCOLA", "mod"], observed=True)["dep"]
        .last()
        .rename("dep")
    )
    ep = ep.merge(dep_ep, left_on=["CO_ESCOLA", "mod"], right_index=True)
    ep = ordenar_categorias(ep, "mod")
    versoes["5 escola × período"] = ep

    medianas, ns = {}, {}
    for nome, acc in versoes.items():
        cx = caixas(acc, ["dep", "mod"]).set_index(["dep", "mod", "taxa"])
        medianas[nome] = cx["mediana"]
        ns[nome] = cx["n"]
    tab_med = pd.DataFrame(medianas).round(2)
    tab_n = pd.DataFrame(ns).groupby(level=["dep", "mod"]).max()
    tab_med = tab_med.reindex(
        pd.MultiIndex.from_product(
            [ORDEM_DEP, ORDEM_MOD, ORDEM_TAXA], names=["dep", "mod", "taxa"]
        )
    ).dropna(how="all")
    rel("Medianas (%):")
    rel.tabela(tab_med)
    rel("")
    rel("Número de unidades por célula (escolas; na versão 5, escola × modalidade):")
    rel.tabela(
        tab_n.reindex(pd.MultiIndex.from_product([ORDEM_DEP, ORDEM_MOD]))
        .dropna(how="all")
        .astype("Int64")
    )
    salvar_csv(tab_med, pasta, "6_sensibilidade_medianas.csv", index=True)
    salvar_csv(tab_n, pasta, "6_sensibilidade_n.csv", index=True)

    rel("")
    anos_ep = (
        ep.groupby(["dep", "mod"], observed=True)["ano_mediano"].median().unstack()
    )
    rel(
        "Versão 5: ano mediano dos períodos em cada célula. Se o integral estiver concentrado"
    )
    rel(
        "em anos mais recentes, parte da diferença pode vir da queda geral do abandono no tempo."
    )
    rel.tabela(anos_ep)
    salvar_csv(anos_ep, pasta, "6_escola_periodo_ano_mediano.csv", index=True)
    ep_mista = ep.groupby("CO_ESCOLA").size()
    rel(
        f"Escolas que aparecem em mais de uma modalidade na versão 5: {n((ep_mista > 1).sum())}"
    )

    # ---------------------------------------------------------------- saídas extras
    tabela_escolas = (
        acumular(df, ["CO_ESCOLA"])
        .merge(
            m[["dep", "n_trocas", "padrao", "mod_estavel", "mod_pred", "mudou_dep"]],
            left_on="CO_ESCOLA",
            right_index=True,
        )
        .merge(atributos, left_on="CO_ESCOLA", right_index=True)
    )
    cols = [
        "CO_ESCOLA",
        "dep",
        "UF",
        "REGIAO",
        "LOCALIZACAO",
        "n_anos",
        "mat_media",
        "p_int_acum",
        "n_trocas",
        "padrao",
        "mod_estavel",
        "mod_pred",
        "mudou_dep",
        "tx_apr",
        "tx_rep",
        "tx_abn",
    ]
    salvar_csv(tabela_escolas[cols].round(4), pasta, "escolas_acumulado.csv")

    rel.secao("ARQUIVOS")
    if not args.sem_figuras:
        rel(
            figuras(
                pasta,
                faixas,
                principal.rename(columns={"mod": col_mod}),
                col_mod,
                integral_ano,
                etapa,
                args.regra,
            )
        )
    rel(
        "escolas_acumulado.csv: uma linha por escola, com taxas acumuladas e classificação."
    )
    rel(f"Tabelas e relatório em: {pasta.resolve()}")
    rel.salvar(pasta / "relatorio.txt")


if __name__ == "__main__":
    main()
