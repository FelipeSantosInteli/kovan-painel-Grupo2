/* Modelo de churn da Aula 08, rodando no navegador sobre a planilha original.

   Entrada: as linhas da aba "Dataset 1" de datasets_case_modulo2.xlsx
   (account_id, periodo, segment, country, churn_label, receita_usd, qtd_pedidos).

   As regras são as mesmas de dados/analise_aula08.py, e o teste
   tools/tests/test_painel_modelo.py compara os dois número a número:

   - histórico até 2024-02, o último mês inteiro antes do corte de 07/03/2024
     que define o churn_label (usar março vazaria o rótulo);
   - mês com compra é linha com qtd_pedidos > 0;
   - elegível é a conta com pelo menos uma compra até 2024-02;
   - regressão logística por IRLS sobre seis colunas padronizadas;
   - escore fora da amostra em 5 dobras, a conta i (na ordem do account_id)
     na dobra i % 5;
   - fila por valor esperado (escore vezes receita dos 12 meses), desempate
     pelo account_id.

   Nada aqui sai do navegador. Quem envia dado ao n8n é a tela, e só a fila. */
(function (global) {
  "use strict";

  var ULTIMO_MES = "2026-03";
  var DOBRAS = 5;
  var RIDGE = 1e-6;
  var TETO_DA_RAZAO = 10;
  var COLUNAS = [
    "meses_desde_ultima_compra",
    "meses_com_compra_12m",
    "log_receita_12m",
    "razao_receita_12m",
    "receita_12m_sobre_pico",
    "meses_de_casa"
  ];

  function mes(periodo) {
    var t = String(periodo).trim();
    return parseInt(t.slice(0, 4), 10) * 12 + parseInt(t.slice(5, 7), 10) - 1;
  }

  /* Uma linha por conta elegível, em ordem de account_id. */
  function tabela(linhas) {
    var corte = mes(ULTIMO_MES);
    var contas = {};
    linhas.forEach(function (l) {
      var id = String(l.account_id);
      var c = contas[id];
      if (!c) {
        // A base curta chamava estas colunas de segmento_lenovo e regiao.
        c = contas[id] = {
          account_id: id, segment: l.segment || l.segmento_lenovo, country: l.country || l.regiao,
          churn: 0, meses: {}, r12: 0, r24: 0, r36: 0
        };
      }
      if (Number(l.churn_label) > c.churn) c.churn = Number(l.churn_label);
      if (!(Number(l.qtd_pedidos) > 0)) return;
      var m = mes(l.periodo);
      if (m > corte) return;
      c.meses[m] = true;
      var r = Number(l.receita_usd) || 0;
      if (m > corte - 12) c.r12 += r;
      else if (m > corte - 24) c.r24 += r;
      else if (m > corte - 36) c.r36 += r;
    });
    var ids = Object.keys(contas).sort();
    var saida = [];
    ids.forEach(function (id) {
      var c = contas[id];
      var ms = Object.keys(c.meses).map(Number);
      if (!ms.length) return;
      var ultimo = Math.max.apply(null, ms), primeiro = Math.min.apply(null, ms);
      var pico = Math.max(c.r12, c.r24, c.r36);
      saida.push({
        account_id: id, segment: c.segment, country: c.country, churn: c.churn,
        receita_12m: c.r12,
        meses_desde_ultima_compra: corte - ultimo,
        meses_com_compra_12m: ms.filter(function (m) { return m > corte - 12; }).length,
        log_receita_12m: Math.log1p(Math.max(c.r12, 0)),
        razao_receita_12m: c.r24 > 0 ? Math.min(Math.max(c.r12 / c.r24, 0), TETO_DA_RAZAO) : 1,
        receita_12m_sobre_pico: pico > 0 ? Math.min(Math.max(c.r12 / pico, 0), 1) : 1,
        meses_de_casa: corte - primeiro
      });
    });
    return saida;
  }

  function resolver(A, b) {  // eliminação de Gauss com pivô parcial
    var n = b.length, M = A.map(function (l, i) { return l.concat([b[i]]); });
    for (var k = 0; k < n; k++) {
      var p = k;
      for (var i = k + 1; i < n; i++) if (Math.abs(M[i][k]) > Math.abs(M[p][k])) p = i;
      var t = M[k]; M[k] = M[p]; M[p] = t;
      for (i = k + 1; i < n; i++) {
        var f = M[i][k] / M[k][k];
        for (var j = k; j <= n; j++) M[i][j] -= f * M[k][j];
      }
    }
    var x = new Array(n);
    for (i = n - 1; i >= 0; i--) {
      var s = M[i][n];
      for (j = i + 1; j < n; j++) s -= M[i][j] * x[j];
      x[i] = s / M[i][i];
    }
    return x;
  }

  function ajustar(X, y) {
    var n = X.length, d = X[0].length, media = [], desvio = [];
    for (var j = 0; j < d; j++) {
      var s = 0; for (var i = 0; i < n; i++) s += X[i][j];
      media[j] = s / n;
      var v = 0; for (i = 0; i < n; i++) v += (X[i][j] - media[j]) * (X[i][j] - media[j]);
      desvio[j] = Math.sqrt(v / n) || 1;
    }
    var Z = X.map(function (l) { return [1].concat(l.map(function (x, j) { return (x - media[j]) / desvio[j]; })); });
    var k = d + 1, w = new Array(k).fill(0);
    for (var it = 0; it < 50; it++) {
      var H = [], g = new Array(k).fill(0);
      for (var a = 0; a < k; a++) { H.push(new Array(k).fill(0)); H[a][a] = RIDGE; g[a] = -RIDGE * w[a]; }
      for (i = 0; i < n; i++) {
        var z = Z[i], eta = 0;
        for (a = 0; a < k; a++) eta += z[a] * w[a];
        var p = 1 / (1 + Math.exp(-eta)), peso = p * (1 - p), r = y[i] - p;
        for (a = 0; a < k; a++) {
          g[a] += z[a] * r;
          for (var c = 0; c < k; c++) H[a][c] += z[a] * peso * z[c];
        }
      }
      var passo = resolver(H, g), maior = 0;
      for (a = 0; a < k; a++) { w[a] += passo[a]; maior = Math.max(maior, Math.abs(passo[a])); }
      if (maior < 1e-10) break;
    }
    return { w: w, media: media, desvio: desvio };
  }

  function prever(m, l) {
    var eta = m.w[0];
    for (var j = 0; j < l.length; j++) eta += m.w[j + 1] * (l[j] - m.media[j]) / m.desvio[j];
    return 1 / (1 + Math.exp(-eta));
  }

  function auc(escore, y) {
    var idx = escore.map(function (_, i) { return i; }).sort(function (a, b) { return escore[a] - escore[b]; });
    var rank = new Array(escore.length), i = 0;
    while (i < idx.length) {
      var j = i;
      while (j + 1 < idx.length && escore[idx[j + 1]] === escore[idx[i]]) j++;
      for (var k = i; k <= j; k++) rank[idx[k]] = (i + j) / 2 + 1;
      i = j + 1;
    }
    var n1 = 0, soma = 0;
    y.forEach(function (v, i) { if (v === 1) { n1++; soma += rank[i]; } });
    var n0 = y.length - n1;
    return (soma - n1 * (n1 + 1) / 2) / (n1 * n0);
  }

  function rodar(linhas, capacidade) {
    var menor = linhas.reduce(function (m, l) { var p = String(l.periodo); return p < m ? p : m; }, "9999");
    if (menor > ULTIMO_MES) {
      throw new Error("esta planilha começa em " + menor + " e não tem histórico antes do corte de 31/03/2026. " +
        "Use a base longa do case, datasets_case_modulo2_5yrs.xlsx de cerca de 69,5 MB");
    }
    var t = tabela(linhas);
    if (t.length < 50) throw new Error("A planilha tem " + t.length + " contas elegíveis. Confira se é a aba Dataset 1 da base do case.");
    var X = t.map(function (c) { return COLUNAS.map(function (n) { return c[n]; }); });
    var y = t.map(function (c) { return c.churn; });
    var escore = new Array(t.length);
    for (var k = 0; k < DOBRAS; k++) {
      var Xt = [], yt = [];
      for (var i = 0; i < t.length; i++) if (i % DOBRAS !== k) { Xt.push(X[i]); yt.push(y[i]); }
      var m = ajustar(Xt, yt);
      for (i = k; i < t.length; i += DOBRAS) escore[i] = prever(m, X[i]);
    }
    var final = ajustar(X, y);
    t.forEach(function (c, i) {
      c.escore = escore[i];
      c.valor_em_risco = Math.max(c.receita_12m, 0);
      c.valor_esperado = c.escore * c.valor_em_risco;
    });
    var ordem = t.slice().sort(function (a, b) {
      return b.valor_esperado - a.valor_esperado || (a.account_id < b.account_id ? -1 : 1);
    });
    ordem.forEach(function (c, i) { c.posicao = i + 1; });
    var fila = ordem.slice(0, capacidade);
    var pesos = {};
    COLUNAS.forEach(function (n, j) { pesos[n] = final.w[j + 1]; });
    return {
      contas: t.length,
      perdidas: y.reduce(function (a, b) { return a + b; }, 0),
      auc: auc(escore, y),
      pesos: pesos,
      fila: fila,
      acertos_fila: fila.reduce(function (a, c) { return a + c.churn; }, 0),
      valor_esperado_fila: fila.reduce(function (a, c) { return a + c.valor_esperado; }, 0)
    };
  }

  /* O pacote que segue para um modelo de linguagem: a fila, com nomes de
     coluna que se explicam sozinhos, e o resumo do modelo. Nenhuma linha de
     pedido. O nome receita_12m_como_fracao_do_pico_anual existe porque o
     nome anterior, queda_contra_pico, fez dois modelos lerem 0,63 como queda
     de 63%. */
  function contexto(r) {
    return {
      modelo: {
        contas_elegiveis: r.contas, perdidas: r.perdidas, auc_fora_da_amostra: +r.auc.toFixed(4),
        historico_ate: ULTIMO_MES, capacidade_do_ciclo: r.fila.length
      },
      contas: r.fila.map(function (l) {
        return {
          posicao_na_fila: l.posicao, account_id: l.account_id, segmento: l.segment, pais: l.country,
          escore_de_perda: +l.escore.toFixed(3),
          receita_12m_usd: Math.round(l.valor_em_risco), valor_esperado_usd: Math.round(l.valor_esperado),
          meses_desde_ultima_compra: l.meses_desde_ultima_compra,
          meses_com_compra_nos_ultimos_12: l.meses_com_compra_12m,
          receita_12m_dividida_pela_dos_12m_anteriores: +l.razao_receita_12m.toFixed(2),
          receita_12m_como_fracao_do_pico_anual: +l.receita_12m_sobre_pico.toFixed(2)
        };
      })
    };
  }

  /* A fila calculada fica neste navegador para os experimentos do
     laboratório não pedirem a planilha de novo. Só a fila, nunca a planilha. */
  var CHAVE = "kovan-contexto";
  function guardar(ctx) { try { localStorage.setItem(CHAVE, JSON.stringify(ctx)); } catch (e) { /* sem armazenamento */ } }
  function recuperar() { try { return JSON.parse(localStorage.getItem(CHAVE) || "null"); } catch (e) { return null; } }

  global.KovanModelo = { contexto: contexto, guardar: guardar, recuperar: recuperar, COLUNAS: COLUNAS, ULTIMO_MES: ULTIMO_MES, tabela: tabela, rodar: rodar, auc: auc };
})(window);
