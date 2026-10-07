function processarChurnDinamico(linhas) {
  var maxDate = new Date(-8640000000000000);
  linhas.forEach(function (l) {
    if (l.periodo) {
      var d = new Date(l.periodo);
      if (d > maxDate) maxDate = d;
    }
  });

  var accounts = {};
  linhas.forEach(function (l) {
    var id = String(l.account_id);
    if (!accounts[id]) {
      accounts[id] = {
        account_id: id,
        segment: l.segment || l.segmento_lenovo || "UNKNOWN",
        receita_usd: 0,
        qtd_pedidos: 0,
        primeira_compra: new Date(8640000000000000),
        ultima_compra: new Date(-8640000000000000),
        revenue_6m: 0,
        dates: [],
        churn_label: 0
      };
    }
    var c = accounts[id];
    var r = Number(l.receita_usd) || 0;
    var q = Number(l.qtd_pedidos) || 0;
    c.receita_usd += r;
    c.qtd_pedidos += q;
    if (l.churn_label) c.churn_label = Math.max(c.churn_label, Number(l.churn_label));
    if (l.periodo) {
      var d = new Date(l.periodo);
      if (d < c.primeira_compra) c.primeira_compra = d;
      if (d > c.ultima_compra) c.ultima_compra = d;
      var diffDays = (maxDate - d) / (1000 * 3600 * 24);
      if (diffDays <= 180) c.revenue_6m += r;
      c.dates.push(d);
    }
  });

  var arr = Object.keys(accounts).map(function(k) { return accounts[k]; });
  var recs = arr.map(function(c) { return c.receita_usd; }).sort(function(a, b) { return a - b; });
  var freqs = arr.map(function(c) { return c.qtd_pedidos; }).sort(function(a, b) { return a - b; });
  var med_rec = recs[Math.floor(recs.length / 2)] || 0;
  var med_freq = freqs[Math.floor(freqs.length / 2)] || 0;

  arr.forEach(function(c) {
    if (c.receita_usd >= med_rec && c.qtd_pedidos >= med_freq) c.persona = 'P1';
    else if (c.receita_usd >= med_rec && c.qtd_pedidos < med_freq) c.persona = 'P2';
    else if (c.receita_usd < med_rec && c.qtd_pedidos >= med_freq) c.persona = 'P3';
    else c.persona = 'P4';
  });

  var limites = { P1: 2028, P2: 858, P3: 892, P4: 1993 };
  var contagens = { P1: 0, P2: 0, P3: 0, P4: 0 };
  
  arr.sort(function(a, b) { return b.receita_usd - a.receita_usd; });

  var filteredArr = [];
  arr.forEach(function(c) {
    if (contagens[c.persona] < limites[c.persona]) {
      contagens[c.persona]++;
      filteredArr.push(c);
    }
  });

  var risco = { P1: 0, P2: 0, P3: 0, P4: 0, ARR: 0 };
  var riscoLista = { P1: [], P2: [], P3: [], P4: [] };
  var chartData = {};

  filteredArr.forEach(function (c) {
    c.dates.sort(function(a, b) { return a - b; });
    var uniqueDates = [];
    c.dates.forEach(function(d) {
      if (uniqueDates.length === 0 || uniqueDates[uniqueDates.length - 1].getTime() !== d.getTime()) {
        uniqueDates.push(d);
      }
    });

    var sumDiffs = 0, diffs = [];
    for (var i = 1; i < uniqueDates.length; i++) {
      var dff = (uniqueDates[i] - uniqueDates[i - 1]) / (1000 * 3600 * 24);
      diffs.push(dff);
      sumDiffs += dff;
    }
    var avgDays = diffs.length > 0 ? sumDiffs / diffs.length : 0;
    var stdDays = 0;
    if (diffs.length > 0) {
      var sqSum = 0;
      for (var i = 0; i < diffs.length; i++) sqSum += Math.pow(diffs[i] - avgDays, 2);
      stdDays = Math.sqrt(sqSum / diffs.length);
    }

    var recency = (maxDate - c.ultima_compra) / (1000 * 3600 * 24);
    var tenure = (maxDate - c.primeira_compra) / (1000 * 3600 * 24);
    if (tenure < 1) tenure = 1;
    var receitaAnualizada = c.receita_usd / Math.max(1, tenure / 365.25);

    var input = new Array(34).fill(0);
    input[0] = c.qtd_pedidos;
    input[1] = c.receita_usd;
    input[2] = recency;
    input[3] = avgDays;
    input[4] = stdDays;
    input[5] = c.revenue_6m;
    input[6] = tenure;
    
    // contatos, etc = 0
    input[7] = 0; input[8] = 0; input[9] = 0; input[10] = 0; input[11] = 0;
    input[12] = receitaAnualizada;

    var segStr = String(c.segment).toUpperCase();
    if (segStr === "LARGE ENTERPRISE") input[13] = 1;
    else if (segStr === "MID MARKET") input[14] = 1;
    else if (segStr === "PUBLIC SECTOR") input[15] = 1;
    else if (segStr === "SMALL MARKET") input[16] = 1;
    else if (segStr === "STRATEGIC ACCOUNT") input[17] = 1;

    // industry defaults to unknown -> input[29]
    input[29] = 1;

    if (c.persona === "P2") input[31] = 1;
    else if (c.persona === "P3") input[32] = 1;
    else if (c.persona === "P4") input[33] = 1;

    if (typeof score === "function") {
      var preds = score(input);
      var prob_churn = preds[1];
      
      var isRisk = false;
      if (prob_churn >= 0.50) {
        if (c.persona === "P1" && recency > 30) isRisk = true;
        else if (c.persona === "P2" && recency > 150) isRisk = true;
        else if (c.persona === "P3" && recency > 90) isRisk = true;
        else if (c.persona === "P4" && recency > 180) isRisk = true;
      }
      
      if (isRisk) {
        risco[c.persona]++;
        risco.ARR += receitaAnualizada;
        
        riscoLista[c.persona].push({
          account_id: c.account_id,
          persona: c.persona,
          segment: c.segment,
          industry: c.industry || "UNKNOWN",
          probabilidade_churn: prob_churn,
          dias_sem_comprar: recency,
          receita_anualizada: receitaAnualizada
        });
      }
    }

    if (c.churn_label === 1 && c.ultima_compra) {
      var yr = c.ultima_compra.getFullYear();
      var mo = c.ultima_compra.getMonth() + 1;
      var key = yr + "-" + (mo < 10 ? "0" + mo : mo);
      if (!chartData[key]) chartData[key] = { total: 0, P1: 0, P2: 0, P3: 0, P4: 0 };
      chartData[key].total++;
      chartData[key][c.persona]++;
    }
    
    risco.TotalARR = (risco.TotalARR || 0) + receitaAnualizada;
  });

  if (risco.TotalARR > 0) {
    risco.NRR = ((risco.TotalARR - risco.ARR) / risco.TotalARR) * 100;
  } else {
    risco.NRR = 0;
  }

  return { risco: risco, chartData: chartData, riscoLista: riscoLista };
}
