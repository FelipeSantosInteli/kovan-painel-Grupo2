# frontend: o painel do grupo

Uma página só, sem build e sem instalação. Ela lê a planilha original do case
(`datasets_case_modulo2.xlsx`, a base longa de cerca de 68 MB), treina uma
regressão logística dentro do navegador, monta a fila do ciclo e conversa com
uma equipe de três agentes no n8n:

- **Atlas**, analista da fila, responde com os números;
- **Vera**, estrategista, recomenda a ação pelos planos de cada área;
- **Ciro**, revisor, confere cada número e dá o veredito.

O guia completo, passo a passo e com os prompts do Antigravity, está em
<https://josercf.github.io/inteli-2026-2-pos-m02/materiais/aula08-guia.html>.

## Arquivos

| Arquivo | Papel |
|---|---|
| `index.html` | a tela |
| `modelo.js` | o modelo: colunas por conta, regressão logística, fila |
| `inteli-brand.css` | as cores e fontes da marca |
| `workflow_n8n.json` | a equipe de agentes, para importar no n8n |

Esta pasta é gerada a partir de `painel/` do acervo da disciplina, por
`tools/exportar_frontend_aula08.py`. Modifique à vontade na cópia do grupo.

## Rodar na sua máquina

```bash
cd frontend
python3 -m http.server 8000      # abra http://localhost:8000
```

## O contrato com o n8n

Qualquer versão do painel precisa manter o formato que vai e volta, porque os
três agentes dependem dele:

```json
POST { "sessionId": "...", "pergunta": "...", "modelo_llm": "nvidia/nemotron-3-super-120b-a12b:free",
       "modelo": { "contas_elegiveis": 4593, "auc_fora_da_amostra": 0.8138, ... },
       "contas": [ { "posicao_na_fila": 1, "account_id": "...", "escore_de_perda": 0.347, ... } ],
       "planos": { "comercial": "...", "atendimento": "...", "pós-vendas": "..." } }

200  { "modelo": "...", "agentes": [ { "nome": "Atlas", "papel": "...", "texto": "..." }, ... ] }
```

## O que nunca entra no repositório do grupo

A planilha, a chave do OpenRouter e notas de avaliação. O painel lê a planilha
dentro do navegador e envia ao n8n só a fila de 138 contas, com identificador
anonimizado.
