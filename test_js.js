const fs = require('fs');
const xlsx = require('xlsx');

// Mock score function if not defined
// Wait, we need to load both scripts
const script1 = fs.readFileSync('modelo_churn_dinamico.js', 'utf8');
const script2 = fs.readFileSync('calcular_churn_dinamico.js', 'utf8');

eval(script1);
eval(script2);

const arquivo = fs.existsSync('./datasets_case_modulo2_5yrs.xlsx')
  ? './datasets_case_modulo2_5yrs.xlsx'
  : (fs.existsSync('./datasets_case_modulo2.xlsx') ? './datasets_case_modulo2.xlsx' : './datasets_case_modulo2_5yrs.xlsx');

const wb = xlsx.readFile(arquivo);
const sheet = wb.Sheets[wb.SheetNames[0]];
const linhas = xlsx.utils.sheet_to_json(sheet);

const result = processarChurnDinamico(linhas);
console.log(result);
