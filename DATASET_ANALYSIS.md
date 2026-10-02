# Importação e compreensão das bases

`POST /analyze-dataset` analisa sem salvar; `POST /datasets` salva o histórico
para treino. Ambos recebem `file` multipart. O primeiro também aceita `target`.
Depois de salvar, `GET /datasets/{id}?target=coluna` recalcula as sugestões;
`/preview` mostra cinco registros com identificadores ocultos e `/values?column=`
retorna classes ou um resumo numérico para definição da faixa de atenção.

## Leitura

O formato é identificado pelo conteúdo: CSV/texto, XLS binário ou XLSX compactado.
Uma planilha XLS com nome `.csv` funciona e produz aviso. Reconhece UTF-8/BOM,
CP1252, Latin-1 e UTF-16; delimitadores vírgula, ponto e vírgula, tab e pipe.
Reconhece a diretiva `sep=` usada pelo Excel. IDs textuais preservam zeros iniciais.

Cabeçalhos vazios/repetidos recebem nomes únicos e avisos. Uma linha de códigos
X1..Xn/Y seguida por nomes reais (caso da base de crédito do projeto) é reconhecida.
Primeira aba Excel não vazia é usada. Linhas físicas vazias e colunas anônimas
totalmente vazias são ignoradas. Linhas com quantidade inconsistente de campos
produzem erro com o número da linha; não são descartadas silenciosamente.
Correções aparecem em `import_info` e `import_warnings` e na primeira tela do assistente.

Números com vírgula decimal e com separadores de milhar mistos são interpretados
sem extrair dígitos de códigos. Valores ambíguos com separadores exigem conferir a
prévia. Estatísticas de IDs e amostras pessoais não são enviadas a serviços externos.

## Semântica e qualidade

`semantic_analyzer.py` usa tokens, acentos normalizados e camelCase; o dicionário
é extensível. `status` em `marital-status` não é resultado. `Target`, `y`, classes
de renda `<=50K`/`>50K` e resultados acadêmicos são reconhecidos. Renda numérica
continua sendo um atributo financeiro/sensível até a pessoa escolher usá-la como
resultado com uma faixa explícita. Confiança é heurística, não probabilidade.

IDs são detectados por nome ou unicidade >95% em >=20 linhas com formato de
e-mail, UUID ou código. Unicidade isolada não exclui medidas contínuas.
Targets sugeridos têm 2–20 classes; a API de confirmação aceita até 30 e suporta
definição de faixa para valores numéricos contínuos. A escolha humana é registrada.

Leakage HIGH: nomes pós-evento ou relação categórica determinística com o target
em pelo menos 30 pares válidos, 2–20 categorias e 5 exemplos por categoria.
Leakage MEDIUM: correlação numérica absoluta >=.98 em pelo menos 30 pares.
Ausência de alerta não prova ausência de vazamento; revise disponibilidade temporal.

Qualidade mostra ausências, duplicatas, constantes, infinitos, classes pequenas,
desbalanceamento abaixo de 10% e leakage. Não há nota agregada opaca.
Predição exclui por padrão IDs, target, sensíveis, constantes e leakage HIGH;
essas exclusões são aplicadas pelo pipeline de treino atual.

## Limites e erros

100 MiB, 500000 linhas e 500 colunas por padrão; configuração em `core/config.py`.
XLSX expandido tem limite de 10 vezes o upload. Processamento é local, em threadpool,
sem LLM. Grandes bases exigem memória apropriada no servidor.

Erros: HTTP 400 para entrada inválida, 413 para limites, 503 para leitor Excel
ausente, 500 para falha interna sem exposição de células ou stack traces.
Formato: `{"detail":{"error":"CODIGO","message":"Mensagem"}}`.
Pydantic mantém HTTP 422 para campos obrigatórios/configurações inválidos.

O [README principal](../README.md) detalha o fluxo completo, treinamento,
fairness, explicações, persistência e testes.
