# FairDrop Backend

API FastAPI com leitura de bases, AutoML, avaliação AIF360, explicações SHAP,
modelos persistentes e histórico de previsões. O [guia principal](../README.md)
explica o fluxo completo, configurações, critérios metodológicos e endpoints.

## Executar localmente

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

API: http://localhost:8000 · documentação: http://localhost:8000/docs.
Copie `.env.example` para `.env` caso queira configurar armazenamento, limites
ou chave de acesso. Dois modelos fictícios de `demo_assets` são instalados no
primeiro início. São artefatos reais e funcionam sem rede ou treinamento prévio.

## Docker

Nesta pasta: `docker compose up --build -d` executa somente a API.
Na raiz FairDrop: `docker compose up --build -d` executa a aplicação completa.
Volumes preservam o banco e os modelos entre reinícios. AIF360 é instalado sem
extras `[all]`; não há dependência de R/rpy2.

## Testar

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m pytest tests -q
.\venv\Scripts\python.exe -m compileall -q app
```

Testes cobrem leitura, heurísticas, pipeline, seleção, fairness, calibração,
persistência, inferência, SHAP, exemplos prontos e compatibilidade. As bases
locais são verificadas por `test_project_datasets.py` quando presentes.
O banco de testes é isolado. O volume de desenvolvimento fica em `data/platform`.

## Organização

- `app/routes`: análise, datasets, treino, modelos, predições e demonstração.
- `app/schemas`: contratos Pydantic.
- `app/services`: serviços independentes de análise, ML, fairness e armazenamento.
- `app/core/config.py`: configurações e pesos explícitos.
- `demo_assets`: modelos e arquivos fictícios distribuídos.
- `scripts/build_demo.py`: reprodução dos exemplos com versões atuais.

Use um worker; os jobs em segundo plano são locais ao processo, sem fila
distribuída. Consulte [detalhes da importação](DATASET_ANALYSIS.md).
