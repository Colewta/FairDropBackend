# FairDrop Backend

API do FairDrop responsável por analisar datasets em CSV, treinar modelos de classificação e retornar métricas de desempenho e fairness para consumo do frontend. A pasta `data` contém a base de dados recomendada para ser usada para testes no sistema, porém, qualquer base de dados relacionada a evasão que seja `.csv` poderá ser utilizada.

## Visão Geral

O backend foi projetado para rodar em contêiner Docker. Essa é a forma prevista de execução do projeto neste momento e deve ser mantida dessa maneira.

Ao subir o serviço, a API FastAPI fica disponível na porta `8000` e expõe, entre outras, as seguintes rotas:

- `GET /health`: verificação simples de funcionamento da API.
- `POST /analyze`: análise inicial do dataset enviado em CSV.
- `POST /train`: treinamento e comparação dos modelos disponíveis.
- `POST /simulate`: simulação com base no modelo treinado em memória.
- `GET /docs`: documentação interativa gerada pelo FastAPI.

## Pré-requisitos

- Docker (`https://docs.docker.com/desktop/setup/install/windows-install/` para Windows)
- Docker Compose

## Como Executar

No diretório `FairDropBackend`, execute:

```bash
docker compose up --build
```

Se preferir rodar em segundo plano:

```bash
docker compose up --build -d
```

Após a inicialização, a API estará disponível em:

- `http://localhost:8000`
- `http://localhost:8000/docs`
- `http://localhost:8000/health`

## Configuração Atual

O `docker-compose.yml` já está preparado para:

- publicar a API na porta `8000`;
- permitir acesso do frontend local nas portas `5173` e `8080`;
- persistir uploads no volume Docker `backend_uploads`;
- reiniciar o contêiner automaticamente com `unless-stopped`.

## Comandos Úteis

Subir o backend:

```bash
docker compose up --build
```

Parar os serviços:

```bash
docker compose down
```

Visualizar logs:

```bash
docker compose logs -f backend
```

Remover contêineres e volume persistido:

```bash
docker compose down -v
```

Use `down -v` apenas quando quiser limpar também os arquivos persistidos no volume do Docker.

## Integração com o Frontend

O frontend do projeto não deve ser executado em Docker neste momento. A arquitetura atual é:

- backend em Docker;
- frontend local com `npm install` e `npm run dev`.

Com essa configuração, o frontend em desenvolvimento acessa a API em `http://localhost:8000`.

## Observações

- Sempre execute os comandos a partir da pasta `FairDropBackend`.
- Se houver alteração no código Python ou nas dependências, reconstrua o serviço com `docker compose up --build`.
- Caso o frontend seja exposto em outra origem além das já previstas, será necessário ajustar a variável `FRONTEND_ORIGINS` no `docker-compose.yml`.
