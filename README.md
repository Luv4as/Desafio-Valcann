# Produto Alfa - Forecasting Pipeline

Este projeto implementa um pipeline completo de previsão de demanda para o Produto Alfa, incluindo ingestão, análise exploratória, limpeza, modelagem, deploy em nuvem e monitoramento.

## Pipeline de Solução

```mermaid
flowchart LR
    A[Ingestão de Dados (S3/Blob)] --> B[Pré-processamento (EC2/Job)]
    B --> C[Treinamento (EC2/Vertex AI/SageMaker)]
    C --> D[Armazenamento do Modelo (S3/Blob)]
    D --> E[API de Previsão (FastAPI + Docker + Kubernetes)]
    E --> F[Monitoramento (Prometheus/Evidently)]
    F --> G[Alertas (Grafana/CloudWatch)]
```

## Como usar

1. Gere dados de exemplo ou use seu arquivo `vendas_produto_alfa.csv`.
2. Execute `main.py` para treinar, avaliar e salvar o modelo.
3. Faça deploy da API com Docker:
   ```sh
   docker build -t produtoalfa-api .
   docker run -p 8000:8000 produtoalfa-api
   ```
4. Acesse a API em `http://localhost:8000/docs`.

## Principais arquivos
- `forecasting_implementation.py`: Lógica de modelagem e previsão.
- `api.py`: API FastAPI para servir previsões.
- `main.py`: Pipeline de experimentação e treinamento.
- `monitoring.py`: Monitoramento e métricas.

## Monitoramento
- Métricas de previsão expostas via Prometheus.
- Drift e qualidade de dados monitorados com Evidently.

## Exemplo de uso da API
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"periods": 14}'
```

---

> Para produção, recomenda-se orquestração com Kubernetes, armazenamento de modelos em nuvem e CI/CD para automação.
