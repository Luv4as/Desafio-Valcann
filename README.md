# Produto Alfa
Lucas Venancio

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

