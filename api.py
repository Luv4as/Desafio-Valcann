from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import datetime, timedelta
import joblib
import logging
from typing import List, Optional
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Produto Alfa Forecasting API",
    description="API para previsão de demanda do Produto Alfa",
    version="1.0.0"
)

# Carregar modelo no startup
model_forecaster = None

@app.on_event("startup")
async def load_model():
    """Carrega o modelo treinado na inicialização"""
    global model_forecaster
    try:
        from forecasting_implementation import ProdutoAlfaForecaster
        model_forecaster = ProdutoAlfaForecaster()
        model_path = os.getenv('MODEL_PATH', 'modelo_produto_alfa.pkl')
        model_forecaster.load_model(model_path)
        logger.info("Modelo carregado com sucesso")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        raise

# Modelos Pydantic para requisições
class ForecastRequest(BaseModel):
    periods: int = 14
    include_promotions: Optional[List[str]] = None  # Datas de promoção
    include_holidays: Optional[List[str]] = None    # Datas de feriado

class ForecastResponse(BaseModel):
    data: str
    previsao: float
    limite_inferior: float
    limite_superior: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de saúde da aplicação"""
    return HealthResponse(
        status="healthy" if model_forecaster else "unhealthy",
        model_loaded=model_forecaster is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=List[ForecastResponse])
async def predict(request: ForecastRequest):
    """
    Endpoint para fazer previsões de demanda
    """
    try:
        if not model_forecaster:
            raise HTTPException(status_code=500, detail="Modelo não carregado")
        
        # Fazer previsão
        predictions = model_forecaster.predict(periods=request.periods)
        
        # Converter para formato de resposta
        response = []
        for _, row in predictions.iterrows():
            response.append(ForecastResponse(
                data=row['data'].strftime('%Y-%m-%d'),
                previsao=round(row['previsao'], 2),
                limite_inferior=round(row['limite_inferior'], 2),
                limite_superior=round(row['limite_superior'], 2)
            ))
        
        logger.info(f"Previsão gerada para {request.periods} períodos")
        return response
        
    except Exception as e:
        logger.error(f"Erro na previsão: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Retorna informações sobre o modelo"""
    if not model_forecaster:
        raise HTTPException(status_code=500, detail="Modelo não carregado")
    
    return {
        "model_type": model_forecaster.model_type,
        "is_trained": model_forecaster.is_trained,
        "feature_importance": model_forecaster.feature_importance.to_dict('records') if model_forecaster.feature_importance is not None else None
    }