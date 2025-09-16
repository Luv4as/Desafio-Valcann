
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from forecasting_implementation import ProdutoAlfaForecaster




def create_sample_data() -> pd.DataFrame:
    """
    Cria dados de exemplo para demonstração
    """
    np.random.seed(42)
    
    # Gerar datas para 2 anos
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 1, 1)
    dates = pd.date_range(start_date, end_date, freq='D')[:-1]  # Excluir último dia
    
    data = []
    for date in dates:
        # Tendência base com sazonalidade
        base_sales = 100 + np.sin(date.dayofyear * 2 * np.pi / 365) * 20
        
        # Efeito dia da semana
        weekday_effect = [1.2, 1.1, 1.0, 1.0, 1.1, 1.5, 1.4][date.weekday()]
        
        # Promoção (10% dos dias)
        is_promotion = np.random.random() < 0.1
        promotion_effect = 1.3 if is_promotion else 1.0
        
        # Feriado (5% dos dias)
        is_holiday = np.random.random() < 0.05
        holiday_effect = 0.7 if is_holiday else 1.0
        
        # Ruído
        noise = np.random.normal(0, 10)
        
        # Vendas finais
        sales = int(base_sales * weekday_effect * promotion_effect * holiday_effect + noise)
        sales = max(0, sales)  # Não pode ser negativo
        
        # Nome do dia
        weekday_names = ['segunda-feira', 'terca-feira', 'quarta-feira', 
                        'quinta-feira', 'sexta-feira', 'sabado', 'domingo']
        
        data.append({
            'data': date.strftime('%Y-%m-%d'),
            'vendas': sales,
            'dia_da_semana': weekday_names[date.weekday()],
            'em_promocao': is_promotion,
            'feriado_nacional': is_holiday
        })
    
    return pd.DataFrame(data)


# Exemplo de uso
if __name__ == "__main__":
    print("🚀 Sistema de Forecasting - Produto Alfa")
    print("=" * 50)
    
    # Criar dados de exemplo (substitua por seus dados reais)
    print("📊 Gerando dados de exemplo...")
    sample_data = create_sample_data()
    sample_data.to_csv('vendas_produto_alfa.csv', index=False)
    print("✅ Dados salvos em 'vendas_produto_alfa.csv'")
    
    # Inicializar forecaster
    forecaster = ProdutoAlfaForecaster(model_type='prophet')
    
    # Carregar dados
    df = forecaster.load_data('vendas_produto_alfa.csv')
    
    # Análise exploratória
    print("\n📈 Realizando análise exploratória...")
    analysis = forecaster.exploratory_analysis(df)
    print(f"Total de registros: {analysis['stats']['total_records']}")
    print(f"Vendas médias: {analysis['stats']['mean_sales']:.2f}")
    print(f"Impacto da promoção: +{((analysis['seasonality']['promotion_impact']['with_promo'] / analysis['seasonality']['promotion_impact']['without_promo'] - 1) * 100):.1f}%")
    
    # Treinar modelo
    print("\n🤖 Treinando modelo...")
    forecaster.train_model(df)
    
    # Avaliar modelo
    print("\n📊 Avaliando performance...")
    metrics = forecaster.evaluate_model(df)
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"R²: {metrics['R2']:.4f}")
    
    # Fazer previsões
    print("\n🔮 Fazendo previsões para 14 dias...")
    predictions = forecaster.predict(periods=14)
    print(predictions.head())
    
    # Salvar modelo
    forecaster.save_model('modelo_produto_alfa.pkl')
    print("\n💾 Modelo salvo com sucesso!")
    
    print("\n✨ Pipeline concluído com sucesso!")