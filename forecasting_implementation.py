"""
Módulo de modelagem para previsão de demanda do Produto Alfa
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

class ProdutoAlfaForecaster:
    def exploratory_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Realiza análise exploratória dos dados
        """
        analysis = {}
        analysis['stats'] = {
            'total_records': len(df),
            'date_range': (df['data'].min(), df['data'].max()),
            'mean_sales': df['vendas'].mean(),
            'std_sales': df['vendas'].std(),
            'missing_values': df.isnull().sum().to_dict()
        }
        df['weekday'] = df['data'].dt.dayofweek
        df['month'] = df['data'].dt.month
        df['year'] = df['data'].dt.year
        analysis['seasonality'] = {
            'by_weekday': df.groupby('dia_da_semana')['vendas'].mean().to_dict() if 'dia_da_semana' in df.columns else {},
            'by_month': df.groupby('month')['vendas'].mean().to_dict(),
            'promotion_impact': {
                'with_promo': df[df['em_promocao']]['vendas'].mean(),
                'without_promo': df[~df['em_promocao']]['vendas'].mean()
            },
            'holiday_impact': {
                'holiday': df[df['feriado_nacional']]['vendas'].mean(),
                'non_holiday': df[~df['feriado_nacional']]['vendas'].mean()
            }
        }
        Q1 = df['vendas'].quantile(0.25)
        Q3 = df['vendas'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df['vendas'] < Q1 - 1.5*IQR) | (df['vendas'] > Q3 + 1.5*IQR)]
        analysis['outliers'] = len(outliers)
        self.logger.info("Análise exploratória concluída")
        return analysis

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features para o modelo
        """
        df_features = df.copy()
        df_features['weekday'] = df_features['data'].dt.dayofweek
        df_features['month'] = df_features['data'].dt.month
        df_features['year'] = df_features['data'].dt.year
        df_features['quarter'] = df_features['data'].dt.quarter
        df_features['day_of_year'] = df_features['data'].dt.dayofyear
        for lag in [1, 7, 14, 30]:
            df_features[f'lag_{lag}'] = df_features['vendas'].shift(lag)
        for window in [7, 14, 30]:
            df_features[f'ma_{window}'] = df_features['vendas'].rolling(window).mean()
        if 'dia_da_semana' in df_features.columns:
            le = LabelEncoder()
            df_features['dia_da_semana_encoded'] = le.fit_transform(df_features['dia_da_semana'])
            self.encoders['dia_da_semana'] = le
        df_features['promo_weekend'] = (df_features['em_promocao'] & (df_features['weekday'] >= 5)).astype(int)
        df_features['holiday_promo'] = (df_features['feriado_nacional'] & df_features['em_promocao']).astype(int)
        return df_features

    def prepare_prophet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = df['data']
        prophet_df['y'] = df['vendas']
        prophet_df['promocao'] = df['em_promocao'].astype(int)
        prophet_df['feriado'] = df['feriado_nacional'].astype(int)
        prophet_df['fim_de_semana'] = (df['data'].dt.dayofweek >= 5).astype(int)
        return prophet_df

    def train_model(self, df: pd.DataFrame) -> None:
        try:
            if self.model_type == 'prophet':
                self._train_prophet(df)
            elif self.model_type == 'rf':
                self._train_random_forest(df)
            else:
                raise ValueError("Tipo de modelo não suportado")
            self.is_trained = True
            self.logger.info(f"Modelo '{self.model_type}' treinado com sucesso!")
        except Exception as e:
            self.logger.error(f"Erro ao treinar modelo: {e}")
            raise

    def _train_prophet(self, df: pd.DataFrame) -> None:
        prophet_data = self.prepare_prophet_data(df)
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=0.1
        )
        self.model.add_regressor('promocao')
        self.model.add_regressor('feriado')
        self.model.add_regressor('fim_de_semana')
        self.model.fit(prophet_data)

    def _train_random_forest(self, df: pd.DataFrame) -> None:
        df_features = self.create_features(df)
        df_clean = df_features.dropna()
        feature_cols = ['weekday', 'month', 'quarter', 'day_of_year',
                       'em_promocao', 'feriado_nacional', 'dia_da_semana_encoded',
                       'promo_weekend', 'holiday_promo'] + \
                      [col for col in df_clean.columns if 'lag' in col or 'ma' in col]
        X = df_clean[feature_cols]
        y = df_clean['vendas']
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        self.model.fit(X, y)
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def predict(self, periods: int = 14, future_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        if self.model_type == 'prophet':
            return self._predict_prophet(periods, future_features)
        elif self.model_type == 'rf':
            raise NotImplementedError("Previsão para Random Forest não implementada neste exemplo.")

    def _predict_prophet(self, periods: int, future_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=periods)
        if future_features is not None:
            for col in ['promocao', 'feriado', 'fim_de_semana']:
                if col in future_features.columns:
                    future[col] = future_features[col].values
                else:
                    future[col] = 0
        else:
            future['promocao'] = 0
            future['feriado'] = 0
            future['fim_de_semana'] = (future['ds'].dt.dayofweek >= 5).astype(int)
        forecast = self.model.predict(future)
        future_forecast = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        future_forecast = future_forecast.rename(columns={
            'ds': 'data',
            'yhat': 'previsao',
            'yhat_lower': 'limite_inferior',
            'yhat_upper': 'limite_superior'
        })
        return future_forecast.reset_index(drop=True)

    def evaluate_model(self, df: pd.DataFrame, test_size: float = 0.2, n_splits: int = 3) -> Dict:
        """
        Avalia o modelo usando validação cruzada temporal (TimeSeriesSplit)
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        maes, mapes, r2s = [], [], []
        df = df.sort_values('data')
        for train_idx, test_idx in tscv.split(df):
            train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
            self.train_model(train_df)
            preds = self.predict(len(test_df))
            actual = test_df['vendas'].values
            predicted = preds['previsao'].values
            maes.append(mean_absolute_error(actual, predicted))
            mapes.append(np.mean(np.abs((actual - predicted) / actual)) * 100)
            r2s.append(r2_score(actual, predicted))
        metrics = {
            'MAE': np.mean(maes),
            'MAPE': np.mean(mapes),
            'R2': np.mean(r2s)
        }
        self.logger.info(f"Métricas de validação cruzada: {metrics}")
        return metrics

    def save_model(self, path: str) -> None:
        import joblib
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'encoders': self.encoders,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, path)
        self.logger.info(f"Modelo salvo em {path}")

    def load_model(self, path: str) -> None:
        import joblib
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.encoders = model_data['encoders']
        self.feature_importance = model_data.get('feature_importance')
        self.is_trained = True
        self.logger.info(f"Modelo carregado de {path}")
    """
    Classe principal para previsão de demanda do Produto Alfa.

    Exemplos de uso:
    -----------
    >>> forecaster = ProdutoAlfaForecaster(model_type='prophet')
    >>> df = forecaster.load_data('vendas_produto_alfa.csv')
    >>> analysis = forecaster.exploratory_analysis(df)
    >>> forecaster.train_model(df)
    >>> metrics = forecaster.evaluate_model(df)
    >>> previsoes = forecaster.predict(periods=14)
    >>> forecaster.save_model('modelo_produto_alfa.pkl')
    """
    def __init__(self, model_type: str = 'prophet'):
        self.model_type = model_type
        self.model = None
        self.encoders = {}
        self.is_trained = False
        self.feature_importance = None
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa e valida os dados: trata valores ausentes, tipos e outliers.
        """
        df = df.copy()
        # Corrigir tipos
        df['data'] = pd.to_datetime(df['data'])
        df['em_promocao'] = df['em_promocao'].astype(bool)
        df['feriado_nacional'] = df['feriado_nacional'].astype(bool)
        # Preencher valores ausentes
        df['vendas'] = df['vendas'].fillna(df['vendas'].median())
        # Tratar outliers (IQR)
        Q1 = df['vendas'].quantile(0.25)
        Q3 = df['vendas'].quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
        df['vendas'] = np.clip(df['vendas'], lim_inf, lim_sup)
        return df

    def load_data(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            df = self.clean_data(df)
            return df
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {e}")
            raise

    # ... (demais métodos: exploratory_analysis, create_features, prepare_prophet_data, train_model, etc.)
    # Copie os métodos do main.py para cá, adaptando importações se necessário.
