
import os
import requests
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FAOSTATLoader:
    """Загрузчик данных из FAOSTAT API"""
    
    BASE_URL = "https://api.fao.org/statistics/sdmx/2.1/data"
    
    def __init__(self, cache_dir: str = "data/raw/faostat"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_product_data(
        self, 
        product_code: str, 
        country_codes: List[str] = ["WLD"],
        start_year: int = 2000,
        end_year: int = 2024,
        element_code: str = "5510"  # Production
    ) -> pd.DataFrame:
        """
        Загрузка данных о производстве сельхозпродукции
        
        Args:
            product_code: Код продукта (например, '15' для пшеницы)
            country_codes: Список кодов стран/регионов
            start_year: Начальный год
            end_year: Конечный год
            element_code: Код элемента (5510=Production, 511=Yield, etc.)
        """
        # Для демо используем mock-данные, так как FAOSTAT API требует SDMX-запросы
        # В продакшене: реализовать SDMX-парсинг согласно [[6]][[10]]
        
        logger.info(f"Loading {product_code} data for {country_codes}")
        
        # Mock data generator для демонстрации
        return self._generate_mock_agricultural_data(
            product_code, country_codes, start_year, end_year
        )
    
    def _generate_mock_agricultural_data(
        self, product: str, countries: List[str], 
        start: int, end: int
    ) -> pd.DataFrame:
        """Генерация реалистичных mock-данных для тестирования"""
        records = []
        
        # Параметры для разных продуктов
        product_params = {
            'wheat': {'base': 750, 'trend': 2.5, 'seasonality': 0.15, 'volatility': 0.08},
            'milk': {'base': 800, 'trend': 1.8, 'seasonality': 0.25, 'volatility': 0.12},
            'rice': {'base': 500, 'trend': 3.0, 'seasonality': 0.20, 'volatility': 0.10},
        }
        
        params = product_params.get(product, product_params['wheat'])
        
        for country in countries:
            for year in range(start, end + 1):
                for month in range(1, 13):
                    # Базовый тренд
                    time_idx = (year - start) * 12 + month
                    trend = params['base'] + params['trend'] * time_idx
                    
                    # Сезонность (пик урожая)
                    seasonal = params['seasonality'] * np.sin(2 * np.pi * (month - 3) / 12)
                    
                    # Случайный шум
                    noise = np.random.normal(0, params['volatility'] * params['base'])
                    
                    value = trend * (1 + seasonal) + noise
                    value = max(0, value)  # Неотрицательные значения
                    
                    records.append({
                        'date': pd.Timestamp(year=year, month=month, day=1),
                        'country': country,
                        'product': product,
                        'value': round(value, 2),
                        'unit': '1000 tonnes',
                        'element': 'Production'
                    })
        
        df = pd.DataFrame(records)
        df['time_idx'] = (df['date'] - df['date'].min()).dt.days // 30
        
        return df


class USDALoader:
    """Загрузчик данных из USDA Market News API [[13]][[19]]"""
    
    BASE_URL = "https://mymarketnews.ams.usda.gov/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('USDA_API_KEY')
        if not self.api_key:
            logger.warning("USDA API key not found. Using mock data.")
    
    def fetch_market_prices(
        self, 
        commodity: str, 
        start_date: str, 
        end_date: str,
        location: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Загрузка рыночных цен из USDA
        
        Args:
            commodity: Тип товара (wheat, milk, corn, etc.)
            start_date: Дата начала в формате YYYY-MM-DD
            end_date: Дата окончания
            location: Опционально, код региона
        """
        if not self.api_key:
            return self._generate_mock_usda_data(commodity, start_date, end_date)
        
        headers = {'Authorization': f'Bearer {self.api_key}'}
        params = {
            'report_type': commodity,
            'start_date': start_date,
            'end_date': end_date,
            'format': 'json'
        }
        if location:
            params['location'] = location
            
        try:
            response = requests.get(
                f"{self.BASE_URL}/reports", 
                params=params, 
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return self._parse_usda_response(response.json())
            
        except requests.RequestException as e:
            logger.error(f"USDA API request failed: {e}")
            return self._generate_mock_usda_data(commodity, start_date, end_date)
    
    def _generate_mock_usda_data(
        self, commodity: str, start: str, end: str
    ) -> pd.DataFrame:
        """Mock-данные для USDA"""
        dates = pd.date_range(start=start, end=end, freq='W-MON')
        
        price_params = {
            'wheat': {'base': 6.5, 'volatility': 0.8},
            'milk': {'base': 18.5, 'volatility': 2.1},
        }
        params = price_params.get(commodity, price_params['wheat'])
        
        prices = []
        for date in dates:
            # Тренд + сезонность + шум
            time_factor = (date - pd.Timestamp(start)).days / 365
            seasonal = 0.3 * np.sin(2 * np.pi * date.month / 12)
            noise = np.random.normal(0, params['volatility'])
            
            price = params['base'] * (1 + 0.02 * time_factor + seasonal) + noise
            prices.append(max(0.1, price))
        
        return pd.DataFrame({
            'date': dates,
            'commodity': commodity,
            'price_per_unit': np.round(prices, 2),
            'unit': 'USD/bushel' if commodity == 'wheat' else 'USD/cwt',
            'volume': np.random.randint(1000, 50000, size=len(dates))
        })
    
    def _parse_usda_response(self, data: dict) -> pd.DataFrame:
        """Парсинг реального ответа USDA API"""
        # Реализация зависит от структуры ответа API
        # См. документацию [[13]][[19]]
        return pd.DataFrame(data.get('reports', []))


def load_agricultural_data(
    products: List[str] = ['wheat', 'milk'],
    countries: List[str] = ['WLD', 'USA', 'RUS'],
    start_year: int = 2010,
    end_year: int = 2024
) -> Dict[str, pd.DataFrame]:
    """
    Основная функция загрузки данных для всех продуктов
    
    Returns:
        Dict с DataFrame для каждого продукта
    """
    fao_loader = FAOSTATLoader()
    usda_loader = USDALoader()
    
    datasets = {}
    
    for product in products:
        logger.info(f"Loading data for {product}")
        
        # Данные производства из FAOSTAT
        production_df = fao_loader.fetch_product_data(
            product_code=product,
            country_codes=countries,
            start_year=start_year,
            end_year=end_year
        )
        
        # Данные цен из USDA
        prices_df = usda_loader.fetch_market_prices(
            commodity=product,
            start_date=f"{start_year}-01-01",
            end_date=f"{end_year}-12-31"
        )
        
        # Объединение данных
        merged = pd.merge(
            production_df,
            prices_df,
            left_on=['date', 'product'],
            right_on=['date', 'commodity'],
            how='left'
        )
        
        # Добавление целевой переменной: объём продаж ≈ производство * цена
        if 'price_per_unit' in merged.columns:
            merged['sales_volume'] = merged['value'] * merged['price_per_unit']
        else:
            merged['sales_volume'] = merged['value']  # fallback
            
        datasets[product] = merged
        logger.info(f"Loaded {len(merged)} records for {product}")
    
    return datasets
