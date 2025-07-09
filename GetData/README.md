# Основной ресурс для данных https://www.coinapi.io/.
- Сначала тянем данные с coinapi_symbol_id_by_exchanges.py, где смотрим, чтобы "symbol_id": содержал "BINANCEFTS_PERP_НАШ_АКТИВ_USDT" (пример "symbol_id": "BINANCEFTS_PERP_TURBO_USDT")
- Вставляем значение "symbol_id" в binance_data.py после https://rest.coinapi.io/v1/ohlcv/ (https://rest.coinapi.io/v1/ohlcv/BINANCEFTS_PERP_BTC_USDT/) и добавляем нужные параметры (https://rest.coinapi.io/v1/ohlcv/BINANCEFTS_PERP_BTC_USDT/history?period_id=1DAY&time_start=2020-07-09T02:47:19.000Z&time_end=2020-07-20T02:47:19.000Z)


Время можно перевести здесь https://www.timestamp-converter.com/
