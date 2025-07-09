import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests

# --- 1. Загрузка данных с Binance ---
symbol = "BTCUSDT"
interval = "1d"
limit = 500 #примерно 1 месяц
url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"

response = requests.get(url)
if response.status_code != 200:
    raise Exception(f"Ошибка API: {response.status_code} - {response.text}")

data = response.json()
df = pd.DataFrame(data, columns=[
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'num_trades',
    'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
])
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
df.set_index('open_time', inplace=True)
df['symbol'] = 'BTC/USD'
df.set_index('symbol', append=True, inplace=True)
df = df.swaplevel('open_time', 'symbol')

# --- 2. Параметры ---
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume']
time_steps = 60
horizon = 1
target_col = 'close'
symbol = 'BTC/USD'

# --- 3. Масштабирование ---
feature_scalers = {symbol: MinMaxScaler()}
df_btc = df.xs(symbol, level='symbol')
df_scaled = pd.DataFrame(
    feature_scalers[symbol].fit_transform(df_btc[FEATURE_COLS]),
    columns=FEATURE_COLS,
    index=df_btc.index
)

# --- 4. Разделение данных ---
train_size = int(len(df_scaled) * 0.8)
train_scaled = df_scaled.iloc[:train_size]
test_scaled = df_scaled.iloc[train_size:]

# --- 5. Функция для создания последовательностей ---
def create_sequences(df_sym, feature_cols, time_steps, horizon, target_col):
    X, y, dates = [], [], []
    values = df_sym[feature_cols].values
    for i in range(len(values) - time_steps - horizon + 1):
        X.append(values[i:i + time_steps])
        target_idx = i + time_steps + horizon - 1
        y.append(values[target_idx, feature_cols.index(target_col)])
        dates.append(df_sym.index[target_idx])
    return np.array(X), np.array(y), np.array(dates)

# --- 6. Обучение модели ---
X_train, y_train, _ = create_sequences(
    train_scaled, FEATURE_COLS, time_steps, horizon, target_col
)
model = Sequential([
    LSTM(50, activation='relu', input_shape=(time_steps, len(FEATURE_COLS))),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# --- 7. Последовательности для теста ---
df_btc_test = test_scaled
X_btc, y_btc_scaled, dates_btc = create_sequences(
    df_btc_test, FEATURE_COLS, time_steps, horizon, target_col
)

# --- 8. Прогноз ---
y_btc_pred_scaled = model.predict(X_btc, verbose=0).flatten()

# --- 9. Обратное преобразование ---
scaler_btc = feature_scalers[symbol]
close_idx = FEATURE_COLS.index(target_col)

def inv_transform_close(scaled_vec):
    tmp = np.zeros((scaled_vec.shape[0], len(FEATURE_COLS)))
    tmp[:, close_idx] = scaled_vec
    log_close = scaler_btc.inverse_transform(tmp)[:, close_idx]
    return log_close  # Убрал np.exp, так как Binance данные не логарифмированы

true_prices = inv_transform_close(y_btc_scaled)
pred_prices = inv_transform_close(y_btc_pred_scaled)

# --- 10. Датафрейм с результатами ---
btc_preds = pd.DataFrame({
    'date': pd.to_datetime(dates_btc),
    'true_price': true_prices,
    'pred_price': pred_prices
}).set_index('date')

# --- Метрики ---
mse = mean_squared_error(btc_preds.true_price, btc_preds.pred_price)
rmse = np.sqrt(mse)  # Исправление для RMSE
mae = mean_absolute_error(btc_preds.true_price, btc_preds.pred_price)

mean_price = btc_preds.true_price.mean()
rmse_pct = rmse / mean_price * 100
mae_pct = mae / mean_price * 100

print(f"RMSE  : {rmse:,.2f}  ({rmse_pct:.2f}% of mean price)")
print(f"MAE   : {mae:,.2f}   ({mae_pct:.2f}% of mean price)")
print(btc_preds.head())

# --- График ---
plt.figure(figsize=(12, 5))
plt.plot(btc_preds.index, btc_preds.true_price, label='True')
plt.plot(btc_preds.index, btc_preds.pred_price, label='Predicted')
plt.title('BTC/USD ‒ прогноз vs. фактическая цена (весь тест)')
plt.xlabel('Date')
plt.ylabel('Price, $')
plt.legend()
plt.tight_layout()
plt.show()