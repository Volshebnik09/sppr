import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import moexalgo
from datetime import datetime, timedelta
import numpy as np

st.title('Система поддержки принятия решений для инвестиций')

# Переключатель для выбора списка акций
show_russian = st.checkbox('Показывать российские акции', value=False)

TICKERS_FOREIGN = [
    'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'BTC-USD', 'ETH-USD',
]
TICKERS_RU = [
    'SBER', 'GAZP', 'LKOH', 'GMKN', "IRAO", 'TATN', 'SNGS', 'ROSN', 'ALRS',
]

TICKERS = TICKERS_RU if show_russian else TICKERS_FOREIGN

ticker = st.selectbox('Выберите актив для анализа', TICKERS)

@st.cache_data
def load_data(ticker, is_russian):
    if is_russian:
        # Для российских акций через moexalgo 
        ticker_data = moexalgo.Ticker(ticker)
        end = datetime.today().date()
        start = end - timedelta(days=730)  
        df = ticker_data.candles(
            start=start,
            end=end,
            period=24,  # дневные свечи
            use_dataframe=True
        )
        df = df.rename(columns={
            'close': 'Close',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'volume': 'Volume',
            'begin': 'Date',
        })
        df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
        df = df.set_index('Date')
        return df
    else:
        data = yf.download(ticker, period='2y', interval='1d')
        data = data.dropna()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(-2)
        if not data.columns.is_unique:
            cols = [col for col in ['Close', 'High', 'Low', 'Open', 'Volume'] if col in data.columns]
            data = data[cols]
        return data

data = load_data(ticker, show_russian)
data = data.sort_index()  
if 'Close' not in data.columns:
    st.error('Нет данных "Close" для выбранного тикера. Попробуйте другой актив.')
    st.stop()
st.write('Текущие данные:', data.tail(30))

# Подготовка данных для модели
window = 5
def prepare_data(df, window=5):
    features = ['Close', 'Open', 'High', 'Low', 'Volume']
    X, y = [], []
    values = df[features].to_numpy()
    for i in range(window, len(df)):
        window_values = values[i-window:i].flatten()
        X.append(window_values)
        y.append(values[i][0])  
    return pd.DataFrame(X), pd.Series(y)

X, y = prepare_data(data, window)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Обучение модели
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание
preds = model.predict(X_test)

# Визуализация (текущий график: фактические + прогноз на тесте)
dates = data.index[window:]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dates, y, label='Истинные значения')
ax.plot(dates[len(y_train):], preds, label='Прогноз', linestyle='--')
ax.set_title(f'Прогноз курса {ticker} (тестовый прогноз)')
ax.legend()
ax.set_xlabel('Дата')
ax.set_ylabel('Цена')
fig.autofmt_xdate(rotation=30)
st.pyplot(fig)

# --- Будущий прогноз ---
future_steps_option = st.selectbox('Горизонт будущего прогноза', options=[5, 10, 15, 45], format_func=lambda x: f'{x} дней')
future_steps = future_steps_option

last_window = X.iloc[-1].values.reshape(1, -1)
future_preds = []
for _ in range(future_steps):
    next_pred = model.predict(last_window)[0]
    future_preds.append(next_pred)
    last_window = np.roll(last_window, -1)
    last_window[0, -1] = next_pred
last_date = data.index[-1]
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, future_steps+1)]

fig2, ax2 = plt.subplots(figsize=(10, 5))
show_hist = min(10, len(y))
hist_start = max(0, len(y)-show_hist)
hist_dates = [data.index[i].date() for i in range(hist_start, len(y))]
ax2.plot(range(hist_start, len(y)), y[-show_hist:], label='Истинные значения (последние)', color='blue')
ax2.plot(range(len(y), len(y) + future_steps), future_preds, label='Будущий прогноз', linestyle=':', color='orange')
# Формируем список всех дат для оси X только для последних истинных значений и будущих прогнозов
future_dates_str = [d.date() for d in future_dates]
all_dates_short = hist_dates + future_dates_str
xticks = range(hist_start, len(y) + future_steps)
if future_steps > 10:
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([str(d) for d in all_dates_short], rotation=60, ha='right')
    for label in ax2.get_xticklabels():
        label.set_fontsize(8)
else:
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([str(d) for d in all_dates_short], rotation=30, ha='right')
ax2.set_title(f'Будущий прогноз курса {ticker} (на {future_steps} дней)')
ax2.legend()
st.pyplot(fig2)

# Таблица с ближайшими предсказаниями курса (будущий прогноз)
pred_table = pd.DataFrame({'Дата': future_dates, 'Прогноз': future_preds})
pred_table = pred_table.sort_values('Дата').reset_index(drop=True)
st.dataframe(pred_table, hide_index=True)

st.write(f'MSE: {mean_squared_error(y_test, preds):.4f}')

# --- Рекомендация по покупке/продаже ---
current_price = data['Close'].iloc[-1]
future_price = future_preds[-1] if future_preds else current_price

first_future_price = future_preds[0] if future_preds else current_price

horizon_delta = future_price - current_price
threshold = 0.005 * current_price  # 0.5% для нейтральной зоны

if horizon_delta > threshold:
    st.success(f'Рекомендация на {future_steps} дней: Покупать (ожидается рост с {current_price:.2f} до {future_price:.2f})')
elif horizon_delta < -threshold:
    st.error(f'Рекомендация на {future_steps} дней: Продавать (ожидается снижение с {current_price:.2f} до {future_price:.2f})')
else:
    st.info(f'Рекомендация на {future_steps} дней: Держать (существенных изменений не ожидается)')
