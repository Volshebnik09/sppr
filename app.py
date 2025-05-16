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
    'SBER', 'GAZP', 'LKOH', 'GMKN'
]

TICKERS = TICKERS_RU if show_russian else TICKERS_FOREIGN

ticker = st.selectbox('Выберите актив для анализа', TICKERS)

@st.cache_data
def load_data(ticker, is_russian):
    if is_russian:
        # Для российских акций через moexalgo (актуальный способ)
        ticker_data = moexalgo.Ticker(ticker)
        end = datetime.today().date()
        start = end - timedelta(days=730)  # 2 года
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
# Проверяем наличие столбца 'Close' перед подготовкой данных
if 'Close' not in data.columns:
    st.error('Нет данных "Close" для выбранного тикера. Попробуйте другой актив.')
    st.stop()
st.write('Текущие данные:', data.tail())

# Подготовка данных для модели
window = 5
def prepare_data(df, window=5):
    X, y = [], []
    close = df['Close'].to_numpy().flatten()  # гарантируем 1D массив
    for i in range(window, len(df)):
        X.append(close[i-window:i])
        y.append(close[i])
    return pd.DataFrame(X), pd.Series(y)

X, y = prepare_data(data, window)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Обучение модели
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание
preds = model.predict(X_test)

# Визуализация (текущий график: фактические + прогноз на тесте)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(len(y)), y, label='Истинные значения')
ax.plot(range(len(y_train), len(y)), preds, label='Прогноз', linestyle='--')
ax.set_title(f'Прогноз курса {ticker} (тестовый прогноз)')
ax.legend()
st.pyplot(fig)

# --- Будущий прогноз ---
# Выбор горизонта прогноза
future_steps_option = st.selectbox('Горизонт будущего прогноза', options=[5, 10, 15, 45], format_func=lambda x: f'{x} дней')
future_steps = future_steps_option

# future_steps теперь задаётся пользователем
last_window = X.iloc[-1].values.reshape(1, -1)
future_preds = []
for _ in range(future_steps):
    next_pred = model.predict(last_window)[0]
    future_preds.append(next_pred)
    last_window = np.roll(last_window, -1)
    last_window[0, -1] = next_pred
last_date = data.index[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_steps, freq='B')

# Отдельный график: будущий прогноз
fig2, ax2 = plt.subplots(figsize=(10, 5))
# Показываем только последние несколько истинных значений (например, последние 10), чтобы видеть тренд перед прогнозом
show_hist = min(10, len(y))
ax2.plot(range(len(y)-show_hist, len(y)), y[-show_hist:], label='Истинные значения (последние)', color='blue')
ax2.plot(range(len(y), len(y) + future_steps), future_preds, label='Будущий прогноз', linestyle=':', color='orange')
# Формируем список всех дат для оси X только для последних истинных значений и будущих прогнозов
hist_dates = [data.index[i].date() for i in range(len(y)-show_hist, len(y))]
future_dates_str = [d.date() for d in future_dates]
all_dates_short = hist_dates + future_dates_str
xticks = range(len(y)-show_hist, len(y) + future_steps)
# Если прогноз длиннее 10 дней, уменьшаем размер шрифта для подписей дат
if future_steps > 10:
    for label in ax2.get_xticklabels():
        label.set_fontsize(8)
    ax2.set_xticklabels([str(d) for d in all_dates_short], rotation=60, ha='right')
else:
    ax2.set_xticklabels([str(d) for d in all_dates_short], rotation=30, ha='right')
ax2.set_xticks(xticks)
ax2.set_title(f'Будущий прогноз курса {ticker} (на {future_steps} дней)')
ax2.legend()
st.pyplot(fig2)

# Таблица с ближайшими предсказаниями курса (будущий прогноз)
pred_table = pd.DataFrame({'Дата': future_dates, 'Прогноз': future_preds})
st.write(f'{future_steps} ближайших прогнозов:', pred_table)

st.write(f'MSE: {mean_squared_error(y_test, preds):.2f}')
