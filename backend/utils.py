import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
import os
# --- THE NEW WORKAROUND: Monkey-Patching ---
# 1. Save the original initialization method of the built-in Dense layer
_original_dense_init = Dense.__init__

# 2. Create a wrapper function that removes the bad keyword
def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)  # Strip the unrecognized argument
    _original_dense_init(self, *args, **kwargs) # Call the original init

# 3. Forcibly overwrite the Dense layer's init method globally
Dense.__init__ = _patched_dense_init
# ---------------------------------------------

# Now we can load the model normally without custom_objects!
model = None

def load_my_model():
    global model
    if model is None:
        # from tensorflow.keras.models import load_model
        model = load_model("lstm_stock_model.keras", compile=False)
    return model

scaler = MinMaxScaler(feature_range=(0, 1))

def get_prediction(stock_symbol):
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Only use proxy in production (Render) — set SCRAPER_PROXY_URL env var there
    proxy_url = os.environ.get("SCRAPER_PROXY_URL")

    download_kwargs = {
        "tickers": stock_symbol,
        "period": "6mo",
        "threads": False,
        "progress": False,
    }

    if proxy_url:
        session = requests.Session()
        session.verify = False
        session.proxies.update({
            "http": proxy_url,
            "https": proxy_url,
        })
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        download_kwargs["session"] = session

    print(f"Fetching data for: {stock_symbol} (proxy={'yes' if proxy_url else 'no'})")
    data = yf.download(**download_kwargs)
    if data.empty:
        raise ValueError(f"Yahoo Finance returned no data for {stock_symbol}. It might be delisted or Render IP is temporarily blocked.")

    close_prices = data[['Close']]

    # Current price
    current_price = float(close_prices.iloc[-1][0])

    # Normalize
    scaled_data = scaler.fit_transform(close_prices)

    # Last 60 days
    last_60 = scaled_data[-60:]
    X_test = np.reshape(last_60, (1, 60, 1))

    # Predict
    model = load_my_model()
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    predicted_price = float(predicted_price[0][0])

    # % change
    change_percent = ((predicted_price - current_price) / current_price) * 100

    return {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "change_percent": change_percent
    }