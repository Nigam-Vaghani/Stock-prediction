import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

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
    # Fix for Render (free tier has limited CPU, making yf multi-threading fail)
    import requests
    proxy_url = "http://scraperapi:abf81d1434ca40c3b67a83ed2e0e7627b6d7eb277bd@proxy-server.scraperapi.com:8001"
    
    session = requests.Session()
    session.proxies.update({
        "http": proxy_url,
        "https": proxy_url
    })
    session.headers.update({
        "User-Agent": "Mozilla/5.0 ..."
    })
    print("hello")
    data = yf.download(stock_symbol, period="3mo", threads=False)

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