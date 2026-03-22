import { useState } from "react";
import "./App.css";

const STOCKS = [
  { name: "Apple", symbol: "AAPL", exchange: "NASDAQ" },
  { name: "Tesla", symbol: "TSLA", exchange: "NASDAQ" },
  { name: "Google", symbol: "GOOGL", exchange: "NASDAQ" },
  { name: "Amazon", symbol: "AMZN", exchange: "NASDAQ" },
  { name: "Microsoft", symbol: "MSFT", exchange: "NASDAQ" },
  { name: "Nvidia", symbol: "NVDA", exchange: "NASDAQ" },

  { name: "Reliance", symbol: "RELIANCE.NS", exchange: "NSE" },
  { name: "TCS", symbol: "TCS.NS", exchange: "NSE" },
  { name: "Infosys", symbol: "INFY.NS", exchange: "NSE" }
];

function App() {
  const [selectedStock, setSelectedStock] = useState("AAPL");
  const [customStock, setCustomStock] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  // 🔹 Find selected stock object
  const selectedStockData = STOCKS.find(
    (s) => s.symbol === selectedStock
  );

  // 🔹 Final stock (custom input or dropdown)
  const getBackendSymbol = () => {
    let symbol = customStock || selectedStock;
    if (!symbol.includes(".") && symbol.length <= 10) {
      if (
        symbol === symbol.toUpperCase() &&
        !["AAPL","TSLA","GOOGL","AMZN","MSFT","NVDA"].includes(symbol)
      ) {
        return symbol + ".NS";
      }
    }
    return symbol;
  };

  const finalStock = getBackendSymbol();

  // 🔹 FIX: Convert symbol for TradingView
  const getChartSymbol = () => {
    if (customStock) return customStock;

    if (!selectedStockData) return selectedStock;

    if (selectedStockData.exchange === "NSE") {
      const cleanSymbol = selectedStockData.symbol.replace(".NS", "");
      return `NSE:${cleanSymbol}`;
    }

    return `${selectedStockData.exchange}:${selectedStockData.symbol}`;
  };

  const predictStock = async () => {
    setLoading(true);
    setData(null);

    try {
      const res = await fetch("https://stock-prediction-46fi.onrender.com/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ stock: finalStock }),
      });

      const result = await res.json();

      if (result.predicted_price) {
        setData(result);
      } else {
        alert(result.error);
      }
    } catch (err) {
      alert("Server error");
    }

    setLoading(false);
  };

  return (
    <div className="app">
      {/* ── HEADER ── */}
      <h1 className="title">
        <div className="title-top">
          <span className="title-text">AI Stock Predictor</span>
          <span className="title-badge">Beta</span>
        </div>
        <div className="title-sub">24BCE532, 33 · Powered by Nirma</div>
      </h1>

      {/* ── MAIN CARD ── */}
      <div className="card">

        {/* ── CONTROLS ── */}
        <div className="controls-section">
          <div className="input-row">
            {/* 🔍 SEARCH INPUT */}
            <input
              type="text"
              placeholder="Enter symbol (e.g. META, SBIN.NS)"
              value={customStock}
              onChange={(e) => setCustomStock(e.target.value.toUpperCase())}
            />

            {/* 🔽 DROPDOWN */}
            <select
              value={selectedStock}
              onChange={(e) => {
                setSelectedStock(e.target.value);
                setCustomStock("");
              }}
            >
              {STOCKS.map((stock) => (
                <option key={stock.symbol} value={stock.symbol}>
                  {stock.name} ({stock.symbol})
                </option>
              ))}
            </select>

            <button onClick={predictStock} disabled={loading}>
              {loading ? "Predicting…" : "Predict Price"}
            </button>
          </div>
        </div>

        {/* ── DASHBOARD ── */}
        {data && (
          <div className="dashboard">
            <div className="card-box">
              <p className="label">Current Price</p>
              <h3>${data.current_price.toFixed(2)}</h3>
            </div>

            <div className="card-box highlight">
              <p className="label">Predicted Price</p>
              <h3>${data.predicted_price.toFixed(2)}</h3>
            </div>

            <div className="card-box">
              <p className="label">Change</p>
              <h3
                style={{
                  color: data.change_percent > 0 ? "var(--success)" : "var(--danger)",
                }}
              >
                {data.change_percent > 0 ? "+" : ""}
                {data.change_percent.toFixed(2)}%
              </h3>
            </div>
          </div>
        )}

        {/* ── CHART ── */}
        <div className="chart-section">
          <div className="chart-header">
            <span className="chart-label">Live Chart — {finalStock}</span>
            <span className="chart-dot" />
          </div>

          <div className="chart-container">
            <iframe
              title="stock-chart"
              src={`https://s.tradingview.com/widgetembed/?symbol=${getChartSymbol()}&interval=D&hidesidetoolbar=1&theme=dark`}
              width="100%"
              height="500"
              frameBorder="0"
            />
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;