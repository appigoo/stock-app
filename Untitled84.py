#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt

# 網頁標題與設定
st.title("📈 股票技術分析助手")
symbol = st.text_input("輸入股票代號（如 TSLA）", "TSLA")
interval = st.selectbox("選擇時間週期", ["1m", "15m", "1h", "1d"])
if st.button("開始分析"):

    st.write(f"🔍 正在取得 {symbol} 的 {interval} 資料...")
    data = yf.download(tickers=symbol, period="7d", interval=interval)

    # 技術指標分析
    data["RSI"] = ta.rsi(data["Close"])
    macd = ta.macd(data["Close"])
    data["MACD"] = macd["MACD_12_26_9"]
    data["MACD_signal"] = macd["MACDs_12_26_9"]

    # 畫圖
    fig, ax = plt.subplots()
    ax.plot(data["Close"], label="Close")
    ax.plot(data["RSI"], label="RSI", linestyle="--")
    ax.set_title(f"{symbol} 股價與 RSI")
    ax.legend()
    st.pyplot(fig)

    # 策略建議
    latest_rsi = data["RSI"].dropna().iloc[-1]
    latest_macd = data["MACD"].dropna().iloc[-1]
    latest_signal = data["MACD_signal"].dropna().iloc[-1]

    if latest_rsi > 70:
        st.warning("⚠️ RSI 顯示超買，可能回調")
    elif latest_rsi < 30:
        st.success("📈 RSI 顯示超賣，可能反彈")
    else:
        st.info("📊 RSI 中性")

    if latest_macd > latest_signal:
        st.success("✅ MACD 顯示買入動能")
    else:
        st.error("❌ MACD 顯示賣出訊號")


# In[ ]:




