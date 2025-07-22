import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.font_manager as fm
import streamlit as st
import re # For splitting ticker input

###
# --- Configuration Defaults ---
DEFAULT_MFI_PERIOD = 14
DEFAULT_RSI_PERIOD = 14
DEFAULT_OVERBOUGHT = 80
DEFAULT_OVERSOLD = 20
DEFAULT_MA_SHORT = 5
DEFAULT_MA_MEDIUM = 20
DEFAULT_MA_LONG = 50
DEFAULT_MA_VERY_LONG = 200
DEFAULT_BB_LENGTH = 20
DEFAULT_BB_STD = 2.0

# Configure matplotlib to use a font that supports Chinese characters
# Ensure 'Noto Sans CJK JP' or another suitable CJK font is installed on your system.
try:
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Arial Unicode MS'] # Add a fallback font
    plt.rcParams['axes.unicode_minus'] = False # Correctly display minus signs
except Exception as e:
    st.warning(f"Warning: Could not set Chinese font for Matplotlib. Charts may not display Chinese characters correctly. Error: {e}")
    # Fallback to default font settings
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


# --- Data Acquisition ---
@st.cache_data(ttl=3600) # Cache data for 1 hour to avoid repeated downloads
def get_stock_data(ticker, period="1y"):
    """
    Fetches historical stock data from Yahoo Finance.
    Args:
        ticker (str): Stock ticker symbol.
        period (str): Data period (e.g., "1y", "5y", "max").
    Returns:
        pd.DataFrame: Historical stock data, or None if an error occurs.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            st.error(f"無法獲取 {ticker} 的數據，請檢查股票代碼或時間範圍。")
            return None
        # Ensure columns are in the expected order
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    except Exception as e:
        st.error(f"獲取 {ticker} 數據時發生錯誤：{e}")
        return None

# --- Indicator Calculation Functions (Modularized) ---

def calculate_mfi(df, period=DEFAULT_MFI_PERIOD):
    """Calculates Money Flow Index (MFI)."""
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Money_Flow'] = df['Typical_Price'] * df['Volume']
    df['Price_Change'] = df['Typical_Price'].diff()
    df['Positive_MF'] = np.where(df['Price_Change'] > 0, df['Money_Flow'], 0)
    df['Negative_MF'] = np.where(df['Price_Change'] < 0, df['Money_Flow'], 0)
    positive_mf_sum = df['Positive_MF'].rolling(window=period).sum()
    negative_mf_sum = df['Negative_MF'].rolling(window=period).sum()
    # Add a small epsilon to the denominator to avoid division by zero
    df['Money_Ratio'] = positive_mf_sum / (negative_mf_sum + 1e-9)
    df[f'MFI_{period}'] = 100 - (100 / (1 + df['Money_Ratio']))
    return df

def calculate_rsi(df, period=DEFAULT_RSI_PERIOD):
    """Calculates Relative Strength Index (RSI)."""
    df[f'RSI_{period}'] = ta.rsi(df['Close'], length=period)
    return df

def calculate_macd(df):
    """Calculates Moving Average Convergence Divergence (MACD)."""
    macd_results = ta.macd(df['Close'])
    df['MACD'] = macd_results['MACD_12_26_9']
    df['MACD_Signal'] = macd_results['MACDs_12_26_9']
    df['MACD_Hist'] = macd_results['MACDh_12_26_9'] # Renamed for clarity
    return df

def calculate_support_resistance(df, window=20):
    """Calculates rolling support and resistance levels."""
    df['Support'] = df['Low'].rolling(window=window).min()
    df['Resistance'] = df['High'].rolling(window=window).max()
    return df

def calculate_price_volume_changes(df):
    """Calculates daily and rolling average price and volume changes."""
    df['Price_Change_Percent'] = df['Close'].pct_change()
    df['Volume_Change_Percent'] = df['Volume'].pct_change()
    df['Abs_Price_Change_Percent'] = df['Price_Change_Percent'].abs()
    df['Avg_Abs_Price_Change_Percent_5d'] = df['Abs_Price_Change_Percent'].rolling(window=5).mean()
    df['Avg_Volume_Change_Percent_5d'] = df['Volume_Change_Percent'].rolling(window=5).mean()
    return df

def calculate_moving_averages(df, periods=[DEFAULT_MA_SHORT, DEFAULT_MA_MEDIUM, DEFAULT_MA_LONG, DEFAULT_MA_VERY_LONG]):
    """Calculates specified Simple Moving Averages."""
    for p in periods:
        df[f'MA{p}'] = ta.sma(df['Close'], length=p)
    return df

def calculate_bollinger_bands(df, length=DEFAULT_BB_LENGTH, std=DEFAULT_BB_STD):
    """Calculates Bollinger Bands."""
    bollinger = ta.bbands(df['Close'], length=length, std=std)
    df['BBL'] = bollinger[f'BBL_{length}_{std}']
    df['BBM'] = bollinger[f'BBM_{length}_{std}']
    df['BBU'] = bollinger[f'BBU_{length}_{std}']
    return df

def calculate_all_indicators(df, mfi_period, rsi_period, ma_periods, bb_length, bb_std):
    """Combines all indicator calculations."""
    df = calculate_mfi(df.copy(), mfi_period)
    df = calculate_rsi(df.copy(), rsi_period)
    df = calculate_macd(df.copy())
    df = calculate_support_resistance(df.copy()) # Default window 20
    df = calculate_price_volume_changes(df.copy())
    df = calculate_moving_averages(df.copy(), periods=ma_periods)
    df = calculate_bollinger_bands(df.copy(), length=bb_length, std=bb_std)
    return df

# --- Divergence and Volume Analysis ---

def detect_divergence(df, mfi_period):
    """
    Detects MFI bullish and bearish divergences.
    Requires 'Close' and 'MFI_{mfi_period}' columns.
    """
    if f'MFI_{mfi_period}' not in df.columns:
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

    # Use rolling min/max for price and MFI over a small window to detect local extremes
    window_div = 5 # Window for detecting local lows/highs for divergence
    price_lows = df['Close'].rolling(window=window_div).min().dropna()
    mfi_lows = df[f'MFI_{mfi_period}'].rolling(window=window_div).min().dropna()
    price_highs = df['Close'].rolling(window=window_div).max().dropna()
    mfi_highs = df[f'MFI_{mfi_period}'].rolling(window=window_div).max().dropna()

    # Align indices for comparison
    common_index_low = price_lows.index.intersection(mfi_lows.index)
    common_index_high = price_highs.index.intersection(mfi_highs.index)

    bullish_div = pd.Series(False, index=df.index)
    bearish_div = pd.Series(False, index=df.index)

    # Check for divergence on the common indices
    if not common_index_low.empty:
        # Bullish Divergence: Price makes lower low, MFI makes higher low
        bullish_div.loc[common_index_low] = (price_lows.loc[common_index_low] < price_lows.loc[common_index_low].shift(1)) & \
                                            (mfi_lows.loc[common_index_low] > mfi_lows.loc[common_index_low].shift(1))

    if not common_index_high.empty:
        # Bearish Divergence: Price makes higher high, MFI makes lower high
        bearish_div.loc[common_index_high] = (price_highs.loc[common_index_high] > price_highs.loc[common_index_high].shift(1)) & \
                                             (mfi_highs.loc[common_index_high] < mfi_highs.loc[common_index_high].shift(1))

    return bullish_div, bearish_div

def analyze_volume_trend(df, window=10):
    """
    Analyzes the current volume trend compared to its recent average.
    """
    if 'Volume' not in df.columns or df['Volume'].empty:
        return "未知"
    avg_volume = df['Volume'].rolling(window=window).mean().dropna()
    if not avg_volume.empty:
        # Get the latest non-NaN average volume
        latest_avg_volume = avg_volume.iloc[-1]
        if pd.notna(df['Volume'].iloc[-1]) and latest_avg_volume > 1e-9: # Avoid division by zero
            if df['Volume'].iloc[-1] > latest_avg_volume * 1.2: # Significantly higher
                return "顯著放大"
            elif df['Volume'].iloc[-1] > latest_avg_volume: # Slightly higher
                return "放大"
            elif df['Volume'].iloc[-1] < latest_avg_volume * 0.8: # Significantly lower
                return "顯著縮減"
            elif df['Volume'].iloc[-1] < latest_avg_volume: # Slightly lower
                return "縮減"
            else:
                return "持平"
    return "未知"

# --- Recommendation Logic ---

def generate_recommendation(df, mfi_period, rsi_period, overbought, oversold):
    """
    Generates a stock recommendation based on various technical indicators.
    Returns recommendation, reasons, target price, stop loss, and ratios.
    """
    df_cleaned = df.dropna().copy()
    if df_cleaned.empty or len(df_cleaned) < 2:
        return "無法生成建議", ["數據不足以計算指標或判斷趨勢"], None, None, None, None

    latest = df_cleaned.iloc[-1]
    prev = df_cleaned.iloc[-2]

    bullish_div, bearish_div = detect_divergence(df_cleaned, mfi_period)
    volume_trend = analyze_volume_trend(df_cleaned)

    recommendation = "持有"
    reasons = []
    
    # Default target and stop loss (can be refined)
    target_price = latest['Resistance'] if pd.notna(latest.get('Resistance')) and latest['Close'] < latest['Resistance'] else latest['Close'] * 1.05
    stop_loss = latest['Support'] if pd.notna(latest.get('Support')) and latest['Close'] > latest['Support'] else latest['Close'] * 0.95

    # Access calculated values with .get() for safety
    mfi_value = latest.get(f'MFI_{mfi_period}')
    rsi_value = latest.get(f'RSI_{rsi_period}')
    macd_value = latest.get('MACD')
    macd_signal_value = latest.get('MACD_Signal')
    support_value = latest.get('Support')
    resistance_value = latest.get('Resistance')
    
    abs_price_change_percent = latest.get('Abs_Price_Change_Percent')
    avg_abs_price_change_percent_5d = latest.get('Avg_Abs_Price_Change_Percent_5d')
    volume_change_percent = latest.get('Volume_Change_Percent')
    avg_volume_change_percent_5d = latest.get('Avg_Volume_Change_Percent_5d')

    ma5 = latest.get('MA5')
    ma20 = latest.get('MA20')
    ma50 = latest.get('MA50')
    ma200 = latest.get('MA200')

    prev_ma50 = prev.get('MA50')
    prev_ma200 = prev.get('MA200')

    bb_middle = latest.get('BBM')
    bb_upper = latest.get('BBU')
    bb_lower = latest.get('BBL')
    prev_bb_upper = prev.get('BBU') # For breakout detection

    # Calculate ratios, handling potential division by zero
    price_change_ratio = None
    if pd.notna(abs_price_change_percent) and pd.notna(avg_abs_price_change_percent_5d):
        if abs(avg_abs_price_change_percent_5d) > 1e-9:
            price_change_ratio = abs_price_change_percent / avg_abs_price_change_percent_5d
        elif abs_price_change_percent < 1e-9:
            price_change_ratio = 1.0
        else:
            price_change_ratio = np.inf

    volume_change_percent_ratio = None
    if pd.notna(volume_change_percent) and pd.notna(avg_volume_change_percent_5d):
        if abs(avg_volume_change_percent_5d) > 1e-9:
            volume_change_percent_ratio = volume_change_percent / avg_volume_change_percent_5d
        elif abs(volume_change_percent) < 1e-9:
            volume_change_percent_ratio = 1.0
        else:
            volume_change_percent_ratio = np.inf * np.sign(volume_change_percent)

    bullish_score = 0
    bearish_score = 0
    added_reasons = set()

    def add_reason(score_type, reason_text, score_points=1):
        nonlocal bullish_score, bearish_score # Declare as nonlocal to modify outer scope variables
        if reason_text not in added_reasons:
            if score_type == "bullish":
                bullish_score += score_points
            elif score_type == "bearish":
                bearish_score += score_points
            reasons.append(reason_text)
            added_reasons.add(reason_text)

    # --- Bullish Signals ---
    if pd.notna(mfi_value) and mfi_value < oversold:
        add_reason("bullish", f"MFI ({mfi_value:.2f}) 進入超賣區（<{oversold}），可能表示價格被低估，存在反彈機會。", 1)
    if pd.notna(rsi_value) and rsi_value < 30: # RSI 30 for strong buy
        add_reason("bullish", f"RSI ({rsi_value:.2f}) 進入超賣區（<30），可能表示價格超跌，存在反彈機會。", 1)
    if pd.notna(macd_value) and pd.notna(macd_signal_value) and macd_value > macd_signal_value and prev.get('MACD', -np.inf) <= prev.get('MACD_Signal', -np.inf):
        add_reason("bullish", "MACD金叉（MACD線在訊號線上），顯示上漲動能增強。", 1.5) # Higher weight for crossover
    if pd.notna(support_value) and latest['Close'] > support_value and (latest['Close'] - support_value) / support_value < 0.02:
        add_reason("bullish", f"價格接近支撐位 ({support_value:.2f})，可能有反彈機會。", 1)
    if not bullish_div.empty and latest.name in bullish_div.index and bullish_div.loc[latest.name]:
        add_reason("bullish", "看漲背離：價格創低但MFI未創新低，預示反彈機會。", 2) # Stronger signal

    if price_change_ratio is not None and volume_change_percent_ratio is not None:
        if price_change_ratio > 2.0 and volume_change_percent_ratio > 2.0:
            add_reason("bullish", f"今日絕對價格變化百分比 ({abs_price_change_percent:.2%}) 和成交量變化百分比 ({volume_change_percent:.2%}) 均超過5日平均的兩倍，顯示強勁買盤動能。", 2.5)

    if pd.notna(ma50) and pd.notna(ma200) and pd.notna(prev_ma50) and pd.notna(prev_ma200):
        if ma50 > ma200 and prev_ma50 <= prev_ma200 and volume_trend in ['放大', '顯著放大']:
            add_reason("bullish", "MA50金叉MA200，且成交量放大，顯示長期趨勢轉強，資金大幅流入。", 3) # Very strong

    if pd.notna(latest['Close']) and pd.notna(bb_middle) and pd.notna(bb_upper) and pd.notna(prev['Close']):
        # Price crossing above middle band
        if latest['Close'] > bb_middle and prev['Close'] <= bb_middle and latest['Close'] < bb_upper:
            add_reason("bullish", "股價突破布林通道中軸，預示上漲動能可能持續。", 1.5)
        # Price bouncing off lower band
        if latest['Close'] > bb_lower and prev['Close'] <= bb_lower:
            add_reason("bullish", "股價從布林通道下軸反彈，可能預示短期築底。", 1.5)


    # --- Bearish Signals ---
    if pd.notna(mfi_value) and mfi_value > overbought:
        add_reason("bearish", f"MFI ({mfi_value:.2f}) 進入超買區（>{overbought}），價格可能過熱，存在回調風險。", 1)
    if pd.notna(rsi_value) and rsi_value > 70: # RSI 70 for strong sell
        add_reason("bearish", f"RSI ({rsi_value:.2f}) 進入超買區（>70），股價可能超漲，有回調風險。", 1)
    if pd.notna(macd_value) and pd.notna(macd_signal_value) and macd_value < macd_signal_value and prev.get('MACD', np.inf) >= prev.get('MACD_Signal', np.inf):
        add_reason("bearish", "MACD死叉（MACD線在訊號線下），顯示下跌動能減弱。", 1.5)
    if pd.notna(resistance_value) and latest['Close'] < resistance_value and (resistance_value - latest['Close']) / resistance_value < 0.02:
        add_reason("bearish", f"價格接近阻力位 ({resistance_value:.2f})，可能遇賣壓。", 1)
    if not bearish_div.empty and latest.name in bearish_div.index and bearish_div.loc[latest.name]:
        add_reason("bearish", "看跌背離：價格創高但MFI未創新高，預示回落風險。", 2)

    if price_change_ratio is not None and volume_change_percent_ratio is not None:
        if price_change_ratio < 0.5 and abs(volume_change_percent_ratio) < 0.5:
            add_reason("bearish", f"今日絕對價格變化百分比 ({abs_price_change_percent:.2%}) 和成交量變化百分比 ({volume_change_percent:.2%}) 均少於5日平均的一半，顯示市場動能顯著減弱或賣壓。", 2.5)

    if pd.notna(ma50) and pd.notna(ma200) and pd.notna(prev_ma50) and pd.notna(prev_ma200):
        if ma50 < ma200 and prev_ma50 >= prev_ma200 and volume_trend in ['放大', '顯著放大']:
            add_reason("bearish", "MA50死叉MA200，且成交量放大，顯示長期趨勢轉弱，資金大幅流出。", 3)

    if pd.notna(latest['Close']) and pd.notna(bb_upper) and pd.notna(prev['Close']):
        # Price breaking above upper band and pulling back
        if latest['Close'] < prev['Close'] and prev['Close'] > prev_bb_upper and latest['Close'] <= bb_upper:
            add_reason("bearish", "股價突破布林通道上軸後出現回調，可能遇強勁賣壓。", 2)
        # Price breaking below middle band
        if latest['Close'] < bb_middle and prev['Close'] >= bb_middle and latest['Close'] > bb_lower:
            add_reason("bearish", "股價跌破布林通道中軸，預示下跌動能可能持續。", 1.5)


    # Final recommendation logic based on scores, with strong signals having priority
    if bullish_score > bearish_score + 1.5: # Require a clearer bullish advantage
        recommendation = "買入"
    elif bearish_score > bullish_score + 1.5: # Require a clearer bearish advantage
        recommendation = "賣出"
    else:
        recommendation = "持有"

    # Refine reasons to match the final recommendation
    final_reasons = []
    # Prioritize reasons that strongly align with the final recommendation
    if recommendation == "買入":
        if any("超過5日平均的兩倍" in r for r in reasons): final_reasons.append("基於今日價格及成交量相較於5日平均的顯著放大，顯示強勁買盤動能。")
        if any("MA50金叉MA200" in r for r in reasons): final_reasons.append("MA50金叉MA200，且成交量放大，顯示長期趨勢轉強，資金大幅流入。")
        if any("突破布林通道中軸" in r for r in reasons): final_reasons.append("股價突破布林通道中軸，預示上漲動能可能持續。")
        if any("布林通道下軸反彈" in r for r in reasons): final_reasons.append("股價從布林通道下軸反彈，可能預示短期築底。")
        if any("MFI" in r and "超賣區" in r for r in reasons): final_reasons.append(f"MFI ({mfi_value:.2f}) 進入超賣區，存在反彈機會。")
        if any("RSI" in r and "超賣區" in r for r in reasons): final_reasons.append(f"RSI ({rsi_value:.2f}) 進入超賣區，存在反彈機會。")
        if any("MACD金叉" in r for r in reasons): final_reasons.append("MACD金叉，顯示上漲動能增強。")
        if any("支撐位" in r for r in reasons): final_reasons.append(f"價格接近支撐位 ({support_value:.2f})。")
        if any("看漲背離" in r for r in reasons): final_reasons.append("看漲背離：價格創低但MFI未創新低。")

    elif recommendation == "賣出":
        if any("少於5日平均的一半" in r for r in reasons): final_reasons.append("基於今日價格及成交量相較於5日平均的顯著縮減，顯示市場動能減弱或賣壓。")
        if any("MA50死叉MA200" in r for r in reasons): final_reasons.append("MA50死叉MA200，且成交量放大，顯示長期趨勢轉弱，資金大幅流出。")
        if any("突破布林通道上軸後出現回調" in r for r in reasons): final_reasons.append("股價突破布林通道上軸後出現回調，可能遇強勁賣壓。")
        if any("跌破布林通道中軸" in r for r in reasons): final_reasons.append("股價跌破布林通道中軸，預示下跌動能可能持續。")
        if any("MFI" in r and "超買區" in r for r in reasons): final_reasons.append(f"MFI ({mfi_value:.2f}) 進入超買區，存在回調風險。")
        if any("RSI" in r and "超買區" in r for r in reasons): final_reasons.append(f"RSI ({rsi_value:.2f}) 進入超買區，有回調風險。")
        if any("MACD死叉" in r for r in reasons): final_reasons.append("MACD死叉，顯示下跌動能減弱。")
        if any("阻力位" in r for r in reasons): final_reasons.append(f"價格接近阻力位 ({resistance_value:.2f})。")
        if any("看跌背離" in r for r in reasons): final_reasons.append("看跌背離：價格創高但MFI未創新高。")
    else: # Hold
        if len(reasons) > 0:
            final_reasons.extend([r for r in reasons if "多空訊號混雜" not in r])
        if not final_reasons:
            final_reasons.append("無明確技術訊號，建議觀望。")
        else:
            final_reasons.append("多空訊號混雜，建議觀望。")

    # Ensure volume trend reason is included if relevant and not already covered by ratio/MA cross
    current_volume_trend_reason = f"成交量趨勢：{volume_trend}。"
    if volume_trend != '未知' and current_volume_trend_reason not in final_reasons:
        final_reasons.append(current_volume_trend_reason)

    return recommendation, list(set(final_reasons)), target_price, stop_loss, price_change_ratio, volume_change_percent_ratio

# --- Plotting ---

def plot_data(df, ticker, mfi_period, rsi_period, overbought, oversold, ma_periods, bb_length, bb_std):
    """
    Plots stock price, MFI, and Volume data with indicators.
    """
    df_cleaned = df.dropna().copy()
    if df_cleaned.empty:
        st.warning("數據不足，無法繪製圖表。")
        return None

    index_np = df_cleaned.index.to_numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Price, MA, S/R, Bollinger Bands
    ax1.plot(index_np, df_cleaned['Close'].to_numpy(), label='收盤價', color='blue')
    if 'Support' in df_cleaned.columns and pd.notna(df_cleaned['Support']).any():
        ax1.plot(index_np, df_cleaned['Support'].to_numpy(), label='支撐位', color='green', linestyle='--')
    if 'Resistance' in df_cleaned.columns and pd.notna(df_cleaned['Resistance']).any():
        ax1.plot(index_np, df_cleaned['Resistance'].to_numpy(), label='阻力位', color='red', linestyle='--')

    for p in ma_periods:
        if f'MA{p}' in df_cleaned.columns and pd.notna(df_cleaned[f'MA{p}']).any():
            ax1.plot(index_np, df_cleaned[f'MA{p}'].to_numpy(), label=f'MA{p}', linestyle='-')

    if 'BBM' in df_cleaned.columns and pd.notna(df_cleaned['BBM']).any():
        ax1.plot(index_np, df_cleaned['BBM'].to_numpy(), label=f'布林中軌 (MA{bb_length})', color='gray', linestyle='-')
    if 'BBU' in df_cleaned.columns and pd.notna(df_cleaned['BBU']).any():
        ax1.plot(index_np, df_cleaned['BBU'].to_numpy(), label='布林上軌', color='red', linestyle=':')
    if 'BBL' in df_cleaned.columns and pd.notna(df_cleaned['BBL']).any():
        ax1.plot(index_np, df_cleaned['BBL'].to_numpy(), label='布林下軌', color='green', linestyle=':')

    ax1.set_title(f'{ticker} 股價與技術指標')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)


    # Plot 2: MFI and RSI
    if f'MFI_{mfi_period}' in df_cleaned.columns and pd.notna(df_cleaned[f'MFI_{mfi_period}']).any():
        ax2.plot(index_np, df_cleaned[f'MFI_{mfi_period}'].to_numpy(), label=f'MFI ({mfi_period})', color='purple')
        ax2.axhline(overbought, color='red', linestyle='--', label=f'MFI超買 ({overbought})')
        ax2.axhline(oversold, color='green', linestyle='--', label=f'MFI超賣 ({oversold})')

    if f'RSI_{rsi_period}' in df_cleaned.columns and pd.notna(df_cleaned[f'RSI_{rsi_period}']).any():
        ax2.plot(index_np, df_cleaned[f'RSI_{rsi_period}'].to_numpy(), label=f'RSI ({rsi_period})', color='darkorange')
        ax2.axhline(70, color='red', linestyle=':', label='RSI超買 (70)')
        ax2.axhline(30, color='green', linestyle=':', label='RSI超賣 (30)')

    ax2.set_title('MFI 與 RSI 指標')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Volume and MACD
    ax3.bar(index_np, df_cleaned['Volume'].to_numpy(), label='成交量', color='gray', alpha=0.7)
    if 'MACD' in df_cleaned.columns and pd.notna(df_cleaned['MACD']).any():
        ax3_macd = ax3.twinx() # Create a second y-axis for MACD
        ax3_macd.plot(index_np, df_cleaned['MACD'].to_numpy(), label='MACD', color='blue', linestyle='-')
        ax3_macd.plot(index_np, df_cleaned['MACD_Signal'].to_numpy(), label='MACD訊號線', color='red', linestyle='--')
        
        # Plot MACD Histogram
        macd_hist_colors = ['green' if x > 0 else 'red' for x in df_cleaned['MACD_Hist'].to_numpy()]
        ax3_macd.bar(index_np, df_cleaned['MACD_Hist'].to_numpy(), label='MACD柱狀圖', color=macd_hist_colors, alpha=0.3)
        ax3_macd.axhline(0, color='black', linestyle='--', linewidth=0.5) # Zero line for MACD hist
        ax3_macd.set_ylabel('MACD Values')
        ax3_macd.legend(loc='upper left')

    ax3.set_title('成交量 與 MACD 指標')
    ax3.set_xlabel('日期')
    ax3.set_ylabel('成交量')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.6)


    plt.tight_layout()
    return fig

# --- Backtesting Function (Simple Strategy) ---
def run_backtest(df, strategy_type="MFI", entry_level=20, exit_level=80, mfi_period=14, rsi_period=14):
    """
    Runs a simple backtesting strategy.
    Args:
        df (pd.DataFrame): Stock data with indicators.
        strategy_type (str): "MFI" or "MACD" or "RSI".
        entry_level (int): Entry threshold for MFI/RSI (e.g., 20 for MFI, 30 for RSI).
        exit_level (int): Exit threshold for MFI/RSI (e.g., 80 for MFI, 70 for RSI).
        mfi_period (int): MFI period.
        rsi_period (int): RSI period.
    Returns:
        dict: Backtest results (total return, trades, etc.)
    """
    df_bt = df.copy().dropna()
    if df_bt.empty:
        return {"total_return": 0, "num_trades": 0, "max_drawdown": 0, "profit_trades": 0, "loss_trades": 0}

    # Initialize portfolio
    initial_cash = 100000
    cash = initial_cash
    shares = 0
    trade_log = []
    
    # Track daily portfolio value
    portfolio_value = pd.Series(index=df_bt.index, dtype=float)
    
    # Ensure indicator columns exist
    if strategy_type == "MFI" and f'MFI_{mfi_period}' not in df_bt.columns:
        st.warning(f"MFI_{mfi_period} column not found for MFI strategy backtest.")
        return {"total_return": 0, "num_trades": 0, "max_drawdown": 0, "profit_trades": 0, "loss_trades": 0}
    elif strategy_type == "RSI" and f'RSI_{rsi_period}' not in df_bt.columns:
        st.warning(f"RSI_{rsi_period} column not found for RSI strategy backtest.")
        return {"total_return": 0, "num_trades": 0, "max_drawdown": 0, "profit_trades": 0, "loss_trades": 0}
    elif strategy_type == "MACD" and ('MACD' not in df_bt.columns or 'MACD_Signal' not in df_bt.columns):
        st.warning("MACD or MACD_Signal columns not found for MACD strategy backtest.")
        return {"total_return": 0, "num_trades": 0, "max_drawdown": 0, "profit_trades": 0, "loss_trades": 0}

    for i in range(1, len(df_bt)):
        current_close = df_bt['Close'].iloc[i]
        prev_close = df_bt['Close'].iloc[i-1]

        # Calculate current portfolio value
        portfolio_value.iloc[i] = cash + shares * current_close

        signal = "HOLD"

        if strategy_type == "MFI":
            current_indicator = df_bt[f'MFI_{mfi_period}'].iloc[i]
            prev_indicator = df_bt[f'MFI_{mfi_period}'].iloc[i-1]
            if pd.notna(current_indicator) and pd.notna(prev_indicator):
                # Buy signal: MFI crosses above entry_level (e.g., 20) from below
                if current_indicator > entry_level and prev_indicator <= entry_level and shares == 0:
                    signal = "BUY"
                # Sell signal: MFI crosses below exit_level (e.g., 80) from above
                elif current_indicator < exit_level and prev_indicator >= exit_level and shares > 0:
                    signal = "SELL"
        
        elif strategy_type == "RSI":
            current_indicator = df_bt[f'RSI_{rsi_period}'].iloc[i]
            prev_indicator = df_bt[f'RSI_{rsi_period}'].iloc[i-1]
            if pd.notna(current_indicator) and pd.notna(prev_indicator):
                # Buy signal: RSI crosses above entry_level (e.g., 30) from below
                if current_indicator > entry_level and prev_indicator <= entry_level and shares == 0:
                    signal = "BUY"
                # Sell signal: RSI crosses below exit_level (e.g., 70) from above
                elif current_indicator < exit_level and prev_indicator >= exit_level and shares > 0:
                    signal = "SELL"

        elif strategy_type == "MACD":
            current_macd = df_bt['MACD'].iloc[i]
            current_signal = df_bt['MACD_Signal'].iloc[i]
            prev_macd = df_bt['MACD'].iloc[i-1]
            prev_signal = df_bt['MACD_Signal'].iloc[i-1]
            if pd.notna(current_macd) and pd.notna(current_signal) and pd.notna(prev_macd) and pd.notna(prev_signal):
                # Buy signal: MACD crosses above Signal Line (Golden Cross)
                if current_macd > current_signal and prev_macd <= prev_signal and shares == 0:
                    signal = "BUY"
                # Sell signal: MACD crosses below Signal Line (Death Cross)
                elif current_macd < current_signal and prev_macd >= prev_signal and shares > 0:
                    signal = "SELL"


        if signal == "BUY":
            if cash > current_close: # Ensure enough cash
                num_shares_to_buy = int(cash / current_close)
                if num_shares_to_buy > 0:
                    shares += num_shares_to_buy
                    cash -= num_shares_to_buy * current_close
                    trade_log.append({'date': df_bt.index[i], 'type': 'BUY', 'price': current_close, 'shares': num_shares_to_buy})
        elif signal == "SELL":
            if shares > 0:
                cash += shares * current_close
                trade_log.append({'date': df_bt.index[i], 'type': 'SELL', 'price': current_close, 'shares': shares})
                shares = 0

    # Finalize portfolio value on the last day
    final_value = cash + shares * df_bt['Close'].iloc[-1]
    total_return = (final_value - initial_cash) / initial_cash * 100 if initial_cash > 0 else 0

    # Calculate max drawdown
    portfolio_value = portfolio_value.dropna()
    if not portfolio_value.empty:
        peak = portfolio_value.expanding(min_periods=1).max()
        drawdown = (portfolio_value - peak) / peak
        max_drawdown = drawdown.min() * 100
    else:
        max_drawdown = 0

    # Count profit/loss trades
    profit_trades = 0
    loss_trades = 0
    
    # Simplified trade profit/loss calculation (assumes each buy is eventually closed by a sell)
    # This is a simplification; a more robust backtest would pair trades
    buy_prices = []
    for trade in trade_log:
        if trade['type'] == 'BUY':
            buy_prices.append(trade['price'])
        elif trade['type'] == 'SELL' and buy_prices:
            # Assuming the sell closes the most recent buy position
            if trade['price'] > buy_prices[0]: # Simple check
                profit_trades += 1
            else:
                loss_trades += 1
            buy_prices.pop(0) # Remove the "closed" buy price

    return {
        "total_return": total_return,
        "num_trades": len(trade_log),
        "max_drawdown": max_drawdown,
        "profit_trades": profit_trades,
        "loss_trades": loss_trades,
        "portfolio_value_history": portfolio_value # For plotting
    }

# --- Streamlit App ---
st.set_page_config(page_title="股票技術分析儀表板", layout="wide", initial_sidebar_state="expanded")

st.title("股票技術分析儀表板")
st.markdown("輸入股票代碼（可多個，用逗號分隔）以獲取其技術分析報告和圖表。")

# --- Sidebar for Inputs ---
st.sidebar.header("配置")

ticker_input = st.sidebar.text_input("請輸入股票代碼 (例如：TSLA, AAPL, NVDA)", "TSLA").upper()
# Split by comma and clean up spaces
tickers = [t.strip() for t in re.split(r'[,;]', ticker_input) if t.strip()]

period_options = {
    "1 年": "1y", "2 年": "2y", "5 年": "5y", "10 年": "10y", "最大": "max"
}
selected_period_label = st.sidebar.selectbox("選擇數據時間範圍", list(period_options.keys()))
period = period_options[selected_period_label]

st.sidebar.subheader("指標參數設定")
mfi_period_input = st.sidebar.slider("MFI 週期", 7, 30, DEFAULT_MFI_PERIOD)
mfi_overbought_input = st.sidebar.slider("MFI 超買閾值", 70, 90, DEFAULT_OVERBOUGHT)
mfi_oversold_input = st.sidebar.slider("MFI 超賣閾值", 10, 30, DEFAULT_OVERSOLD)

rsi_period_input = st.sidebar.slider("RSI 週期", 7, 30, DEFAULT_RSI_PERIOD)

ma_periods_input = st.sidebar.multiselect(
    "選擇移動平均線週期",
    options=[5, 10, 20, 50, 100, 200],
    default=[DEFAULT_MA_SHORT, DEFAULT_MA_MEDIUM, DEFAULT_MA_LONG, DEFAULT_MA_VERY_LONG]
)
ma_periods_input.sort() # Keep them sorted

bb_length_input = st.sidebar.slider("布林帶週期 (天)", 10, 50, DEFAULT_BB_LENGTH)
bb_std_input = st.sidebar.slider("布林帶標準差倍數", 1.0, 3.0, DEFAULT_BB_STD, 0.1)


# --- Main Content ---
if st.button("分析股票"):
    if not tickers:
        st.warning("請輸入至少一個股票代碼。")
    else:
        for ticker in tickers:
            st.markdown(f"---") # Separator for multiple stocks
            st.subheader(f"🔍 正在分析：{ticker}")

            with st.spinner(f"正在獲取和分析 {ticker} 的數據..."):
                df = get_stock_data(ticker, period=period)

                if df is not None:
                    # Calculate all indicators with user-defined parameters
                    df = calculate_all_indicators(df, mfi_period_input, rsi_period_input, ma_periods_input, bb_length_input, bb_std_input)

                    df_cleaned = df.dropna().copy()
                    if not df_cleaned.empty:
                        recommendation, reasons, target_price, stop_loss, price_change_ratio, volume_change_percent_ratio = \
                            generate_recommendation(df_cleaned, mfi_period_input, rsi_period_input, mfi_overbought_input, mfi_oversold_input)

                        st.subheader(f"📊 {ticker} 每日分析報告 ({datetime.now().strftime('%Y-%m-%d')})")
                        latest = df_cleaned.iloc[-1]

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("最新收盤價", f"${latest['Close']:.2f}")
                        with col2:
                            st.metric("投資建議", recommendation)
                        with col3:
                            if target_price is not None:
                                st.metric("目標價", f"${target_price:.2f}")
                            if stop_loss is not None:
                                st.metric("止損價", f"${stop_loss:.2f}")

                        st.markdown("### 詳細技術指標")

                        # MFI
                        mfi_value = latest.get(f'MFI_{mfi_period_input}')
                        mfi_explanation = f"**MFI ({mfi_period_input}天):** {mfi_value:.2f}" if pd.notna(mfi_value) else "**MFI:** 無法計算"
                        if pd.notna(mfi_value):
                            if mfi_value > mfi_overbought_input:
                                mfi_explanation += f" (超買區 >{mfi_overbought_input}，可能表示價格過熱，存在回調風險)"
                            elif mfi_value < mfi_oversold_input:
                                mfi_explanation += f" (超賣區 <{mfi_oversold_input}，可能表示價格被低估，存在反彈機會)"
                            else:
                                mfi_explanation += " (中間區間，無明確買賣訊號)"
                        st.write(mfi_explanation)

                        # RSI
                        rsi_value = latest.get(f'RSI_{rsi_period_input}')
                        rsi_explanation = f"**RSI ({rsi_period_input}天):** {rsi_value:.2f}" if pd.notna(rsi_value) else "**RSI:** 無法計算"
                        if pd.notna(rsi_value):
                            if rsi_value > 70:
                                rsi_explanation += " (超買區 >70，股價可能超漲，有回調風險)"
                            elif rsi_value < 30:
                                rsi_explanation += " (超賣區 <30，股價可能超跌，有反彈機會)"
                            else:
                                rsi_explanation += " (中間區域，市場處於均衡狀態)"
                        st.write(rsi_explanation)

                        # MACD
                        macd_value = latest.get('MACD')
                        macd_signal_value = latest.get('MACD_Signal')
                        macd_hist_value = latest.get('MACD_Hist')
                        macd_explanation_parts = []
                        if pd.notna(macd_value) and pd.notna(macd_signal_value):
                            macd_explanation_parts.append(f"MACD: {macd_value:.2f}, 訊號線: {macd_signal_value:.2f}")
                            if macd_value > macd_signal_value:
                                macd_explanation_parts.append("MACD金叉，顯示上漲動能增強。")
                            elif macd_value < macd_signal_value:
                                macd_explanation_parts.append("MACD死叉，顯示下跌動能減弱。")
                            else:
                                macd_explanation_parts.append("MACD線與訊號線交叉或接近，動能可能正在轉變。")
                        if pd.notna(macd_hist_value):
                            macd_explanation_parts.append(f"柱狀圖: {macd_hist_value:.2f} ({'正' if macd_hist_value > 0 else '負'})")
                        if macd_explanation_parts:
                            st.write(f"**MACD:** " + " ".join(macd_explanation_parts))
                        else:
                            st.write("**MACD:** 無法計算")

                        # Support and Resistance
                        support_value = latest.get('Support')
                        resistance_value = latest.get('Resistance')
                        sr_explanation_parts = []
                        if pd.notna(support_value):
                            sr_explanation_parts.append(f"支撐位: ${support_value:.2f}")
                        if pd.notna(resistance_value):
                            sr_explanation_parts.append(f"阻力位: ${resistance_value:.2f}")
                        if sr_explanation_parts:
                            st.write(" **支撐與阻力位:** " + ", ".join(sr_explanation_parts))

                        # Moving Averages
                        ma_explanation_parts = []
                        for p in ma_periods_input:
                            ma_value = latest.get(f'MA{p}')
                            if pd.notna(ma_value):
                                ma_explanation_parts.append(f"MA{p}: ${ma_value:.2f}")
                        if ma_explanation_parts:
                            ma_explanation = "**移動平均線:** " + ", ".join(ma_explanation_parts)
                            if len(ma_periods_input) >= 2:
                                if DEFAULT_MA_SHORT in ma_periods_input and DEFAULT_MA_MEDIUM in ma_periods_input:
                                    if latest.get(f'MA{DEFAULT_MA_SHORT}') > latest.get(f'MA{DEFAULT_MA_MEDIUM}'):
                                        ma_explanation += " (短期MA在長期MA之上，短期趨勢偏強)"
                                    elif latest.get(f'MA{DEFAULT_MA_SHORT}') < latest.get(f'MA{DEFAULT_MA_MEDIUM}'):
                                        ma_explanation += " (短期MA在長期MA之下，短期趨勢偏弱)"
                            st.write(ma_explanation)

                        # Bollinger Bands
                        bb_middle = latest.get('BBM')
                        bb_upper = latest.get('BBU')
                        bb_lower = latest.get('BBL')
                        bb_explanation_parts = []
                        if pd.notna(bb_middle): bb_explanation_parts.append(f"中軌 (MA{bb_length_input}): ${bb_middle:.2f}")
                        if pd.notna(bb_upper): bb_explanation_parts.append(f"上軌: ${bb_upper:.2f}")
                        if pd.notna(bb_lower): bb_explanation_parts.append(f"下軌: ${bb_lower:.2f}")
                        if bb_explanation_parts:
                            st.write("**布林帶:** " + ", ".join(bb_explanation_parts))
                            if pd.notna(latest['Close']):
                                if pd.notna(bb_upper) and latest['Close'] > bb_upper:
                                    st.write(f"  - 股價 ({latest['Close']:.2f}) 突破布林上軌，可能超漲。")
                                elif pd.notna(bb_lower) and latest['Close'] < bb_lower:
                                    st.write(f"  - 股價 ({latest['Close']:.2f}) 跌破布林下軌，可能超跌。")
                                elif pd.notna(bb_middle) and latest['Close'] > bb_middle:
                                    st.write(f"  - 股價 ({latest['Close']:.2f}) 在布林中軌之上，短期偏多。")
                                elif pd.notna(bb_middle) and latest['Close'] < bb_middle:
                                    st.write(f"  - 股價 ({latest['Close']:.2f}) 在布林中軌之下，短期偏空。")

                        st.write(f"**今日收盤價變化百分比:** {latest.get('Price_Change_Percent', 'N/A'):.2%}")
                        st.write(f"**今日成交量變化百分比:** {latest.get('Volume_Change_Percent', 'N/A'):.2%}")
                        st.write(f"**今日絕對價格變化百分比與5日均值比:** {price_change_ratio:.2f}" if price_change_ratio is not None else "**今日絕對價格變化百分比與5日均值比:** 無法計算")
                        st.write(f"**今日成交量變化百分比與5日均值比:** {volume_change_percent_ratio:.2f}" if volume_change_percent_ratio is not None else "**今日成交量變化百分比與5日均值比:** 無法計算")
                        st.write(f"**過去5日平均絕對股價變化百分比:** {latest.get('Avg_Abs_Price_Change_Percent_5d', 'N/A'):.2%}")
                        st.write(f"**過去5日平均成交量變化百分比:** {latest.get('Avg_Volume_Change_Percent_5d', 'N/A'):.2%}")

                        st.markdown("### 投資建議")
                        st.markdown(f"**{recommendation}**")
                        st.write("原因:")
                        for reason in reasons:
                            st.write(f"- {reason}")

                        st.markdown("---")
                        st.subheader("📈 股價走勢與技術指標圖表")
                        fig = plot_data(df_cleaned, ticker, mfi_period_input, rsi_period_input, mfi_overbought_input, mfi_oversold_input, ma_periods_input, bb_length_input, bb_std_input)
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig) # Close the figure to free up memory
                        else:
                            st.warning("無法生成圖表，數據可能不足。")

                        st.markdown("---")
                        st.subheader("🤖 簡單策略歷史回測")
                        st.write("此回測功能僅提供基於MFI、RSI或MACD的簡單交易策略表現預估，不代表未來實際收益。")
                        
                        strategy_choice = st.radio(f"為 {ticker} 選擇回測策略類型:", ["MFI", "RSI", "MACD"], key=f"strategy_{ticker}")
                        
                        entry_exit_col1, entry_exit_col2 = st.columns(2)
                        with entry_exit_col1:
                            if strategy_choice in ["MFI", "RSI"]:
                                entry_level = st.slider(f"{strategy_choice} 買入閾值", 10, 40, 20 if strategy_choice=="MFI" else 30, key=f"entry_{ticker}_{strategy_choice}")
                        with entry_exit_col2:
                            if strategy_choice in ["MFI", "RSI"]:
                                exit_level = st.slider(f"{strategy_choice} 賣出閾值", 60, 90, 80 if strategy_choice=="MFI" else 70, key=f"exit_{ticker}_{strategy_choice}")
                        
                        if st.button(f"執行 {ticker} 回測", key=f"run_backtest_{ticker}"):
                            with st.spinner(f"正在執行 {ticker} 的 {strategy_choice} 策略回測..."):
                                if strategy_choice in ["MFI", "RSI"]:
                                    backtest_results = run_backtest(df_cleaned, strategy_type=strategy_choice, 
                                                                    entry_level=entry_level, exit_level=exit_level, 
                                                                    mfi_period=mfi_period_input, rsi_period=rsi_period_input)
                                else: # MACD
                                    backtest_results = run_backtest(df_cleaned, strategy_type="MACD", 
                                                                    mfi_period=mfi_period_input, rsi_period=rsi_period_input) # MFI/RSI params not used for MACD

                                if backtest_results:
                                    st.markdown(f"**回測結果總結:**")
                                    st.write(f"- **總回報率:** {backtest_results['total_return']:.2f}%")
                                    st.write(f"- **交易次數:** {backtest_results['num_trades']}")
                                    st.write(f"- **最大回撤:** {backtest_results['max_drawdown']:.2f}%")
                                    st.write(f"- **盈利交易次數:** {backtest_results['profit_trades']}")
                                    st.write(f"- **虧損交易次數:** {backtest_results['loss_trades']}")
                                    
                                    # Plot portfolio value history
                                    if not backtest_results['portfolio_value_history'].empty:
                                        fig_bt = plt.figure(figsize=(10, 5))
                                        plt.plot(backtest_results['portfolio_value_history'].index, backtest_results['portfolio_value_history'].to_numpy(), label='投資組合價值', color='blue')
                                        plt.title(f'{ticker} {strategy_choice} 策略投資組合價值變化')
                                        plt.xlabel('日期')
                                        plt.ylabel('投資組合價值 ($)')
                                        plt.grid(True, linestyle='--', alpha=0.6)
                                        plt.legend()
                                        plt.tight_layout()
                                        st.pyplot(fig_bt)
                                        plt.close(fig_bt)
                                else:
                                    st.warning(f"無法為 {ticker} 生成回測結果。")
                        
                        st.markdown("---")
                        st.subheader("📰 相關新聞與情緒分析 (參考)")
                        st.write(f"以下為 {ticker} 在 Google News 上的相關新聞連結，請自行查閱以獲取最新資訊。情緒分析需更複雜的 NLP 技術和數據源。")
                        st.markdown(f"[點擊查看 {ticker} Google 新聞](https://news.google.com/search?q={ticker} stock&hl=zh-TW&gl=TW&ceid=TW%3Azh-Hant)")
                        st.markdown("""
                        **關於情緒分析：**
                        情緒分析 (Sentiment Analysis) 旨在判斷文本（如新聞標題、文章內容、社群媒體貼文）所表達的情緒是正向、負向還是中性。
                        在股票市場中，情緒分析可以幫助投資者評估市場對特定股票或行業的整體看法。
                        * **正向情緒：** 可能預示利好消息，潛在買入機會。
                        * **負向情緒：** 可能預示利空消息，潛在賣出風險。
                        * **中性情緒：** 市場反應不明顯，可能需要更多資訊。
                        """)

                    else:
                        st.error("數據不足以進行完整的分析和報告。")
                else:
                    st.error(f"未能獲取 {ticker} 的數據。")
