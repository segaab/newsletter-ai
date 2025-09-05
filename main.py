# ===== Chunk 1/3 =====

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import os
import logging
import threading

from sodapy import Socrata
from yahooquery import Ticker
from huggingface_hub import InferenceClient

# Forecasting
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.frequencies import to_offset

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants and Setup ---
SODAPY_APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# Initialize HF client once, expects HF_TOKEN in environment
hf_client = InferenceClient(provider="cerebras", api_key=os.getenv("HF_TOKEN"))

# --- Asset mapping (COT market names -> Yahoo futures tickers) ---
assets = {
    "GOLD - COMMODITY EXCHANGE INC.": "GC=F",
    "SILVER - COMMODITY EXCHANGE INC.": "SI=F",
    "PLATINUM - NEW YORK MERCANTILE EXCHANGE": "PL=F",
    "PALLADIUM - NEW YORK MERCANTILE EXCHANGE": "PA=F",
    "WTI CRUDE OIL FINANCIAL - NEW YORK MERCANTILE EXCHANGE": "CL=F",
    "COPPER - COMMODITY EXCHANGE INC.": "HG=F",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6A=F",
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": "6B=F",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6C=F",
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "6E=F",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "6J=F",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": "6S=F",
    "BITCOIN - CHICAGO MERCANTILE EXCHANGE": "BTC=F",
    "MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE": "MBT=F",
    "MICRO ETHER - CHICAGO MERCANTILE EXCHANGE": "ETH=F",
    "E-MINI S&P FINANCIAL INDEX - CHICAGO MERCANTILE EXCHANGE": "ES=F",
    "DOW JONES U.S. REAL ESTATE IDX - CHICAGO BOARD OF TRADE": "DJR",
}

# --- Fetch COT Data ---
def fetch_cot_data(market_name: str, max_attempts: int = 3) -> pd.DataFrame:
    logger.info(f"Fetching COT data for {market_name}")
    where_clause = f'market_and_exchange_names="{market_name}"'
    attempt = 0
    while attempt < max_attempts:
        try:
            results = client.get(
                "6dca-aqww",
                where=where_clause,
                order="report_date_as_yyyy_mm_dd DESC",
                limit=1500,
            )
            if results:
                df = pd.DataFrame.from_records(results)
                df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
                df["open_interest_all"] = pd.to_numeric(df["open_interest_all"], errors="coerce")
                # Derive nets (safe)
                try:
                    df["commercial_net"] = df["commercial_long_all"].astype(float) - df["commercial_short_all"].astype(float)
                    df["non_commercial_net"] = df["non_commercial_long_all"].astype(float) - df["non_commercial_short_all"].astype(float)
                except KeyError:
                    df["commercial_net"] = 0.0
                    df["non_commercial_net"] = 0.0
                return df.sort_values("report_date")
            else:
                logger.warning(f"No COT data for {market_name}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching COT data for {market_name}: {e}")
        attempt += 1
        time.sleep(2)
    logger.error(f"Failed fetching COT data for {market_name} after {max_attempts} attempts.")
    return pd.DataFrame()

# --- Fetch Price Data (yahooquery) ---
def fetch_yahooquery_data(ticker: str, start_date: str, end_date: str, max_attempts: int = 3) -> pd.DataFrame:
    logger.info(f"Fetching Yahoo data for {ticker} from {start_date} to {end_date}")
    attempt = 0
    while attempt < max_attempts:
        try:
            t = Ticker(ticker)
            hist = t.history(start=start_date, end=end_date, interval="1d")
            if hist.empty:
                logger.warning(f"No price data for {ticker}")
                return pd.DataFrame()
            if isinstance(hist.index, pd.MultiIndex):
                hist = hist.loc[ticker]
            hist = hist.reset_index()
            # ensure datetime
            hist["date"] = pd.to_datetime(hist["date"])
            return hist.sort_values("date")
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {ticker}: {e}")
        attempt += 1
        time.sleep(2)
    logger.error(f"Failed fetching Yahoo data for {ticker} after {max_attempts} attempts.")
    return pd.DataFrame()

# --- Calculate Relative Volume (RVOL) ---
def calculate_rvol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    vol_col = "volume" if "volume" in df.columns else ("Volume" if "Volume" in df.columns else None)
    if vol_col is None:
        df["rvol"] = np.nan
        return df
    rolling_avg = df[vol_col].rolling(window).mean()
    df["rvol"] = df[vol_col] / (rolling_avg.replace(0, np.nan))
    return df

# --- Merge COT and Price Data (asof fill for weekly COT onto daily prices) ---
def merge_cot_price(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if cot_df.empty or price_df.empty:
        return pd.DataFrame()

    cot_df_small = cot_df[["report_date", "open_interest_all", "commercial_net", "non_commercial_net"]].copy()
    cot_df_small.rename(columns={"report_date": "date"}, inplace=True)
    cot_df_small["date"] = pd.to_datetime(cot_df_small["date"])

    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.sort_values("date")
    cot_df_small = cot_df_small.sort_values("date")

    full_dates = pd.DataFrame({"date": pd.date_range(price_df["date"].min(), price_df["date"].max())})
    cot_df_filled = pd.merge_asof(full_dates, cot_df_small, on="date", direction="backward")

    merged = pd.merge(
        price_df,
        cot_df_filled[["date", "open_interest_all", "commercial_net", "non_commercial_net"]],
        on="date",
        how="left",
    )
    merged["open_interest_all"] = merged["open_interest_all"].ffill()
    merged["commercial_net"] = merged["commercial_net"].ffill()
    merged["non_commercial_net"] = merged["non_commercial_net"].ffill()
    return merged

# ===== Chunk 2/3 =====

# --- Health Gauge (0-10) ---
# 25% Open Interest
# 35% COT Analytics = 40% short-term (3m) commercial trend + 60% long-term (1y) non-commercial trend
# 40% Price Return w/ RVOL >= 75th percentile and volume spike
def calculate_health_gauge(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> float:
    if cot_df.empty or price_df.empty:
        return np.nan

    last_date = pd.to_datetime(price_df["date"]).max()
    one_year_ago = last_date - pd.Timedelta(days=365)
    three_months_ago = last_date - pd.Timedelta(days=90)

    # --- Open Interest (25%) ---
    oi = price_df["open_interest_all"].dropna()
    if oi.empty:
        oi_score = 0.0
    else:
        rng = (oi.max() - oi.min()) + 1e-9
        oi_norm = (oi - oi.min()) / rng
        oi_score = float(oi_norm.iloc[-1])  # 0..1

    # --- COT Analytics (35%) ---
    commercial = cot_df[["report_date", "commercial_net"]].dropna().copy()
    commercial["report_date"] = pd.to_datetime(commercial["report_date"])
    short_term = commercial[commercial["report_date"] >= three_months_ago]

    noncomm = cot_df[["report_date", "non_commercial_net"]].dropna().copy()
    noncomm["report_date"] = pd.to_datetime(noncomm["report_date"])
    long_term = noncomm[noncomm["report_date"] >= one_year_ago]

    if short_term.empty:
        st_score = 0.0
    else:
        cmin, cmax = short_term["commercial_net"].min(), short_term["commercial_net"].max()
        st_score = float((short_term["commercial_net"].iloc[-1] - cmin) / (cmax - cmin + 1e-9))  # 0..1

    if long_term.empty:
        lt_score = 0.0
    else:
        nmin, nmax = long_term["non_commercial_net"].min(), long_term["non_commercial_net"].max()
        lt_score = float((long_term["non_commercial_net"].iloc[-1] - nmin) / (nmax - nmin + 1e-9))  # 0..1

    cot_analytics = 0.4 * st_score + 0.6 * lt_score  # 0..1

    # --- Price Return + RVOL + Volume spike (40%) ---
    recent = price_df[pd.to_datetime(price_df["date"]) >= three_months_ago].copy()
    if recent.empty:
        pv_score = 0.0
    else:
        close_col = "close" if "close" in recent.columns else ("Close" if "Close" in recent.columns else None)
        vol_col = "volume" if "volume" in recent.columns else ("Volume" if "Volume" in recent.columns else None)

        if close_col is None or vol_col is None or "rvol" not in recent.columns:
            pv_score = 0.0
        else:
            recent["return"] = recent[close_col].pct_change().fillna(0.0)
            rvol_75 = recent["rvol"].quantile(0.75)
            recent["vol_avg20"] = recent[vol_col].rolling(20).mean()
            recent["vol_spike"] = recent[vol_col] > recent["vol_avg20"]

            filt = recent[(recent["rvol"] >= rvol_75) & (recent["vol_spike"])]
            if filt.empty:
                pv_score = 0.0
            else:
                last_ret = float(filt["return"].iloc[-1])
                # Map to 1..5 buckets, then to 0..1
                if last_ret >= 0.02:
                    bucket = 5
                elif 0.01 <= last_ret < 0.02:
                    bucket = 4
                elif -0.01 <= last_ret < 0.01:
                    bucket = 3
                elif -0.02 <= last_ret < -0.01:
                    bucket = 2
                else:
                    bucket = 1
                pv_score = (bucket - 1) / 4.0  # 0..1

    # Weighted sum -> 0..10
    health_score = (0.25 * oi_score + 0.35 * cot_analytics + 0.40 * pv_score) * 10.0
    return float(health_score)


# --- ARIMA 7-day forecast (runs AFTER health gauge is computed) ---
def arima_forecast_next_7(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit ARIMA(1,1,1) on the 'close' (or 'Close') column and forecast 7 calendar days.
    Anchors from the last available date in price_df.
    Returns DataFrame with columns ['date','forecast'].
    """
    if price_df.empty:
        return pd.DataFrame(columns=["date", "forecast"])

    df = price_df.copy()
    close_col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
    if close_col is None:
        return pd.DataFrame(columns=["date", "forecast"])

    s = df.dropna(subset=[close_col]).set_index(pd.to_datetime(df["date"]))[close_col].asfreq("D")
    # Fill missing daily steps via forward-fill to keep ARIMA stable
    s = s.ffill()

    # Need enough length to fit ARIMA
    if s.dropna().shape[0] < 20:
        return pd.DataFrame(columns=["date", "forecast"])

    # Fit & forecast
    try:
        model = ARIMA(s, order=(1, 1, 1))
        fitted = model.fit(method_kwargs={"warn_convergence": False})
        fc = fitted.forecast(steps=7)
    except Exception as e:
        logger.warning(f"ARIMA failed, fallback to naive last value. Error: {e}")
        fc = pd.Series([s.iloc[-1]] * 7, index=pd.date_range(s.index[-1] + to_offset("1D"), periods=7, freq="D"))

    forecast_df = pd.DataFrame({"date": fc.index.normalize(), "forecast": fc.values})
    return forecast_df.reset_index(drop=True)


# --- Thread-safe batch fetch with throttling and error handling ---
def fetch_batch(batch_assets, start_date, end_date, cot_results, price_results, forecast_results, lock):
    for cot_name, ticker in batch_assets:
        try:
            cot_df = fetch_cot_data(cot_name)
            if cot_df.empty:
                logger.warning(f"No COT data for {cot_name}")
                with lock:
                    cot_results[cot_name] = pd.DataFrame()
                    price_results[cot_name] = pd.DataFrame()
                    forecast_results[cot_name] = pd.DataFrame()
                continue

            cot_start = cot_df["report_date"].min().date()
            cot_end = cot_df["report_date"].max().date()
            adj_start = max(start_date, cot_start)
            adj_end = min(end_date, cot_end + datetime.timedelta(days=7))

            price_df = fetch_yahooquery_data(ticker, adj_start.isoformat(), adj_end.isoformat())
            if price_df.empty:
                logger.warning(f"No price data for {ticker}")
                with lock:
                    cot_results[cot_name] = cot_df
                    price_results[cot_name] = pd.DataFrame()
                    forecast_results[cot_name] = pd.DataFrame()
                continue

            price_df = calculate_rvol(price_df)
            merged_df = merge_cot_price(cot_df, price_df)
            score = calculate_health_gauge(cot_df, merged_df)
            merged_df["health_score"] = score

            # --- Forecast AFTER health gauge ---
            forecast_df = arima_forecast_next_7(merged_df)

            with lock:
                cot_results[cot_name] = cot_df
                price_results[cot_name] = merged_df
                forecast_results[cot_name] = forecast_df

        except Exception as e:
            logger.error(f"Error loading data for {cot_name}: {e}")
            with lock:
                cot_results[cot_name] = pd.DataFrame()
                price_results[cot_name] = pd.DataFrame()
                forecast_results[cot_name] = pd.DataFrame()


# --- Fetch all data in batches of 5, running TWO batches concurrently at a time ---
def fetch_all_data(assets_dict, start_date, end_date, batch_size: int = 5):
    cot_results: dict[str, pd.DataFrame] = {}
    price_results: dict[str, pd.DataFrame] = {}
    forecast_results: dict[str, pd.DataFrame] = {}
    lock = threading.Lock()

    items = list(assets_dict.items())
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    # process two batches at a time
    for i in range(0, len(batches), 2):
        active_threads = []
        for j in range(i, min(i + 2, len(batches))):
            t = threading.Thread(
                target=fetch_batch,
                args=(batches[j], start_date, end_date, cot_results, price_results, forecast_results, lock),
                daemon=True,
            )
            t.start()
            active_threads.append(t)

        # wait for the two batches to finish before starting next two
        for t in active_threads:
            t.join()

        # small throttle gap between waves
        time.sleep(0.5)

    return cot_results, price_results, forecast_results

# ===== Chunk 3/3 =====

def build_llm_prompt(price_results: dict, cot_results: dict, forecast_results: dict) -> str:
    """
    Builds the analysis prompt and asks the model to return HTML that Streamlit can render with st.write(..., unsafe_allow_html=True).
    We keep your original analytical instructions and append formatting guidance PLUS the 7-day ARIMA forecasts per asset.
    """
    prompt = (
        "You are an expert market analyst.\n\n"
        "Based on the health gauge scores calculated from Open Interest, COT positional trends "
        "(commercial short term and non-commercial long term), and price & volume spikes "
        "(relative volume above the 75th percentile with a volume spike), analyze the likelihood "
        "of the gauge reaching its full effect and the market reverting to the mean or reversing.\n"
        "If the gauge is low (below 3), discuss the chances of a sell regime developing or ongoing.\n"
        "If the gauge is high (above 5), discuss the chances of a buy regime developing or ongoing.\n"
        "Include key price levels to monitor, and avoid bullet points—each asset should be a detailed paragraph.\n\n"
        "---\n"
        "FORMAT INSTRUCTIONS (very important):\n"
        "Return valid HTML only. For EACH asset, wrap output in:\n"
        "<div class='asset'>\n"
        "  <div class='asset-title'>ASSET TITLE</div>\n"
        "  <div class='asset-body'>Your paragraph analysis here.</div>\n"
        "  <div class='asset-forecast'>A compact 7-day forecast table or inline list: DATE: PRICE; ...</div>\n"
        "</div>\n"
        "Use plain <div> and <br> (no Markdown). Keep titles short and the body as one paragraph (no lists).\n"
        "Do not include any CSS or <style> tags; only plain HTML tags as specified.\n"
        "---\n\n"
    )

    for cot_name, merged_df in price_results.items():
        if merged_df.empty or "health_score" not in merged_df.columns:
            continue

        health_score = float(merged_df["health_score"].iloc[-1])
        recent = merged_df.tail(90).copy()  # ~ last 3 months

        # pick close column safely
        close_col = "close" if "close" in recent.columns else ("Close" if "Close" in recent.columns else None)
        if close_col is None:
            continue

        current_price = float(recent[close_col].iloc[-1])
        high_7d = float(recent[close_col].tail(7).max())
        low_7d = float(recent[close_col].tail(7).min())
        high_30d = float(recent[close_col].tail(30).max())
        low_30d = float(recent[close_col].tail(30).min())

        # Prepare forecast block (only 7-day forecast values)
        fdf = forecast_results.get(cot_name, pd.DataFrame())
        if fdf.empty:
            fc_text = "No forecast available."
        else:
            # produce compact "YYYY-MM-DD: value; ..." line
            pairs = [f"{pd.to_datetime(r.date).date()}: {float(r.forecast):.2f}" for r in fdf.itertuples(index=False)]
            fc_text = "; ".join(pairs)

        prompt += (
            f"<div class='asset'>"
            f"<div class='asset-title'>{cot_name}</div>"
            f"<div class='asset-body'>"
            f"{cot_name} has a Health Gauge Score of {health_score:.2f}. "
            f"The current price is {current_price:.2f}. "
            f"Over the last seven sessions, price traded between {low_7d:.2f} and {high_7d:.2f}, "
            f"and over the last thirty sessions between {low_30d:.2f} and {high_30d:.2f}. "
        )

        if health_score < 3:
            prompt += (
                "This low reading suggests a sell regime may be developing or already active. "
                "Give more weight to sessions where price declines coincide with relative volume spikes, "
                "as these confirm distribution pressure and trend continuity. "
                "Assess whether any rebounds lack volume; that would reduce the odds of a durable reversal. "
                "Identify confirmation with decisive closes through recent swing lows on elevated volume to validate the regime."
            )
        elif health_score > 5:
            prompt += (
                "This elevated reading indicates a buy regime may be developing or already in place. "
                "Prioritize sessions where price advances occur alongside relative volume spikes, "
                "since accumulation requires price and volume rising together. "
                "Pullbacks on muted volume would favor continuation, while heavy-volume failures near highs would warn of exhaustion. "
                "Confirmation would be strong closes through recent highs with persistent participation."
            )
        else:
            prompt += (
                "The reading is moderate, pointing to possible consolidation or indecision. "
                "Watch for expansion days where relative volume spikes in the direction of the break, "
                "as those moves often set the next leg. "
                "Failure swings against the prevailing drift on elevated volume would imply mean reversion risk. "
                "Treat prior 7- and 30-day bands as tactical levels for confirmation and risk control."
            )

        prompt += (
            f" Key levels to monitor are the recent seven-day high at {high_7d:.2f} and low at {low_7d:.2f}, "
            f"and the thirty-day high at {high_30d:.2f} and low at {low_30d:.2f}."
            f"</div>"
            f"<div class='asset-forecast'>{fc_text}</div>"
            f"</div>\n"
        )

    return prompt


def main():
    st.set_page_config(page_title="COT Futures Health Gauge & Newsletter", layout="wide")
    st.title("COT Futures Health Gauge Dashboard")

    # Session state
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "cot_results" not in st.session_state:
        st.session_state.cot_results = {}
    if "price_results" not in st.session_state:
        st.session_state.price_results = {}
    if "forecast_results" not in st.session_state:
        st.session_state.forecast_results = {}
    if "newsletter_text" not in st.session_state:
        st.session_state.newsletter_text = ""

    # Dates
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=365)
    start_date = st.date_input("Select Start Date", default_start)
    end_date = st.date_input("Select End Date", today)
    if start_date >= end_date:
        st.error("Start Date must be before End Date.")
        return

    # Auto-load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            cot_res, price_res, forecast_res = fetch_all_data(assets, start_date, end_date, batch_size=5)
            st.session_state.cot_results = cot_res
            st.session_state.price_results = price_res
            st.session_state.forecast_results = forecast_res
            st.session_state.data_loaded = True
        st.success("Data loaded successfully. You may now generate the newsletter.")

    # Generate newsletter on button
    if st.button("Generate Newsletter"):
        if not st.session_state.data_loaded or not st.session_state.price_results:
            st.warning("Data not ready yet.")
        else:
            with st.spinner("Generating newsletter..."):
                prompt = build_llm_prompt(
                    st.session_state.price_results,
                    st.session_state.cot_results,
                    st.session_state.forecast_results,
                )
                try:
                    completion = hf_client.chat.completions.create(
                        model="meta-llama/Llama-3.1-8B-Instruct",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    msg = completion.choices[0].message
                    content = msg.content if hasattr(msg, "content") else msg["content"]
                    st.session_state.newsletter_text = content
                except Exception as e:
                    st.error(f"Failed to generate newsletter: {e}")

    if st.session_state.newsletter_text:
        st.markdown("### Market Newsletter")
        # Render returned HTML exactly (LLM follows our format instructions)
        st.write(st.session_state.newsletter_text, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
