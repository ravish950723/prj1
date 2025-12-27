# eps_features.py
#
# Quarterly EPS + EPS growth flags + news sentiment
# using Alpha Vantage (no IBKR / ib_insync dependency).
#
# Alpha Vantage docs:
#   - EARNINGS:        https://www.alphavantage.co/documentation/#earnings
#   - NEWS_SENTIMENT:  https://www.alphavantage.co/documentation/#news-sentiment
#
# Requires:
#   pip install requests pandas

from typing import Optional, Dict
import statistics
import requests
import pandas as pd

from config import ALPHA_VANTAGE_API_KEY  # API key now in config.py


# -------------------------------
#   EPS via Alpha Vantage
# -------------------------------

def fetch_quarterly_eps(symbol: str) -> Optional[pd.DataFrame]:
    """
    Fetch quarterly EPS for the given symbol from Alpha Vantage "EARNINGS" endpoint.

    Returns a DataFrame with columns:
        ['reportDate', 'eps']  (latest first)
    or None on failure.
    """
    api_key = ALPHA_VANTAGE_API_KEY
    if not api_key:
        print("⚠️ ALPHA_VANTAGE_API_KEY missing; EPS disabled.")
        return None

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "EARNINGS",
        "symbol": symbol,
        "apikey": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            print(f"⚠️ EPS request failed for {symbol}: HTTP {resp.status_code}")
            return None

        data = resp.json()
        quarterly = data.get("quarterlyEarnings") or []
        if not quarterly:
            print(f"⚠️ No quarterly EPS data for {symbol}")
            return None

        # Build DataFrame of [reportDate, eps]
        rows = []
        for row in quarterly:
            date_str = row.get("fiscalDateEnding")
            eps_str = row.get("reportedEPS")
            if date_str is None or eps_str is None:
                continue
            try:
                eps_val = float(eps_str)
            except (TypeError, ValueError):
                continue
            rows.append({"reportDate": date_str, "eps": eps_val})

        if not rows:
            return None

        eps_df = pd.DataFrame(rows)
        eps_df["reportDate"] = pd.to_datetime(eps_df["reportDate"], errors="coerce")
        eps_df = eps_df.dropna(subset=["reportDate", "eps"])

        if eps_df.empty:
            return None

        # Sort latest first
        eps_df = eps_df.sort_values("reportDate", ascending=False).reset_index(drop=True)
        return eps_df

    except Exception as e:
        print(f"⚠️ EPS fetch error for {symbol}: {e}")
        return None


def eps_growth_flags(eps_df: Optional[pd.DataFrame]) -> Dict[str, Optional[bool]]:
    """
    Given an EPS DataFrame (latest first), compute growth flags.

    Returns a dict with keys:
        - 'EPS Increase 2Q'
        - 'EPS Increase 3Q'
        - 'EPS Increase 4Q'

    Each is True/False, or None if not enough data.
    """
    result: Dict[str, Optional[bool]] = {
        "EPS Increase 2Q": None,
        "EPS Increase 3Q": None,
        "EPS Increase 4Q": None,
    }

    if eps_df is None or len(eps_df) < 2:
        return result

    eps = eps_df["eps"].tolist()

    if len(eps) >= 2:
        result["EPS Increase 2Q"] = bool(eps[0] > eps[1])

    if len(eps) >= 3:
        result["EPS Increase 3Q"] = bool(eps[0] > eps[1] > eps[2])

    if len(eps) >= 4:
        result["EPS Increase 4Q"] = bool(eps[0] > eps[1] > eps[2] > eps[3])

    return result


# -------------------------------
#   News / Market Sentiment
# -------------------------------

def fetch_market_sentiment(symbol: str, max_articles: int = 50) -> Dict[str, Optional[float]]:
    """
    Fetch basic news sentiment for the symbol using Alpha Vantage NEWS_SENTIMENT.

    Returns a dict:
      {
        "news_sentiment_score": float or None,   # avg ticker_sentiment_score  (approx -1 .. +1)
        "news_positive_ratio": float or None,    # fraction of bullish vs (bullish+bearish)
        "news_article_count": int,               # number of articles considered
        "sentiment_confidence": float or None,   # composite [0..1] using score & ratio
      }

    If anything fails, fields will be None / 0.
    """
    base = {
        "news_sentiment_score": None,
        "news_positive_ratio": None,
        "news_article_count": 0,
        "sentiment_confidence": None,
    }

    api_key = ALPHA_VANTAGE_API_KEY
    if not api_key:
        print("⚠️ ALPHA_VANTAGE_API_KEY missing; sentiment disabled.")
        return base

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "sort": "LATEST",
        "apikey": api_key,
        # "limit": max_articles,  # Alpha Vantage supports limit on some plans
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            print(f"⚠️ Sentiment request failed for {symbol}: HTTP {resp.status_code}")
            return base

        data = resp.json()
        feed = data.get("feed") or []
        if not feed:
            return base

        ticker = symbol.upper()
        scores = []
        labels = []

        for article in feed[:max_articles]:
            for ts in (article.get("ticker_sentiment") or []):
                if ts.get("ticker", "").upper() == ticker:
                    score_str = ts.get("ticker_sentiment_score")
                    label = ts.get("ticker_sentiment_label", "")
                    try:
                        score_val = float(score_str)
                    except (TypeError, ValueError):
                        continue
                    scores.append(score_val)
                    labels.append(label)

        if not scores:
            return base

        avg_score = statistics.mean(scores)  # roughly -1 .. +1 by AV docs
        pos = sum(1 for l in labels if "bull" in l.lower() or "positive" in l.lower())
        neg = sum(1 for l in labels if "bear" in l.lower() or "negative" in l.lower())
        denom = pos + neg
        pos_ratio = (pos / denom) if denom > 0 else None

        # === Composite "sentiment_confidence" in [0..1]
        # Normalize avg_score (-1..+1) -> (0..1)
        score_norm = (avg_score + 1.0) / 2.0  # -1→0, 0→0.5, +1→1
        score_norm = max(0.0, min(1.0, score_norm))

        if pos_ratio is not None:
            pos_norm = max(0.0, min(1.0, pos_ratio))
            # Heavier weight on % of positive news (matches your intuition that 0.14 + 0.89 ≈ “moderately strong positive”)
            sentiment_confidence = 0.4 * score_norm + 0.6 * pos_norm
            sentiment_confidence = round(max(0.0, min(1.0, sentiment_confidence)), 4)
        else:
            sentiment_confidence = round(score_norm, 4)

        return {
            "news_sentiment_score": round(avg_score, 4),
            "news_positive_ratio": round(pos_ratio, 4) if pos_ratio is not None else None,
            "news_article_count": len(scores),
            "sentiment_confidence": sentiment_confidence,
        }

    except Exception as e:
        print(f"⚠️ Sentiment fetch error for {symbol}: {e}")
        return base
