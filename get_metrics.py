#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fetch and shape protocol + per-asset metrics for the Nado dashboard.

This mirrors the product-id -> asset-name replacement strategy used in get_charts.py
and provides both protocol-wide aggregates and per-asset breakdowns suitable for
stacked bar charts in Streamlit/Plotly.
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any, Dict, Tuple

import pandas as pd
import requests


ARCHIVE_V1 = "https://archive.prod.nado.xyz/v1"
ARCHIVE_V2_TICKERS = "https://archive.prod.nado.xyz/v2/tickers"


def _get_eod_time() -> int:
    today = datetime.now(timezone.utc).date()
    end_of_day = datetime.combine(today, time(23, 59, 59), tzinfo=timezone.utc)
    return int(end_of_day.timestamp())


def ts_floor(ts: int) -> pd.Timestamp:
    return pd.to_datetime(ts, unit="s").floor("D")


def replace_product_ids(mapper: Dict[int, str], data: Any) -> Any:
    """Recursively replace product-id keys inside nested dicts."""
    if isinstance(data, list):
        return [replace_product_ids(mapper, item) for item in data]

    if isinstance(data, dict):
        new_dict: Dict[Any, Any] = {}
        for key, value in data.items():
            try:
                key_int = int(key)
                new_key = mapper.get(key_int, key)
            except (ValueError, TypeError):
                new_key = key
            new_dict[new_key] = replace_product_ids(mapper, value)
        return new_dict

    return data


def fetch_raw_tickers(url: str = ARCHIVE_V2_TICKERS) -> Dict[str, Any]:
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def fetch_product_id_mapper() -> Dict[int, str]:
    """Return mapping of product_id -> asset_name (e.g. {57073: 'BTC-USD'})."""
    raw = fetch_raw_tickers()
    mapper: Dict[int, str] = {}
    for asset_name, payload in raw.items():
        pid = payload.get("product_id")
        if pid is not None:
            try:
                mapper[int(pid)] = str(asset_name)
            except Exception:
                # Keep going; safest behavior is to just omit bad entries
                pass
    return mapper


def fetch_general_stats_snapshots(
    url: str = ARCHIVE_V1,
    count: int = 10000,
    granularity: int = 86400,
) -> list[dict]:
    payload = {
        "edge_market_snapshots": {
            "interval": {
                "count": count,
                "max_time": _get_eod_time(),
                "granularity": granularity,
            }
        }
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()["snapshots"]["57073"]


def fetch_nlp_snapshots(
    url: str = ARCHIVE_V1,
    count: int = 10000,
    granularity: int = 86400,
) -> dict:
    payload = {
        "nlp_snapshots": {
            "interval": {
                "count": count,
                "max_time": _get_eod_time(),
                "granularity": granularity,
            }
        }
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()


def extract_tvl(raw_data: dict) -> pd.DataFrame:
    data = pd.DataFrame(raw_data["snapshots"])
    data["date"] = data["timestamp"].apply(ts_floor)
    data = data.set_index("date")
    data = data[["oracle_price_x18", "tvl"]].copy()
    data = data.rename(columns={"oracle_price_x18": "NLP Price", "tvl": "NLP TVL"})
    data[["NLP Price", "NLP TVL"]] = data[["NLP Price", "NLP TVL"]].astype(float) / 1e18
    return data.drop(columns=["timestamp"], errors="ignore")


def _safe_numeric_df(series_of_dicts: pd.Series, scale: float | None = None) -> pd.DataFrame:
    df = series_of_dicts.apply(pd.Series)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    if scale:
        df = df.astype(float) / scale
    return df


def parse_general_stats_asset_segregated(
    snapshots: list[dict],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Build protocol-wide + per-asset breakdown frames.

    Returns:
      protocol_df: indexed by date with aggregate columns used by the dashboard
      assets: dict of wide DataFrames (index=date, columns=asset)
    """
    # Replace product IDs with asset names (same pattern as get_charts.py)
    mapper = fetch_product_id_mapper()
    snapshots = replace_product_ids(mapper, snapshots)

    base = pd.DataFrame(
        [
            {
                "timestamp": row.get("timestamp"),
                "cumulative_users": row.get("cumulative_users", 0),
                "daily_active_users": row.get("daily_active_users", 0),
                "tvl": row.get("tvl", 0),
                "cumulative_trades": row.get("cumulative_trades", {}),
                "cumulative_volumes": row.get("cumulative_volumes", {}),
                "open_interests": row.get("open_interests", {}),
            }
            for row in snapshots
        ]
    )

    base["date"] = base["timestamp"].apply(ts_floor)
    base = base.set_index("date").sort_index()

    # Wide per-asset frames
    cum_trades = _safe_numeric_df(base["cumulative_trades"])  # counts
    cum_volume = _safe_numeric_df(base["cumulative_volumes"], scale=1e18)  # USD x18
    open_interest = _safe_numeric_df(base["open_interests"], scale=1e18)  # USD x18

    # Daily per-asset (diff of cumulative)
    daily_trades = cum_trades.diff().fillna(cum_trades.iloc[0] if len(cum_trades) else 0)
    daily_volume = cum_volume.diff().fillna(cum_volume.iloc[0] if len(cum_volume) else 0)

    # Protocol-wide aggregates
    protocol = base[["timestamp", "cumulative_users", "daily_active_users", "tvl"]].copy()
    protocol[["tvl"]] = protocol[["tvl"]].astype(float) / 1e18

    protocol["sum_cumulative_trades"] = cum_trades.sum(axis=1)
    protocol["sum_cumulative_volumes"] = cum_volume.sum(axis=1)
    protocol["sum_open_interests"] = open_interest.sum(axis=1)

    protocol["daily_trades"] = daily_trades.sum(axis=1)
    protocol["daily_volume"] = daily_volume.sum(axis=1)

    assets = {
        "cumulative_trades": cum_trades,
        "cumulative_volume": cum_volume,
        "open_interest": open_interest,
        "daily_trades": daily_trades,
        "daily_volume": daily_volume,
    }
    return protocol, assets


def get_stats(return_assets: bool = True):
    """Fetch and return dashboard-ready metrics.

    - If return_assets=False: returns protocol-wide dataframe only (backwards compatible)
    - If return_assets=True: returns (protocol_df, assets_dict)
    """
    raw_stats = fetch_general_stats_snapshots()
    protocol_df, assets = parse_general_stats_asset_segregated(raw_stats)

    nlp_raw = fetch_nlp_snapshots()
    nlp_df = extract_tvl(nlp_raw)

    merged = protocol_df.join(nlp_df, how="left")
    merged = merged.drop(columns=["timestamp"], errors="ignore")

    if return_assets:
        # Align asset frames to merged index (dates)
        for k, df in list(assets.items()):
            assets[k] = df.reindex(merged.index).fillna(0)
        return merged, assets

    return merged


if __name__ == "__main__":
    df, assets = get_stats(return_assets=True)
    print(df.tail())
    print({k: v.shape for k, v in assets.items()})
