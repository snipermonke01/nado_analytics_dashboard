#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import pandas as pd


from datetime import datetime, time, timezone


def _get_eod_time():
    today = datetime.now(timezone.utc).date()

    end_of_day = datetime.combine(today, time(23, 59, 59), tzinfo=timezone.utc)

    return int(end_of_day.timestamp())


def fetch_raw_volume_data(url="https://archive.prod.nado.xyz/v2/tickers"):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def calculate_total_quote_volume(raw_data):
    return sum(item["quote_volume"] for item in raw_data.values())


def ts_floor(ts):
    return pd.to_datetime(ts, unit="s").floor("D")


def fetch_nlp_snapshots(
    url="https://archive.prod.nado.xyz/v1",
    count=10000,
    granularity=86400
):
    payload = {
        "nlp_snapshots": {
            "interval": {
                "count": count,
                "max_time": _get_eod_time(),
                "granularity": granularity
            }
        }
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()

    return data


def fetch_general_stats_snapshots(
    url="https://archive.prod.nado.xyz/v1",
    count=10000,
    granularity=86400
):
    payload = {
        "edge_market_snapshots": {
            "interval": {
                "count": count,
                "max_time": _get_eod_time(),
                "granularity": granularity
            }
        }
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()["snapshots"]["57073"]
    return data


def parse_general_stats(data):

    base_df = pd.DataFrame([{
        "timestamp": row["timestamp"],
        "cumulative_users": row["cumulative_users"],
        "daily_active_users": row["daily_active_users"],
        "tvl": row["tvl"]
    } for row in data])

    base_df["date"] = base_df["timestamp"].apply(ts_floor)
    base_df = base_df.set_index("date")

    def sum_dict(d):
        return sum(int(v) for v in d.values())

    base_df["sum_cumulative_trades"] = [sum_dict(row["cumulative_trades"]) for row in data]
    base_df["sum_cumulative_volumes"] = [sum_dict(row["cumulative_volumes"]) for row in data]
    base_df["sum_open_interests"] = [sum_dict(row["open_interests"]) for row in data]

    scale_cols = ["tvl", "sum_cumulative_volumes", "sum_open_interests"]
    base_df[scale_cols] = base_df[scale_cols].astype(float) / 1e18
    return base_df


def extract_tvl(raw_data):

    data = pd.DataFrame(raw_data['snapshots'])

    data["date"] = data["timestamp"].apply(ts_floor)
    data = data.set_index("date")

    data = data[["oracle_price_x18", "tvl"]].copy()
    data = data.rename(columns={"oracle_price_x18": "NLP Price",
                                "tvl": "NLP TVL"})

    data[["NLP Price", "NLP TVL"]] = data[["NLP Price", "NLP TVL"]].astype(float) / 1e18

    return data


def get_stats():
    raw_stats = fetch_general_stats_snapshots()
    general_stats = parse_general_stats(raw_stats)
    nlp_raw_stats = fetch_nlp_snapshots()
    nlp_stats = extract_tvl(nlp_raw_stats)
    merged_dataframe = general_stats.join(nlp_stats, how="left")
    return merged_dataframe.drop(columns=["timestamp"])


if __name__ == "__main__":

    output = get_stats()
