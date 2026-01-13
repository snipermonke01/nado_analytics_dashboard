#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


from get_metrics import get_stats

st.set_page_config(layout="wide")

st.title("Nado Analytics Dashboard")

# Load Data
df, assets = get_stats(return_assets=True)

# KPI Row
latest = df.iloc[-1]

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    with st.container(border=True):
        st.metric("Daily Active Users", f"{latest['daily_active_users']:,.0f}")
with kpi2:
    with st.container(border=True):
        st.metric("Cumulative Users", f"{latest['cumulative_users']:,.0f}")
with kpi3:
    with st.container(border=True):
        st.metric("Protocol TVL", f"${latest['tvl']:,.0f}")
with kpi4:
    with st.container(border=True):
        st.metric("NLP TVL", f"${latest['NLP TVL']:,.0f}")
with kpi5:
    with st.container(border=True):
        st.metric("Current Open Interest", f"${latest['sum_open_interests']:,.0f}")

# --- Helper: wide (date x asset) -> long (date, asset, value) for stacked bars ---


def _to_long(wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    if wide is None or wide.empty:
        return pd.DataFrame(columns=["date", "asset", value_name])
    tmp = wide.copy()
    tmp.index.name = "date"
    long = tmp.reset_index().melt(id_vars=["date"], var_name="asset", value_name=value_name)
    return long


def _asset_order(long_df: pd.DataFrame, value_col: str) -> list[str]:
    """Order assets by total value (descending) so the largest stacks sit at the bottom."""
    if long_df is None or long_df.empty:
        return []
    return (
        long_df.groupby("asset", dropna=False)[value_col]
        .sum(min_count=1)
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )


def _apply_legend_offset(fig: go.Figure, right_margin: int = 140) -> None:
    """Move the legend slightly right so it doesn't collide with the secondary y-axis labels."""
    fig.update_layout(
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        margin=dict(r=right_margin),
    )


def _nov20_start(index: pd.Index) -> pd.Timestamp | None:
    """Find the first Nov-20 date in the index; fall back to Nov-20 of the min year (or next)"""
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return None
    # try exact month/day match in the data
    mask = (index.month == 11) & (index.day == 20)
    if mask.any():
        return pd.Timestamp(index[mask][0]).normalize()
    # fallback: Nov 20 of the first year, otherwise the next year
    y0 = index.min().year
    candidate = pd.Timestamp(year=y0, month=11, day=20)
    if candidate < index.min():
        candidate = pd.Timestamp(year=y0 + 1, month=11, day=20)
    return candidate


# 1. Volume: stacked daily bars by asset + cumulative total line
daily_vol_long = _to_long(assets.get("daily_volume"), "daily_volume")
vol_order = _asset_order(daily_vol_long, "daily_volume")

fig_vol = px.bar(
    daily_vol_long,
    x="date",
    y="daily_volume",
    color="asset",
    category_orders={"asset": vol_order},
    barmode="stack",
    title="Daily Volume by Asset",
)

# Ensure trace order follows our desired stacking order (bottom = largest)
if vol_order:
    fig_vol.data = tuple(sorted(fig_vol.data, key=lambda t: vol_order.index(t.name)
                         if t.name in vol_order else 10**9))

fig_vol.add_trace(
    go.Scatter(
        x=df.index,
        y=df["sum_cumulative_volumes"],
        name="Cumulative Volume",
        mode="lines",
        yaxis="y2",
    )
)

fig_vol.update_layout(
    yaxis=dict(title="Daily Volume"),
    yaxis2=dict(title=dict(text="Cumulative Volume", standoff=10), overlaying="y", side="right"),
    legend_title_text="Asset",
)
_apply_legend_offset(fig_vol)
# 2. Trades: stacked daily bars by asset + cumulative total line
daily_trades_long = _to_long(assets.get("daily_trades"), "daily_trades")
trades_order = _asset_order(daily_trades_long, "daily_trades")

fig_trades = px.bar(
    daily_trades_long,
    x="date",
    y="daily_trades",
    color="asset",
    category_orders={"asset": trades_order},
    barmode="stack",
    title="Daily Trades by Asset",
)

if trades_order:
    fig_trades.data = tuple(sorted(fig_trades.data, key=lambda t: trades_order.index(
        t.name) if t.name in trades_order else 10**9))

fig_trades.add_trace(
    go.Scatter(
        x=df.index,
        y=df["sum_cumulative_trades"],
        name="Total Cumulative Trades",
        mode="lines",
        yaxis="y2",
    )
)

fig_trades.update_layout(
    yaxis=dict(title="Daily Trades"),
    yaxis2=dict(title=dict(text="Cumulative Trades", standoff=10), overlaying="y", side="right"),
    legend_title_text="Asset",
)
_apply_legend_offset(fig_trades)
# 3. Open Interest: stacked bars by asset (current OI distribution over time)
oi_long = _to_long(assets.get("open_interest"), "open_interest")
oi_order = _asset_order(oi_long, "open_interest")

fig_oi = px.bar(
    oi_long,
    x="date",
    y="open_interest",
    color="asset",
    category_orders={"asset": oi_order},
    barmode="stack",
    title="Open Interest by Asset",
)

if oi_order:
    fig_oi.data = tuple(sorted(fig_oi.data, key=lambda t: oi_order.index(t.name)
                        if t.name in oi_order else 10**9))

fig_oi.update_layout(yaxis=dict(title="Open Interest"), legend_title_text="Asset")
_apply_legend_offset(fig_oi)
# 4. NLP TVL (bar) + NLP Price (line; starts Nov 20)
price_start = _nov20_start(df.index)
price_mask = (df.index >= price_start) if price_start is not None else pd.Series(
    [True] * len(df), index=df.index)

fig_nlp = go.Figure()
fig_nlp.add_trace(go.Bar(x=df.index, y=df["NLP TVL"], name="NLP TVL", opacity=0.7))
fig_nlp.add_trace(
    go.Scatter(
        x=df.index[price_mask],
        y=df.loc[price_mask, "NLP Price"],
        name="NLP Price",
        mode="lines",
        yaxis="y2",
    )
)
fig_nlp.update_layout(
    title="NLP TVL (bar) + NLP Price (line)",
    yaxis=dict(title="NLP TVL"),
    yaxis2=dict(title=dict(text="NLP Price", standoff=10), overlaying="y", side="right"),
)
_apply_legend_offset(fig_nlp)

# 5. Users (bar) + Cumulative Users (line)
fig_users = go.Figure()
fig_users.add_trace(go.Bar(x=df.index, y=df["daily_active_users"], name="Daily Active Users", opacity=0.7))
fig_users.add_trace(go.Scatter(x=df.index, y=df["cumulative_users"],
                    name="Cumulative Users", mode="lines", yaxis="y2"))
fig_users.update_layout(
    title="Daily Active Users",
    yaxis=dict(title="Daily Active Users"),
    yaxis2=dict(title=dict(text="Cumulative Users", standoff=10), overlaying="y", side="right"),
)
_apply_legend_offset(fig_users)

# 6. TVL (line)
fig_tvl = px.line(df, x=df.index, y="tvl", title="Protocol TVL")
_apply_legend_offset(fig_tvl, right_margin=120)

# Build rows for plots
c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.plotly_chart(fig_vol, use_container_width=True)

with c2:
    with st.container(border=True):
        st.plotly_chart(fig_trades, use_container_width=True)

with c3:
    with st.container(border=True):
        st.plotly_chart(fig_oi, use_container_width=True)

c4, c5, c6 = st.columns(3)

with c4:
    with st.container(border=True):
        st.plotly_chart(fig_nlp, use_container_width=True)

with c5:
    with st.container(border=True):
        st.plotly_chart(fig_users, use_container_width=True)

with c6:
    with st.container(border=True):
        st.plotly_chart(fig_tvl, use_container_width=True)
