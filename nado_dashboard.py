#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from get_metrics import get_stats

st.set_page_config(layout="wide")
st.title("Nado Analytics Dashboard")

# Load Data
df, assets = get_stats(return_assets=True)
latest = df.iloc[-1]

# -----------------------------
# KPI Row
# -----------------------------
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

# -----------------------------
# Helpers
# -----------------------------
MONEY_HOVER = "%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra>%{fullData.name}</extra>"
PLAIN_HOVER = "%{x|%Y-%m-%d}<br>%{y:,.2f}<extra>%{fullData.name}</extra>"

def _axis_scale_from_max(max_abs: float) -> tuple[float, str]:
    """Return (divisor, suffix) based on the axis max magnitude."""
    if max_abs >= 1_000_000_000:
        return 1_000_000_000.0, "b"
    if max_abs >= 1_000_000:
        return 1_000_000.0, "m"
    if max_abs >= 1_000:
        return 1_000.0, "k"
    return 1.0, ""

def _fmt_scaled(v: float, divisor: float, suffix: str) -> str:
    """Format value using a fixed divisor/suffix for the whole axis."""
    if v is None:
        return ""
    try:
        if isinstance(v, float) and np.isnan(v):
            return ""
    except Exception:
        pass

    # Special-case 0 so it stays "0" (not "0k/0m/0b")
    if v == 0:
        return "0"

    if divisor == 1.0:
        return f"{v:,.0f}"

    scaled = v / divisor
    av = abs(scaled)

    if av >= 100:
        s = f"{scaled:,.0f}"
    elif av >= 10:
        s = f"{scaled:,.0f}"
    else:
        s = f"{scaled:,.1f}".rstrip("0").rstrip(".")

    return f"{s}{suffix}"

def _nice_step(span: float, nticks: int) -> float:
    """Nice step size (1/2/5 * 10^n)."""
    if span <= 0 or nticks < 2:
        return 1.0
    raw = span / (nticks - 1)
    mag = 10 ** np.floor(np.log10(raw))
    norm = raw / mag
    if norm <= 1:
        nice = 1
    elif norm <= 2:
        nice = 2
    elif norm <= 5:
        nice = 5
    else:
        nice = 10
    return nice * mag

def _compute_nice_ticks(vmin: float, vmax: float, nticks: int = 6) -> list[float]:
    """Generate nice tick values that cover [vmin, vmax]."""
    if vmin == vmax:
        return [vmin]
    span = vmax - vmin
    step = _nice_step(span, nticks)
    start = np.floor(vmin / step) * step
    end = np.ceil(vmax / step) * step
    ticks = np.arange(start, end + step * 0.5, step)
    return ticks.tolist()

def _stack_totals_for_axis(fig: go.Figure, axis_name: str) -> tuple[list[float], list[float]]:
    """
    For stacked bars, Plotly’s axis range depends on the SUM of bar traces at each x.
    Return (stack_sums, all_bar_values) for traces on the given axis.
    """
    bar_traces = [
        tr for tr in fig.data
        if getattr(tr, "type", None) == "bar"
        and getattr(tr, "yaxis", "y") == axis_name
        and getattr(tr, "y", None) is not None
    ]
    if not bar_traces:
        return [], []

    # Use the first bar trace's x as the index; px.bar aligns traces by x position.
    x0 = list(bar_traces[0].x) if getattr(bar_traces[0], "x", None) is not None else list(range(len(bar_traces[0].y)))
    sums = {x: 0.0 for x in x0}

    all_vals = []
    for tr in bar_traces:
        xs = list(tr.x) if getattr(tr, "x", None) is not None else x0
        ys = list(pd.Series(tr.y).fillna(0).astype(float).values)
        for x, y in zip(xs, ys):
            sums[x] = sums.get(x, 0.0) + y
        all_vals.extend(ys)

    return list(sums.values()), all_vals

def _apply_kmb_ticks(fig: go.Figure, which: str = "y", prefix: str = "", nticks: int = 6) -> None:
    """
    Apply k/m/b tick labels to an axis using ONE unit chosen from the AXIS MAX TICK.
    Also correctly handles STACKED bars by using per-x summed totals.
    """
    axis_name = "y" if which == "y" else "y2"

    vals = []

    # 1) If there are bars, include STACK totals (this is what drives the axis max in stacked charts)
    stack_sums, bar_vals = _stack_totals_for_axis(fig, axis_name)
    vals.extend(stack_sums)          # important for max
    vals.extend(bar_vals)            # keep for min (if negatives ever exist)

    # 2) Include non-bar traces (lines, scatter, etc.) on this axis
    for tr in fig.data:
        tr_axis = getattr(tr, "yaxis", "y")
        if tr_axis != axis_name:
            continue
        if getattr(tr, "type", None) == "bar":
            continue
        if getattr(tr, "y", None) is None:
            continue
        try:
            vals.extend(list(pd.Series(tr.y).dropna().astype(float).values))
        except Exception:
            pass

    if not vals:
        return

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    # include 0 for nicer axes
    vmin = min(vmin, 0.0)
    vmax = max(vmax, 0.0)

    # Build nice ticks FIRST, then choose scale based on the TOP tick
    tickvals = _compute_nice_ticks(vmin, vmax, nticks=nticks)
    top_tick = max(abs(tickvals[0]), abs(tickvals[-1]))
    divisor, suffix = _axis_scale_from_max(top_tick)

    ticktext = [f"{prefix}{_fmt_scaled(v, divisor, suffix)}" for v in tickvals]

    if which == "y":
        fig.update_yaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)
    else:
        fig.update_layout(
            yaxis2=dict(
                **(fig.layout.yaxis2.to_plotly_json() if fig.layout.yaxis2 else {}),
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
            )
        )


def _to_long(wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """date-indexed wide (date x asset) -> long (date, asset, value)"""
    if wide is None or wide.empty:
        return pd.DataFrame(columns=["date", "asset", value_name])
    tmp = wide.copy()
    tmp.index.name = "date"
    return tmp.reset_index().melt(id_vars=["date"], var_name="asset", value_name=value_name)

def _asset_order(long_df: pd.DataFrame, value_col: str) -> list[str]:
    """Largest total at bottom of stack."""
    if long_df is None or long_df.empty:
        return []
    return (
        long_df.groupby("asset", dropna=False)[value_col]
        .sum(min_count=1)
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )

def _sort_stack_traces(fig: go.Figure, order: list[str]) -> None:
    """Make stack order match asset order (largest first = bottom)."""
    if not order:
        return
    fig.data = tuple(sorted(fig.data, key=lambda t: order.index(t.name) if t.name in order else 10**9))

def _legend_right(fig: go.Figure, right_margin: int = 140) -> None:
    fig.update_layout(
        legend=dict(x=1.02, y=1, xanchor="left", yanchor="top"),
        margin=dict(r=right_margin),
    )

def _per_point_hover(fig: go.Figure) -> None:
    """Hover only for the point/bar under cursor (not unified x)."""
    fig.update_layout(hovermode="closest")

def _money_axes(fig: go.Figure, y1_title: str, y2_title: str | None = None) -> None:
    fig.update_yaxes(title_text=y1_title, tickprefix="$")
    _apply_kmb_ticks(fig, which="y", prefix="$")

    if y2_title is not None:
        fig.update_layout(
            yaxis2=dict(
                title=dict(text=y2_title, standoff=10),
                overlaying="y",
                side="right",
            )
        )
        _apply_kmb_ticks(fig, which="y2", prefix="$")

def _plain_axes(fig: go.Figure, y1_title: str, y2_title: str | None = None) -> None:
    fig.update_yaxes(title_text=y1_title)
    _apply_kmb_ticks(fig, which="y", prefix="")

    if y2_title is not None:
        fig.update_layout(
            yaxis2=dict(
                title=dict(text=y2_title, standoff=10),
                overlaying="y",
                side="right",
            )
        )
        _apply_kmb_ticks(fig, which="y2", prefix="")

def _set_hover(fig: go.Figure, money: bool) -> None:
    fig.update_traces(hovertemplate=(MONEY_HOVER if money else PLAIN_HOVER))

def _nov20_start(index: pd.Index) -> pd.Timestamp | None:
    """Find first Nov-20 in index; else pick Nov-20 of min year (or next)."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return None
    mask = (index.month == 11) & (index.day == 20)
    if mask.any():
        return pd.Timestamp(index[mask][0]).normalize()
    y0 = index.min().year
    candidate = pd.Timestamp(year=y0, month=11, day=20)
    if candidate < index.min():
        candidate = pd.Timestamp(year=y0 + 1, month=11, day=20)
    return candidate

# -----------------------------
# Figure builders
# -----------------------------
def build_volume(df: pd.DataFrame, assets: dict) -> go.Figure:
    long = _to_long(assets.get("daily_volume"), "daily_volume")
    order = _asset_order(long, "daily_volume")

    fig = px.bar(
        long,
        x="date",
        y="daily_volume",
        color="asset",
        category_orders={"asset": order},
        barmode="stack",
        title="Daily Volume by Asset",
    )
    _sort_stack_traces(fig, order)

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["sum_cumulative_volumes"],
            name="Cumulative Volume",
            mode="lines",
            yaxis="y2",
            hovertemplate=MONEY_HOVER,
        )
    )

    _per_point_hover(fig)
    _money_axes(fig, y1_title="Daily Volume", y2_title="Cumulative Volume")
    _set_hover(fig, money=True)
    fig.update_layout(legend_title_text="Asset")
    _legend_right(fig)
    return fig

def build_trades(df: pd.DataFrame, assets: dict) -> go.Figure:
    long = _to_long(assets.get("daily_trades"), "daily_trades")
    order = _asset_order(long, "daily_trades")

    fig = px.bar(
        long,
        x="date",
        y="daily_trades",
        color="asset",
        category_orders={"asset": order},
        barmode="stack",
        title="Daily Trades by Asset",
    )
    _sort_stack_traces(fig, order)

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["sum_cumulative_trades"],
            name="Total Cumulative Trades",
            mode="lines",
            yaxis="y2",
            hovertemplate=PLAIN_HOVER,
        )
    )

    _per_point_hover(fig)
    _plain_axes(fig, y1_title="Daily Trades", y2_title="Cumulative Trades")
    _set_hover(fig, money=False)
    fig.update_layout(legend_title_text="Asset")
    _legend_right(fig)
    return fig

def build_open_interest(df: pd.DataFrame, assets: dict) -> go.Figure:
    long = _to_long(assets.get("open_interest"), "open_interest")
    order = _asset_order(long, "open_interest")

    fig = px.bar(
        long,
        x="date",
        y="open_interest",
        color="asset",
        category_orders={"asset": order},
        barmode="stack",
        title="Open Interest by Asset",
    )
    _sort_stack_traces(fig, order)

    _per_point_hover(fig)
    _money_axes(fig, y1_title="Open Interest")
    _set_hover(fig, money=True)
    fig.update_layout(legend_title_text="Asset")
    _legend_right(fig)
    return fig

def build_nlp(df: pd.DataFrame) -> go.Figure:
    price_start = _nov20_start(df.index)
    price_mask = (df.index >= price_start) if price_start is not None else pd.Series([True] * len(df), index=df.index)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["NLP TVL"],
            name="NLP TVL",
            opacity=0.7,
            hovertemplate=MONEY_HOVER,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index[price_mask],
            y=df.loc[price_mask, "NLP Price"],
            name="NLP Price",
            mode="lines",
            yaxis="y2",
            hovertemplate=MONEY_HOVER,
        )
    )

    fig.update_layout(title="NLP TVL (bar) + NLP Price (line)")
    _per_point_hover(fig)
    _money_axes(fig, y1_title="NLP TVL", y2_title="NLP Price")
    _legend_right(fig)
    return fig

def build_users(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["daily_active_users"],
            name="Daily Active Users",
            opacity=0.7,
            hovertemplate=PLAIN_HOVER,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["cumulative_users"],
            name="Cumulative Users",
            mode="lines",
            yaxis="y2",
            hovertemplate=PLAIN_HOVER,
        )
    )

    fig.update_layout(title="Daily Active Users")
    _per_point_hover(fig)
    _plain_axes(fig, y1_title="Daily Active Users", y2_title="Cumulative Users")
    _legend_right(fig)
    return fig

def build_tvl(df: pd.DataFrame) -> go.Figure:
    fig = px.line(df, x=df.index, y="tvl", title="Protocol TVL")
    _per_point_hover(fig)
    _money_axes(fig, y1_title="TVL")
    _set_hover(fig, money=True)
    _legend_right(fig, right_margin=120)
    return fig

# -----------------------------
# Build figures
# -----------------------------
fig_vol = build_volume(df, assets)
fig_trades = build_trades(df, assets)
fig_oi = build_open_interest(df, assets)
fig_nlp = build_nlp(df)
fig_users = build_users(df)
fig_tvl = build_tvl(df)

# -----------------------------
# Layout: 2 columns of plots
# -----------------------------
c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.plotly_chart(fig_vol, use_container_width=True)
with c2:
    with st.container(border=True):
        st.plotly_chart(fig_oi, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    with st.container(border=True):
        st.plotly_chart(fig_nlp, use_container_width=True)
with c4:
    with st.container(border=True):
        st.plotly_chart(fig_tvl, use_container_width=True)

c5, c6 = st.columns(2)
with c5:
    with st.container(border=True):
        st.plotly_chart(fig_trades, use_container_width=True)
with c6:
    with st.container(border=True):
        st.plotly_chart(fig_users, use_container_width=True)
