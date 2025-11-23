#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from get_metrics import get_stats

st.set_page_config(layout="wide")

# Page Title
st.title("Nado Analytics Dashboard")

# Load Data
df = get_stats()

# Ensure sorted by date index
df = df.sort_index()

# KPI Summary Box
with st.container():
    st.subheader("📊 Protocol Overview")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    latest = df.iloc[-1]

    # Latest values
    latest_cum_volume = latest["sum_cumulative_volumes"]
    latest_cum_users = latest["cumulative_users"]
    protocol_tvl = latest["tvl"]
    latest_nlp_tvl = latest["NLP TVL"]
    latest_open_interest = latest["sum_open_interests"]

    with kpi1:
        with st.container(border=True):
            st.metric("Cumulative Volume", f"${latest['sum_cumulative_volumes']:,.0f}")

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

df["daily_volume"] = df["sum_cumulative_volumes"].diff().fillna(df["sum_cumulative_volumes"])
df["daily_trades"] = df["sum_cumulative_trades"].diff().fillna(df["sum_cumulative_trades"])
df["daily_new_users"] = df["cumulative_users"].diff().fillna(df["cumulative_users"])

# 1. Daily Volume + Cumulative Volume
fig_vol = go.Figure()
fig_vol.add_trace(go.Bar(x=df.index, y=df["daily_volume"], name="Daily Volume", opacity=0.7))
fig_vol.add_trace(go.Scatter(x=df.index, y=df["sum_cumulative_volumes"], name="Cumulative Volume", mode="lines", yaxis="y2"))
fig_vol.update_layout(
    title="Daily Volume (bar) + Cumulative Volume (line)",
    yaxis=dict(title="Daily Volume"),
    yaxis2=dict(title="Cumulative Volume", overlaying="y", side="right"),
)

# 2. Daily Trades + Cumulative Trades
fig_trades = go.Figure()
fig_trades.add_trace(go.Bar(x=df.index, y=df["daily_trades"], name="Daily Trades", opacity=0.7))
fig_trades.add_trace(go.Scatter(x=df.index, y=df["sum_cumulative_trades"], name="Cumulative Trades", mode="lines", yaxis="y2"))
fig_trades.update_layout(
    title="Daily Trades (bar) + Cumulative Trades (line)",
    yaxis=dict(title="Daily Trades"),
    yaxis2=dict(title="Cumulative Trades", overlaying="y", side="right"),
)

# 3. Open Interest (bar)
fig_oi = px.bar(df, x=df.index, y="sum_open_interests", title="Open Interest")

# 4. NLP TVL (bar) + NLP Price (line)
fig_nlp = go.Figure()
fig_nlp.add_trace(go.Bar(x=df.index, y=df["NLP TVL"], name="NLP TVL", opacity=0.7))
fig_nlp.add_trace(go.Scatter(x=df.index, y=df["NLP Price"], name="NLP Price", mode="lines", yaxis="y2"))
fig_nlp.update_layout(
    title="NLP TVL (bar) + NLP Price (line)",
    yaxis=dict(title="NLP TVL"),
    yaxis2=dict(title="NLP Price", overlaying="y", side="right"),
)

# 5. Daily Active Users + Cumulative Users
fig_users = go.Figure()
fig_users.add_trace(go.Bar(x=df.index, y=df["daily_active_users"], name="Daily Active Users", opacity=0.7))
fig_users.add_trace(go.Scatter(x=df.index, y=df["cumulative_users"], name="Cumulative Users", mode="lines", yaxis="y2"))
fig_users.update_layout(
    title="Daily Active Users (bar) + Cumulative Users (line)",
    yaxis=dict(title="Daily Active Users"),
    yaxis2=dict(title="Cumulative Users", overlaying="y", side="right"),
)

# 6. TVL (line)
fig_tvl = px.line(df, x=df.index, y="tvl", title="Protocol TVL")


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
