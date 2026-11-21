"""Interactive Australian Powerball analysis — run: python -m streamlit run app.py"""
import io
from pathlib import Path
from math import comb
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import binom
from engine import (load_draws, frequency_table, pair_table, draw_features, since_last,
                    rolling_counts, stats_summary, backtest, generate_tickets, probability_summary)

st.set_page_config(page_title="Powerball Analytics Lab", page_icon="🎱", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container {padding-top:1.8rem; max-width:1550px}
h1,h2,h3 {letter-spacing:-.025em}
[data-testid="stMetric"] {background:#111d31; padding:14px 18px; border:1px solid #263a57; border-radius:14px}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color:#f1f5ff !important}
.hero {padding:20px 25px; border:1px solid #34496a; background:linear-gradient(120deg,#132440,#102b38);border-radius:18px;margin-bottom:20px}
.hero h1 {color:white;margin:0}
.hero p {color:#b9d3ef;margin:6px 0 0}
.small {font-size:0.87rem;color:#8799b0}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🎱 Australian Powerball Analytics Lab</h1><p>Explore historical draws • Compare probability models • Build 10 diversified seven-number games. Developed by Dhiraj Sharma</p></div>', unsafe_allow_html=True)
st.sidebar.header("Control room")
file = st.sidebar.file_uploader("Replace dataset (CSV)", type="csv")
source = file if file is not None else Path(__file__).with_name("5132-results.csv")
try:
    raw, all_draws, all_occ = load_draws(source)
except Exception as exc:
    st.error(f"Cannot load dataset: {exc}")
    st.stop()

window = st.sidebar.selectbox("Analysis lookback", ["All valid 7/35 draws", "Last 300 draws", "Last 200 draws", "Last 100 draws", "Last 52 draws"], index=0)
limit = {"All valid 7/35 draws":len(all_draws), "Last 300 draws":300, "Last 200 draws":200, "Last 100 draws":100, "Last 52 draws":52}[window]
draws = all_draws.tail(limit).reset_index(drop=True)
occ = all_occ[-len(draws):]
n = len(draws)
fr = frequency_table(occ)
features = draw_features(draws)
pairs, co = pair_table(occ)

st.sidebar.markdown("---")
st.sidebar.caption("Ticket builder")
style = st.sidebar.selectbox("Ticket selection method", ["Blended", "Uniform / unbiased", "Experimental ML"], index=0)
seed = st.sidebar.number_input("Reproducible random seed", min_value=1, max_value=2_000_000_000, value=20261120, step=1)
max_overlap = st.sidebar.slider("Maximum numbers shared by any 2 games", min_value=2, max_value=6, value=3)
avoid = st.sidebar.checkbox("Avoid exact historical main-number sets", value=True)
st.sidebar.info("Lottery draws are intended to be random. All 7-number sets have identical theoretical odds of hitting 7 main numbers.")

m1,m2,m3,m4 = st.columns(4)
m1.metric("CSV draws", f"{len(raw):,}")
m2.metric("Current-format draws", f"{len(all_draws):,}")
m3.metric("Analysis window", f"{n:,}")
m4.metric("Latest current-format draw", f"{all_draws['date'].max():%d %b %Y}")

(t_overview, t_freq, t_patterns, t_pairs, t_gaps, t_stats, t_ml, t_tickets) = st.tabs([
    "📊 Overview", "🔥 Hot / cold", "🧩 Draw patterns", "🔗 Number pairs", "⏳ Gaps & recency",
    "🎲 Probability lab", "🧠 ML regression", "🎟️ 10 games"])

with t_overview:
    st.subheader("The 3 historical game formats")
    count5 = int(raw['Winning Number 6'].isna().sum())
    count6 = int((raw['Winning Number 6'].notna() & raw['Winning Number 7'].isna()).sum())
    count7 = int(raw['Winning Number 7'].notna().sum())
    fig = px.bar(x=["5 main / up to 45", "6 main / up to 40", "7 main / up to 35"], y=[count5,count6,count7],
                 labels={"x":"Game format", "y":"Draws"}, text=[count5,count6,count7], color=["Earlier", "Intermediate", "Current"],
                 color_discrete_map={"Earlier":"#8098b8", "Intermediate":"#44b7cf", "Current":"#00bfa5"})
    fig.update_layout(showlegend=False, height=345, margin=dict(t=15,b=20))
    st.plotly_chart(fig, use_container_width=True)
    a,b = st.columns([1,1])
    with a:
        st.markdown("**Valid 7/35 records by year**")
        year_counts = all_draws.assign(Year=all_draws['date'].dt.year).groupby('Year').size().reset_index(name='Draws')
        fig = px.line(year_counts, x="Year", y="Draws", markers=True)
        fig.update_traces(line_color="#38bdf8")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.markdown("**Most recent 10 main-number results**")
        t = draws.tail(10).sort_values('date', ascending=False)[['date','Draw number',*[f'Winning Number {i}' for i in range(1,8)]]].copy()
        t['date'] = t['date'].dt.strftime('%d %b %Y')
        st.dataframe(t, use_container_width=True, hide_index=True)
    st.caption("The current-format dataset contains only valid draws with 7 unique main numbers from 1 to 35. The earlier Powerball formats are excluded from comparable-number analysis.")

with t_freq:
    st.subheader("Observed counts versus the fair-draw expectation")
    order = st.radio("Sort", ["Number", "Most frequent", "Least frequent"], horizontal=True)
    plot_df = fr.sort_values('Count', ascending=(order == "Least frequent")) if order != "Number" else fr
    fig = px.bar(plot_df, x='Number', y='Count', color='Z score', color_continuous_scale='RdBu_r',
                 hover_data={'Frequency %':':.1f','Expected':':.1f', 'Z score':':.2f'})
    fig.add_hline(y=n/5, line_dash="dash", line_color="#ffaa66", annotation_text="Expected = draws × 0.20")
    fig.update_layout(height=460, xaxis=dict(dtick=1), coloraxis_colorbar_title="z")
    st.plotly_chart(fig, use_container_width=True)
    l,r = st.columns(2)
    with l:
        st.markdown("**Highest observed frequencies**")
        st.dataframe(fr.nlargest(10,'Count').reset_index(drop=True), hide_index=True, use_container_width=True,
                     column_config={"Frequency %":st.column_config.NumberColumn(format="%.1f%%"), "Z score":st.column_config.NumberColumn(format="%.2f")})
    with r:
        st.markdown("**Lowest observed frequencies**")
        st.dataframe(fr.nsmallest(10,'Count').reset_index(drop=True), hide_index=True, use_container_width=True,
                     column_config={"Frequency %":st.column_config.NumberColumn(format="%.1f%%"), "Z score":st.column_config.NumberColumn(format="%.2f")})
    roll_window = st.slider("Rolling window of draws", 10, min(150,n), min(52,n), key="roll_window")
    selected_nums = st.multiselect("Show trends for numbers", options=list(range(1,36)), default=fr.nlargest(4,'Count')['Number'].tolist())
    if selected_nums:
        trend = pd.DataFrame({f"Number {num}":pd.Series(occ[:,num-1]).rolling(roll_window).mean()*100 for num in selected_nums})
        trend['Draw index'] = np.arange(1,n+1)
        fig = px.line(trend, x='Draw index', y=[f"Number {num}" for num in selected_nums], labels={'value':'Rolling appearance (%)','variable':'Number'})
        fig.add_hline(y=20, line_dash='dash', line_color='#c8a764')
        st.plotly_chart(fig, use_container_width=True)

with t_patterns:
    st.subheader("Shape of winning sets")
    v1,v2 = st.columns(2)
    with v1:
        fig = px.histogram(features, x='Sum', nbins=34, marginal='box', title='Distribution of seven-number sums')
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)
        fig = px.bar(features.groupby('Odd count').size().reset_index(name='Draws'), x='Odd count', y='Draws', title='Odd numbers per draw')
        fig.update_layout(height=345)
        st.plotly_chart(fig, use_container_width=True)
    with v2:
        fig = px.histogram(features,x='Consecutive pairs',title='Adjacent-number pairs per draw',color_discrete_sequence=['#14b8a6'])
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)
        fig = px.scatter(features, x='Sum', y='Range', color='Odd count', hover_data=['Draw','Date'], title='Sum vs number range')
        fig.update_layout(height=345)
        st.plotly_chart(fig, use_container_width=True)
    st.caption("These distributions describe historical draw shapes. Rejecting an unusual-looking ticket does not improve its underlying probability.")

with t_pairs:
    st.subheader("Co-occurrence heatmap")
    heat = np.where(np.eye(35, dtype=bool), np.nan, co)
    fig = px.imshow(heat, labels=dict(x='Number',y='Number', color='Draws together'),
                    x=np.arange(1,36), y=np.arange(1,36), color_continuous_scale='Tealgrn', aspect='auto')
    fig.update_layout(height=650, xaxis=dict(dtick=2), yaxis=dict(dtick=2))
    st.plotly_chart(fig, use_container_width=True)
    expected_pair = n * (7*6)/(35*34)
    st.caption(f"In a fair draw, a given pair appears together with probability 42/(35×34) ≈ 3.53% per draw. Expected count in this sample: {expected_pair:.1f}.")
    a,b = st.columns(2)
    a.markdown("**Most common pairs**")
    a.dataframe(pairs.nlargest(15,'Together'), hide_index=True, use_container_width=True)
    b.markdown("**Least common pairs**")
    b.dataframe(pairs.nsmallest(15,'Together'), hide_index=True, use_container_width=True)

with t_gaps:
    st.subheader("Gaps since last appearance")
    gaps = since_last(occ)
    temp = pd.DataFrame({'Number':np.arange(1,36),'Draws since last seen':gaps,
                         'Appearance count':occ.sum(axis=0)})
    fig = px.bar(temp.sort_values('Draws since last seen', ascending=False), x='Number', y='Draws since last seen',
                 color='Appearance count', title='Current historical gap for every number')
    fig.update_layout(height=470, xaxis=dict(type='category'))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(temp.sort_values('Draws since last seen', ascending=False), hide_index=True, use_container_width=True)
    st.warning("A long absence does not make a number 'due'. Each number still has a 20% chance of appearing in the next fair seven-number draw.")

with t_stats:
    st.subheader("Monte Carlo uniformity test")
    @st.cache_data(show_spinner=False)
    def cached_null(matrix_bytes, shape):
        x = np.frombuffer(matrix_bytes, dtype=np.uint8).reshape(shape)
        return stats_summary(x)
    # Faster vectorized simulation is inside the engine; cache by selected data contents.
    observed, pvalue, null = cached_null(occ.astype(np.uint8).tobytes(), occ.shape)
    c1,c2,c3 = st.columns(3)
    c1.metric('Frequency dispersion statistic', f'{observed:.2f}')
    c2.metric('Monte Carlo p-value', f'{pvalue:.3f}')
    c3.metric('Simulated fair-draw datasets', f'{len(null):,}')
    fig = px.histogram(x=null, nbins=55, labels={'x':'Dispersion statistic','y':'Simulations'},
                       title='Null distribution: each simulation uses 7 distinct numbers per draw')
    fig.add_vline(x=observed, line_color='#ffb86b', line_width=3, annotation_text='Your CSV')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Interpretation:** A small Monte Carlo p-value suggests that this particular frequency imbalance is unusual under an ideal fair 7-of-35 process. It does not provide a reliable prediction for the next draw. Large p-values are compatible with random variation, not proof of perfection.")
    st.subheader("Exact combinatorics")
    p=probability_summary()
    s1,s2 = st.columns(2)
    s1.metric('Possible main-number combinations', f"{p['seven_combinations']:,}")
    s2.metric('Chance for 10 distinct main-number sets', f"1 in {p['seven_combinations']/10:,.0f}")
    st.markdown(f"Each main number has a **20%** appearance probability per draw, but seven distinct numbers must appear. Each specified combination has probability **1 in {p['seven_combinations']:,}** of matching the seven main numbers. If you cover all 20 Powerball values for each of 10 games, the seven-main-number odds above also describe matching a complete 7+Powerball combination among the 200 purchased combinations.")

with t_ml:
    st.subheader("Ridge regression — time-ordered benchmark")
    st.write("Features use the previous 8, 26 and 52 draws only; the final 30% of eligible observations are held out chronologically. Each number is modelled as a 0/1 occurrence target. Predictions are clipped to [0.005, 0.995].")
    alpha = st.slider("Ridge regularisation strength", min_value=1.0, max_value=300.0, value=100.0, step=1.0)
    model_run = backtest(occ, alpha=alpha)
    if model_run is None:
        st.info('Choose a larger history window; model validation requires at least 122 draws.')
    else:
        c1,c2,c3 = st.columns(3)
        c1.metric('Model holdout Brier loss ↓', f"{model_run['model_brier']:.5f}")
        c2.metric('Uniform holdout Brier loss ↓', f"{model_run['baseline_brier']:.5f}")
        c3.metric('Held-out draws', f"{model_run['test_draws']:,}")
        improve = model_run['baseline_brier']-model_run['model_brier']
        st.info('The historical model beat the uniform baseline on this holdout.' if improve > 0 else 'The uniform 20%-per-number baseline matched or beat the regression experiment on this holdout.')
        chart_df = pd.DataFrame({'Number': np.arange(1,36), 'Estimated next-draw appearance':model_run['model_estimate'], 'Fair benchmark':.2})
        fig = px.bar(chart_df, x='Number', y='Estimated next-draw appearance', title='Regression outputs (experimental, not calibrated forecasts)')
        fig.add_hline(y=.2, line_dash='dash', line_color='#ffb86b')
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)
        st.caption('All 35 next-draw appearance probabilities theoretically equal 20% for an independent fair draw. Regression fitting may merely capture random historical noise. Do not interpret higher estimated values as improved lottery odds.')

with t_tickets:
    st.subheader('Generate 10 distinct seven-number games')
    st.markdown('Generate a reproducible, overlap-controlled set of ten games. **No Powerball numbers are assigned**, as requested.')
    if style == 'Experimental ML':
        model_run = backtest(occ)
        estimate = model_run['model_estimate'] if model_run is not None else None
        if estimate is None:
            st.warning('Not enough draws for the regression experiment; the generator will use the blended heuristic.')
    else:
        estimate = None
    try:
        tickets = generate_tickets(occ, n_tickets=10, seed=int(seed), style=style,
                                   model=estimate, max_overlap=max_overlap, avoid_previous=avoid)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    st.dataframe(tickets.style.format({f'N{i}':'{:02d}' for i in range(1,8)}), use_container_width=True, hide_index=True)
    st.download_button('⬇ Download these 10 games (CSV)', tickets.to_csv(index=False).encode('utf-8'),
                       file_name='powerball_10_games.csv', mime='text/csv', use_container_width=True)
    chart = np.zeros((10,35), dtype=int)
    for i,row in tickets.iterrows():
        chart[i, row[[f'N{j}' for j in range(1,8)]].to_numpy(dtype=int)-1] = 1
    heatmap = go.Figure(go.Heatmap(z=chart, x=np.arange(1,36), y=[f'Game {i}' for i in range(1,11)],
                                   colorscale=[[0,'#172943'],[.001,'#172943'],[1,'#1de5ac']], showscale=False,
                                   hovertemplate='Number %{x}<br>%{y}<br>Selected: %{z}<extra></extra>'))
    heatmap.update_layout(title='Ticket coverage across numbers 1–35', height=370, xaxis=dict(dtick=1), yaxis=dict(autorange='reversed'))
    st.plotly_chart(heatmap, use_container_width=True)
    coverage = chart.sum(axis=0)
    c1,c2,c3 = st.columns(3)
    c1.metric('Unique main numbers covered', f'{np.sum(coverage > 0)} / 35')
    c2.metric('Max pairwise game overlap', max(len(set(tickets.loc[i, [f'N{j}' for j in range(1,8)]]) & set(tickets.loc[k, [f'N{j}' for j in range(1,8)]])) for i in range(10) for k in range(i+1,10)))
    c3.metric('7-main-number probability', f'1 in {comb(35,7)/10:,.0f}')
    st.caption('Uniform / unbiased is the principled default for a fair lottery. Blended and ML modes are exploratory ways to sample differing ticket shapes; they cannot outperform uniformly random combinations in theoretical win probability. Avoiding historical combinations or limiting overlap also cannot increase the chance that any one line wins.')

st.markdown('---')
st.caption('For education and entertainment only • Australian Powerball 7/35 • Source: uploaded historical CSV • No guaranteed predictions or increased jackpot odds')
