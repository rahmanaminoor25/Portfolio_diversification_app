import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
from openai import OpenAI
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import squarify  # pip install squarify (algorithm for treemap)
from pypalettes import load_cmap
from highlight_text import fig_text
import matplotlib.patches as patches
import matplotlib.pyplot as plt
cmap = plt.cm.Set1  # or any other colormap

class PortfolioAnalyzer:
    
    def __init__(self, api_key='API key'):
        self.api_key = api_key or os.getenv('Set OPENAI_API_KEY environment variable.')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)
        self.cache = {}
        self.counter_file = Path('data/usage.json')
        self.max_analyses = 1500
        self._init_counter()
    
    def _init_counter(self):
        self.counter_file.parent.mkdir(exist_ok=True)
        if not self.counter_file.exists():
            self._save_counter(0)
    
    def _load_counter(self):
        with open(self.counter_file) as f:
            return json.load(f)['count']
    
    def _save_counter(self, count):
        with open(self.counter_file, 'w') as f:
            json.dump({'count': count, 'updated': datetime.now().isoformat()}, f)
    
    def check_budget(self):
        count = self._load_counter()
        return count < self.max_analyses, self.max_analyses - count
    
    def fetch_market_data(self, tickers):
        results = []
        errors = []
        
        for ticker in tickers:
            if ticker in self.cache:
                results.append(self.cache[ticker])
                continue
            
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                price = info.get('currentPrice') or info.get('regularMarketPrice')
                
                if not price:
                    raise ValueError(f"No price for {ticker}")
                
                data = {
                    'ticker': ticker,
                    'price': price,
                    'sector': info.get('sector', 'Unknown'),
                    'name': info.get('longName', ticker),
                    'market_cap': info.get('marketCap', 0)
                }
                
                self.cache[ticker] = data
                results.append(data)
                
            except Exception as e:
                errors.append(f"{ticker}: {str(e)}")
        
        return pd.DataFrame(results), errors
    
    def calculate_portfolio(self, holdings, market_df):
        df = pd.DataFrame(holdings).merge(market_df, on='ticker')
        
        df['value'] = df['shares'] * df['price']
        total = df['value'].sum()
        df['pct'] = (df['value'] / total * 100).round(2)
        
        if 'cost' in df.columns:
            df['cost_basis'] = df['shares'] * df['cost']
            df['gain'] = df['value'] - df['cost_basis']
            df['gain_pct'] = ((df['price'] - df['cost']) / df['cost'] * 100).round(2)
        
        return df.sort_values('value', ascending=False)
    
    def analyze_sectors(self, df):
        sector_df = df.groupby('sector')['value'].sum().reset_index()
        total = sector_df['value'].sum()
        sector_df['pct'] = (sector_df['value'] / total * 100).round(2)
        return sector_df.sort_values('value', ascending=False)
    
    def get_concentration(self, df):
        return {
            'top_1': df.iloc[0]['pct'],
            'top_3': df.head(3)['pct'].sum(),
            'top_5': df.head(5)['pct'].sum(),
            'num_holdings': len(df)
        }
    
    def compare_sp500(self, sector_df):
        sp500 = {
            'Technology': 28.5, 'Healthcare': 13.1, 'Financials': 12.9,
            'Consumer Cyclical': 10.8, 'Communication Services': 8.7,
            'Industrials': 8.3, 'Consumer Defensive': 6.8, 'Energy': 4.2,
            'Real Estate': 2.5, 'Utilities': 2.4, 'Basic Materials': 2.3
        }
        
        result = []
        for _, row in sector_df.iterrows():
            sector = row['sector']
            portfolio_pct = row['pct']
            sp500_pct = sp500.get(sector, 0)
            diff = portfolio_pct - sp500_pct
            result.append({
                'sector': sector,
                'portfolio': portfolio_pct,
                'sp500': sp500_pct,
                'diff': round(diff, 2)
            })
        
        return sorted(result, key=lambda x: abs(x['diff']), reverse=True)
    
    def generate_analysis(self, df, sector_df, concentration, comparison):
        top_5 = '\n'.join([
            f"  - {row['ticker']} ({row['name']}): {row['pct']}%"
            for _, row in df.head(5).iterrows()
        ])
        
        sectors = '\n'.join([
            f"  - {c['sector']}: Portfolio {c['portfolio']}% vs S&P {c['sp500']}% ({c['diff']:+.1f}%)"
            for c in comparison[:5]
        ])
        
        prompt = f"""You are a financial advisor analyzing a portfolio.

Portfolio:
- Total Value: ${df['value'].sum():,.2f}
- Holdings: {len(df)}

Top 5:
{top_5}

Concentration:
- Largest: {concentration['top_1']:.1f}%
- Top 3: {concentration['top_3']:.1f}%

Sectors vs S&P 500:
{sectors}

Write exactly 3 paragraphs:
1. Overall assessment (diversification, structure)
2. Key risks (2-3 specific concerns)
3. Recommendations (2-3 actionable)

Be direct and specific."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial advisor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    def analyze(self, holdings):
        allowed, remaining = self.check_budget()
        if not allowed:
            raise Exception("Budget exceeded")
        
        tickers = [h['ticker'].upper() for h in holdings]
        
        market_df, errors = self.fetch_market_data(tickers)
        if market_df.empty:
            raise Exception(f"No data fetched: {errors}")
        
        for h in holdings:
            h['ticker'] = h['ticker'].upper()
        
        portfolio_df = self.calculate_portfolio(holdings, market_df)
        sector_df = self.analyze_sectors(portfolio_df)
        concentration = self.get_concentration(portfolio_df)
        comparison = self.compare_sp500(sector_df)
        
        analysis = self.generate_analysis(portfolio_df, sector_df, concentration, comparison)
        
        count = self._load_counter()
        self._save_counter(count + 1)
        
        return {
            'portfolio': portfolio_df.to_dict('records'),
            'sectors': sector_df.to_dict('records'),
            'concentration': concentration,
            'comparison': comparison,
            'analysis': analysis,
            'errors': errors,
            'budget_remaining': remaining - 1
        }


def validate_holdings(holdings):
    if not holdings:
        return False, "Empty portfolio"
    
    if len(holdings) > 20:
        return False, "Max 20 holdings"
    
    tickers = [h.get('ticker', '').upper() for h in holdings]
    if len(tickers) != len(set(tickers)):
        return False, "Duplicate tickers"
    
    for h in holdings:
        if not h.get('ticker'):
            return False, "Missing ticker"
        if not h.get('shares') or h['shares'] <= 0:
            return False, f"Invalid shares for {h.get('ticker')}"
    
    return True, ""



def analyze_valuation(portfolio_df):
    """Calculate P/E ratios and other valuation metrics"""
    valuation_data = []
    
    for _, row in portfolio_df.iterrows():
        ticker = row['ticker']
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            valuation_data.append({
                'ticker': ticker,
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),
                'weight': row['pct']
            })
        except Exception as e:
            valuation_data.append({
                'ticker': ticker,
                'pe_ratio': None,
                'forward_pe': None,
                'peg_ratio': None,
                'price_to_book': None,
                'weight': row['pct']
            })
    
    valuation_df = pd.DataFrame(valuation_data)
    
    # Calculate weighted average P/E
    df_clean = valuation_df.dropna(subset=['pe_ratio'])
    if not df_clean.empty:
        weighted_pe = (df_clean['pe_ratio'] * df_clean['weight']).sum() / df_clean['weight'].sum()
    else:
        weighted_pe = None
    
    return valuation_df, weighted_pe



def pe_ratio_comparison(valuation_df, portfolio_df):
    """Create P/E ratio comparison chart"""
    df = valuation_df.merge(portfolio_df[['ticker', 'pct']], on='ticker')
    df = df.dropna(subset=['pe_ratio']).sort_values('pe_ratio')
    
    if df.empty:
        return None
    
    # Color coding
    colors = ['#4ECDC4' if pe < 20 else '#FFA07A' if pe < 30 else '#FF6B6B' 
              for pe in df['pe_ratio']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['ticker'],
        x=df['pe_ratio'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{pe:.1f}" for pe in df['pe_ratio']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>P/E: %{x:.1f}<br>Weight: ' + 
                      df['pct'].astype(str) + '%<extra></extra>',
        name='Your Holdings'
    ))
    
    # S&P 500 benchmark
    sp500_pe = 22.1
    fig.add_vline(x=sp500_pe, line_dash="dash", line_color="blue",
                  annotation_text=f"S&P 500 (P/E={sp500_pe})", 
                  annotation_position="top right")
    
    # Add zones
    fig.add_vrect(x0=0, x1=15, fillcolor="green", opacity=0.1, 
                  annotation_text="Value", annotation_position="top left")
    fig.add_vrect(x0=15, x1=25, fillcolor="yellow", opacity=0.1,
                  annotation_text="Fair", annotation_position="top left")
    fig.add_vrect(x0=25, x1=max(df['pe_ratio'].max(), 40), fillcolor="red", opacity=0.1,
                  annotation_text="Growth", annotation_position="top left")
    
    fig.update_layout(
        title="P/E Ratio Analysis<br><sub>Lower P/E = Value stocks | Higher P/E = Growth stocks</sub>",
        xaxis_title="Price-to-Earnings Ratio",
        yaxis_title="",
        height=max(400, len(df) * 40),
        showlegend=False
    )
    
    return fig


# Streamlit App
st.set_page_config(
    page_title="Portfolio Analysis Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Portfolio Analysis Dashboard")

with st.expander('About this tool'):
    st.write("""
    This tool provides comprehensive portfolio analysis including:
    - **Diversification Analysis**: Sector allocation and concentration metrics
    - **Valuation Analysis**: P/E ratios and value vs growth tilt
    - **AI Insights**: Personalized recommendations from GPT-4
    
    Enter your holdings in the sidebar and click "Analyze Portfolio" to begin.
    """)

# Sidebar
with st.sidebar:
    st.header("Portfolio Input", divider=True)
    
    # Budget status
    try:
        analyzer = PortfolioAnalyzer()
        allowed, remaining = analyzer.check_budget()
        st.info(f"💰 Budget: {remaining} analyses remaining")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
    
    # Portfolio input
    df = pd.DataFrame([
        {"ticker": "", "shares": 0.0},
        {"ticker": "", "shares": 0.0},
        {"ticker": "", "shares": 0.0}
    ])
    
    edited_df = st.data_editor(
        df, 
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", help="Stock symbol (e.g., AAPL)"),
            "shares": st.column_config.NumberColumn("Shares", help="Number of shares owned", format="%.4f")
        }
    )
    
    st.markdown("*Add stock tickers and number of shares*")
    
    run_analysis = st.button("🔍 Analyze Portfolio", type="primary", use_container_width=True)

# Main content
if run_analysis:
    # Filter out empty rows
    holdings_raw = edited_df.to_dict(orient="records")
    holdings = [h for h in holdings_raw if h.get('ticker') and h.get('shares', 0) > 0]
    
    if not holdings:
        st.error("⚠️ Please add at least one stock with ticker and shares.")
    else:
        # Validate
        valid, msg = validate_holdings(holdings)
        if not valid:
            st.error(f"⚠️ Invalid portfolio: {msg}")
        else:
            try:
                with st.spinner("🔄 Analyzing your portfolio..."):
                    # Run main analysis
                    analyzer = PortfolioAnalyzer()
                    result = analyzer.analyze(holdings)
                    
                    # Convert to dataframes
                    portfolio_df = pd.DataFrame.from_dict(result['portfolio'])
                    sectors_df = pd.DataFrame.from_dict(result['sectors'])
                    comparison_df = pd.DataFrame.from_dict(result['comparison'])
                    analysis = result['analysis']
                    concentration = result['concentration']
                    errors = result['errors']
                    
                    # Calculate additional metrics

                    

                    
                    with st.spinner("💵 Analyzing valuations..."):
                        valuation_df, weighted_pe = analyze_valuation(portfolio_df)
                
                st.success("✅ Analysis complete!")
                
                # Display errors if any
                if errors:
                    with st.expander("⚠️ Warnings"):
                        for error in errors:
                            st.warning(error)
                
                # Create tabs for different analyses
                tab1, tab4 = st.tabs([
                    "📈 Overview",  
                    "💰 Valuation (P/E)"
                ])
                
                # Tab 1: Overview
                with tab1:
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Portfolio Value", f"${sectors_df['value'].sum():,.2f}")
                    col2.metric("Total Holdings", concentration['num_holdings'])
                    col3.metric("Sectors", sectors_df['sector'].nunique())
                    col4.metric("Top 5 Concentration", f"{concentration['top_5']:.1f}%")
                    
                    st.divider()
                    
                    # Sector breakdown
                    st.subheader("Sector Allocation")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(
                            sectors_df,
                            values='pct',
                            names='sector',
                            title='Portfolio by Sector'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.dataframe(
                            sectors_df[['sector', 'pct', 'value']].style.format({
                                'pct': '{:.1f}%',
                                'value': '${:,.0f}'
                            },na_rep='N/A'),
                            hide_index=True,
                            use_container_width=True
                        )
                    
                    st.divider()
                    
                    # Holdings breakdown
                    st.subheader("Holdings Breakdown")
                    st.dataframe(
                        portfolio_df[['ticker', 'name', 'shares', 'price', 'value', 'pct']].style.format({
                            'shares': '{:.4f}',
                            'price': '${:.2f}',
                            'value': '${:,.2f}',
                            'pct': '{:.2f}%'
                        },na_rep='N/A'),
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    st.divider()
                    st.subheader('Treemap of investment portfolio')           
                    
                    category_codes, unique_categories = pd.factorize(portfolio_df["sector"])
                    colors = [cmap(code) for code in category_codes]
                    # customize the labels
                    labels = [
                        f"{ticker}\n${value:.4g}\n{pct}%"
                        for ticker, pct, value in zip(portfolio_df["ticker"], portfolio_df["pct"], portfolio_df["value"])
                    ]

                    # create a treemap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.set_axis_off()
                    squarify.plot(
                        sizes=portfolio_df["pct"],
                        label=labels,
                        color=colors,
                        text_kwargs={"color": "white", "fontsize": 9, "fontweight": "bold"},
                        pad=True,
                        ax=ax,
                    )

                    # add a title and legend
                    text = """<Portfolio treemap>
                    Each color represents a different sector.
                    Each rectangle represents a stock.
                    """
                    fig_text(
                        x=0.133,
                        y=0.98,
                        s=text,
                        color="black",
                        highlight_textprops=[
                            {"fontsize": 20, "fontweight": "bold"}

                        ],
                        fontsize=14,
                        ha="left",
                    )
                    st.pyplot(fig, use_container_width=True)

                                    
    



                    
                    # AI Analysis
                    st.subheader("🤖 AI Portfolio Analysis")
                    st.markdown(analysis)
                
                
                # Tab 4: Valuation
                with tab4:
                    st.subheader("Valuation Analysis (P/E Ratios)")
                    
                    if weighted_pe is not None:
                        col1, col2, col3 = st.columns(3)
                        
                        col1.metric(
                            "Portfolio P/E Ratio",
                            f"{weighted_pe:.1f}",
                            help="Weighted average P/E of your holdings"
                        )
                        
                        sp500_pe = 22.1
                        diff = weighted_pe - sp500_pe
                        col2.metric(
                            "vs S&P 500",
                            f"{sp500_pe:.1f}",
                            delta=f"{diff:+.1f}",
                            help="S&P 500 average P/E ratio"
                        )
                        
                        if weighted_pe < 15:
                            interpretation = "Value tilt"
                            color = "🟢"
                        elif weighted_pe > 25:
                            interpretation = "Growth tilt"
                            color = "🔴"
                        else:
                            interpretation = "Balanced"
                            color = "🟡"
                        
                        col3.metric("Style", f"{color} {interpretation}")
                        
                        st.info("""
                        **What is P/E Ratio?**
                        - Price-to-Earnings ratio measures how expensive a stock is
                        - P/E < 15: Value stocks (cheaper, stable companies)
                        - P/E 15-25: Fairly valued
                        - P/E > 25: Growth stocks (expensive, high-growth companies)
                        - Lower P/E generally means less risk but slower growth
                        """)
                        
                        st.divider()
                        
                        # P/E comparison chart
                        fig = pe_ratio_comparison(valuation_df, portfolio_df)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.divider()
                        
                        # Valuation table
                        st.subheader("Individual Stock Valuations")
                        val_table = valuation_df.merge(portfolio_df[['ticker', 'pct']], on='ticker')
                        val_table = val_table.sort_values('pe_ratio', ascending=True)
                        
                        st.dataframe(
                            val_table[['ticker', 'pe_ratio', 'forward_pe', 'peg_ratio', 'pct']].style.format({
                                'pe_ratio': '{:.1f}',
                                'forward_pe': '{:.1f}',
                                'peg_ratio': '{:.2f}',
                                'pct': '{:.1f}%'
                            }, na_rep='N/A'),
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        st.caption("""
                        **PEG Ratio**: P/E divided by growth rate. PEG < 1.0 = Potentially undervalued  
                        **Forward P/E**: Expected P/E based on future earnings estimates
                        """)
                    else:
                        st.warning("⚠️ Unable to calculate P/E ratios for portfolio. Some stocks may not have P/E data.")
                        
                        # Show whatever data is available
                        val_display = valuation_df[valuation_df['pe_ratio'].notna()]
                        if not val_display.empty:
                            st.dataframe(
                                val_display[['ticker', 'pe_ratio', 'weight']].style.format({
                                    'pe_ratio': '{:.1f}',
                                    'weight': '{:.1f}%'
                                },na_rep='N/A'),
                                hide_index=True,
                                use_container_width=True
                            )
                
            except Exception as e:
                st.error(f" Analysis failed: {str(e)}")
                st.exception(e)