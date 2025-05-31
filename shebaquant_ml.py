# ShebaQuant Enterprise - ML-Enabled Version
import streamlit as st
import numpy as np
from fpdf import FPDF
from datetime import datetime
import pandas as pd
import altair as alt
import joblib
import base64
import os
import tempfile
from typing import List, Tuple

# ======================
# Constants & Config
# ======================
INDUSTRIES = ["Tech", "Biotech", "Consumer", "Industrial", "Energy"]
INDUSTRY_DEFAULTS = {
    "Tech": {"growth": 0.25, "margin": 0.30, "wacc": 0.12},
    "Biotech": {"growth": 0.18, "margin": 0.22, "wacc": 0.15},
    "Consumer": {"growth": 0.08, "margin": 0.15, "wacc": 0.10},
    "Industrial": {"growth": 0.06, "margin": 0.12, "wacc": 0.09},
    "Energy": {"growth": 0.04, "margin": 0.18, "wacc": 0.07},
}

# ======================
# Valuation Engine
# ======================
class ValuationEngine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def ai_risk_score(revenue: float, industry: str, margin: float) -> float:
        """ML risk model loader and predictor"""
        model = joblib.load("risk_model.pkl")  # Must be in the same directory
        df = pd.DataFrame([{
            "Revenue": revenue,
            "Margin": margin,
            "Industry": industry
        }])
        return float(model.predict(df)[0])

    @staticmethod
    def forecast_revenue(base_revenue: float, growth_rates: List[float]) -> List[float]:
        return [base_revenue * np.prod([1 + g for g in growth_rates[:i+1]]) for i in range(len(growth_rates))]

    @staticmethod
    def calculate_dcf(base_revenue: float, growth_rates: List[float], margin_forecast: List[float],
                      discount_rate: float, terminal_growth: float, years: int) -> Tuple:
        revenue_forecast = ValuationEngine.forecast_revenue(base_revenue, growth_rates)
        cash_flows = [r * m for r, m in zip(revenue_forecast, margin_forecast)]
        discount_factors = [1 / ((1 + discount_rate) ** (i+1)) for i in range(years)]
        discounted_cf = [cf * df for cf, df in zip(cash_flows, discount_factors)]
        terminal_value = (cash_flows[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        discounted_terminal = terminal_value * discount_factors[-1]
        enterprise_value = sum(discounted_cf) + discounted_terminal
        return enterprise_value, revenue_forecast, cash_flows, discounted_cf, terminal_value, discounted_terminal

# ======================
# Main App
# ======================
def main():
    st.set_page_config(page_title="ShebaQuant ML", layout="wide")
    st.title("ShebaQuant Enterprise 🧠")
    st.markdown("An AI-powered business valuation platform using machine learning.")

    with st.sidebar:
        st.image("image.png", width=200)
        industry = st.selectbox("Industry", INDUSTRIES)
        base_revenue = st.number_input("Base Revenue ($)", value=500000)
        forecast_years = st.slider("Forecast Years", 3, 10, 5)
        terminal_growth = st.slider("Terminal Growth (%)", 1.0, 5.0, 3.0) / 100

    st.subheader("Growth Rates")
    growth_rates = [st.slider(f"Year {i+1} Growth (%)", 0.01, 0.50, INDUSTRY_DEFAULTS[industry]["growth"]) for i in range(forecast_years)]

    st.subheader("Margin Forecast")
    margin_forecast = [st.slider(f"Year {i+1} Margin (%)", 0.05, 0.50, INDUSTRY_DEFAULTS[industry]["margin"]) for i in range(forecast_years)]

    if st.button("Run Valuation"):
        risk_adj = ValuationEngine.ai_risk_score(base_revenue, industry, np.mean(margin_forecast))
        discount_rate = INDUSTRY_DEFAULTS[industry]["wacc"] + risk_adj

        valuation, revs, cfs, dcfs, tv, dtv = ValuationEngine.calculate_dcf(
            base_revenue, growth_rates, margin_forecast,
            discount_rate, terminal_growth, forecast_years
        )

        df = pd.DataFrame({
            "Year": [f"Year {i+1}" for i in range(forecast_years)],
            "Revenue": revs,
            "Cash Flow": cfs,
            "Discounted CF": dcfs
        }).set_index("Year")

        st.success(f"Enterprise Value: ${valuation:,.2f}")
        st.metric("AI Risk Score", f"{risk_adj*100:.2f}%")
        st.dataframe(df.style.format("${:,.0f}"))

if __name__ == "__main__":
    main()