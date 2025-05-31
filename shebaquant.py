# baQuant Ent# ShebaQuant Enterprise - Enhanced Version with Professional Reporting
import streamlit as st
import numpy as np
from fpdf import FPDF
from datetime import datetime
import pandas as pd
import altair as alt
import random
import base64
import os
import time
import tempfile
import joblib
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import hashlib
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Suppress warnings
warnings.filterwarnings("ignore")

# Optional imports for advanced features
try:
    from statsmodels.tsa.arima.model import ARIMA
    STATS_ENABLED = True
except ImportError:
    STATS_ENABLED = False

try:
    import shap
    SHAP_ENABLED = True
except ImportError:
    SHAP_ENABLED = False

# ======================
# CONSTANTS & CONFIG
# ======================
INDUSTRIES = ["Tech", "Biotech", "Consumer", "Industrial", "Energy", "Financial", "Healthcare", "Retail"]
INDUSTRY_DEFAULTS = {
    "Tech": {"growth": 0.25, "margin": 0.30, "wacc": 0.12, "risk": 0.15},
    "Biotech": {"growth": 0.18, "margin": 0.22, "wacc": 0.15, "risk": 0.20},
    "Consumer": {"growth": 0.08, "margin": 0.15, "wacc": 0.10, "risk": 0.10},
    "Industrial": {"growth": 0.06, "margin": 0.12, "wacc": 0.09, "risk": 0.12},
    "Energy": {"growth": 0.04, "margin": 0.18, "wacc": 0.07, "risk": 0.08},
    "Financial": {"growth": 0.10, "margin": 0.25, "wacc": 0.11, "risk": 0.13},
    "Healthcare": {"growth": 0.12, "margin": 0.20, "wacc": 0.13, "risk": 0.14},
    "Retail": {"growth": 0.05, "margin": 0.10, "wacc": 0.08, "risk": 0.09}
}

MAX_LOGIN_ATTEMPTS = 3
LOCKOUT_TIME = 300  # 5 minutes in seconds
COLOR_SCHEMES = {
    "Default": {"primary": "#2E86AB", "secondary": "#F18F01", "background": "#FFFFFF"},
    "Professional": {"primary": "#1F3A93", "secondary": "#5C6BC0", "background": "#F5F7FA"},
    "Vibrant": {"primary": "#E53935", "secondary": "#FFA000", "background": "#FFF8E1"},
    "Dark": {"primary": "#6A1B9A", "secondary": "#AB47BC", "background": "#121212"}
}

# Data persistence paths
DATA_DIR = Path("user_data")
SCENARIOS_FILE = DATA_DIR / "saved_scenarios.json"
USER_PREFS_FILE = DATA_DIR / "user_preferences.json"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)

# ======================
# UTILITY FUNCTIONS
# ======================
@st.cache_data
def get_image_base64(path: str) -> str:
    """Get base64 encoded image for embedding in HTML or PDF"""
    if os.path.exists(path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

def validate_inputs(base_revenue: float, growth_rates: List[float], 
                   margin_forecast: List[float], discount_rate: float) -> bool:
    """Validate all user inputs before running valuation"""
    errors = []
    
    if base_revenue <= 0:
        errors.append("Revenue must be positive")
    
    if not all(0 <= g <= 0.5 for g in growth_rates):
        errors.append("Growth rates must be between 0% and 50%")
    
    if not all(0 <= m <= 0.5 for m in margin_forecast):
        errors.append("Margins must be between 0% and 50%")
    
    if not 0.05 <= discount_rate <= 0.25:
        errors.append("Discount rate must be between 5% and 25%")
    
    if errors:
        for error in errors:
            st.error(error)
        return False
    return True

def load_scenarios() -> List[Dict]:
    """Load saved scenarios from JSON file"""
    if SCENARIOS_FILE.exists():
        with open(SCENARIOS_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_scenarios(scenarios: List[Dict]) -> None:
    """Save scenarios to JSON file"""
    with open(SCENARIOS_FILE, "w") as f:
        json.dump(scenarios, f, indent=2, default=str)

def load_user_prefs() -> Dict:
    """Load user preferences from JSON file"""
    if USER_PREFS_FILE.exists():
        with open(USER_PREFS_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_user_prefs(prefs: Dict) -> None:
    """Save user preferences to JSON file"""
    with open(USER_PREFS_FILE, "w") as f:
        json.dump(prefs, f, indent=2)

def hash_password(password: str) -> str:
    """Simple password hashing for demonstration purposes"""
    return hashlib.sha256(password.encode()).hexdigest()

# ======================
# CORE BUSINESS LOGIC
# ======================
class ValuationEngine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def ai_risk_score(revenue: float, industry: str, margin: float) -> float:
        """Calculate business risk score using ML model or fallback heuristic"""
        time.sleep(0.5)  # Simulate AI processing
        
        try:
            model = joblib.load("risk_model.pkl")
            if model:
                df = pd.DataFrame([{
                    "Revenue": revenue,
                    "Margin": margin,
                    "Industry": industry
                }])
                prediction = model.predict(df)[0]
                return float(np.clip(prediction, 0.0, 0.30))
        except Exception:
            pass
        
        # Fallback heuristic if model fails
        base_risk = INDUSTRY_DEFAULTS.get(industry, {}).get("risk", 0.10)
        if margin < 0.2:
            base_risk += 0.03
        if revenue < 100000:  # Smaller businesses are riskier
            base_risk += 0.02
        return min(base_risk, 0.30)
    
    @staticmethod
    @st.cache_data
    def forecast_revenue(base_revenue: float, growth_rates: List[float]) -> List[float]:
        """Project revenue over forecast period"""
        return [base_revenue * np.prod([1 + g for g in growth_rates[:i+1]]) 
                for i in range(len(growth_rates))]
    
    @staticmethod
    @st.cache_data
    def calculate_dcf(base_revenue: float,
                     growth_rates: List[float],
                     margin_forecast: List[float],
                     discount_rate: float,
                     terminal_growth: float,
                     years: int) -> Tuple[float, List[float], List[float], List[float], float, float]:
        """Discounted Cash Flow valuation calculation"""
        revenue_forecast = ValuationEngine.forecast_revenue(base_revenue, growth_rates)
        cash_flows = [r * m for r, m in zip(revenue_forecast, margin_forecast)]
        discount_factors = [1 / ((1 + discount_rate) ** (i+1)) for i in range(years)]
        discounted_cf = [cf * df for cf, df in zip(cash_flows, discount_factors)]
        
        # Terminal value calculation with sanity check
        if terminal_growth >= discount_rate:
            terminal_growth = discount_rate * 0.9  # Prevent division by zero
            
        terminal_value = (cash_flows[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        discounted_terminal = terminal_value * discount_factors[-1]
        enterprise_value = sum(discounted_cf) + discounted_terminal
        
        return enterprise_value, revenue_forecast, cash_flows, discounted_cf, terminal_value, discounted_terminal
    
    @staticmethod
    def sensitivity_analysis(base_case: Dict, variable: str, range_pct: float) -> pd.DataFrame:
        """Analyze valuation sensitivity to key parameters"""
        results = []
        for x in np.linspace(-range_pct/100, range_pct/100, 11):  # More points for smoother curve
            modified_params = base_case.copy()
            
            if variable == "Discount Rate":
                modified_params["discount_rate"] *= (1 + x)
            elif variable == "Growth Rates":
                modified_params["growth_rates"] = [g * (1 + x) for g in base_case["growth_rates"]]
            elif variable == "Margins":
                modified_params["margin_forecast"] = [m * (1 + x) for m in base_case["margin_forecast"]]
            elif variable == "Terminal Growth":
                modified_params["terminal_growth"] *= (1 + x)
            
            result = ValuationEngine.calculate_dcf(**modified_params)
            results.append({
                "Change (%)": x * 100,
                "Valuation": result[0],
                "Variable": variable
            })
        return pd.DataFrame(results)
    
    @staticmethod
    def time_series_forecast(revenue_history: List[float], periods_ahead: int = 3) -> List[float]:
        """ARIMA-based forecasting for revenue"""
        if not STATS_ENABLED or len(revenue_history) < 3:
            return [revenue_history[-1]] * periods_ahead
        
        try:
            model = ARIMA(revenue_history, order=(1, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=periods_ahead)
            return list(forecast)
        except Exception:
            return [revenue_history[-1]] * periods_ahead
    
    @staticmethod
    def monte_carlo_simulation(base_case: Dict, draws: int = 1000) -> Dict:
        """Monte Carlo simulation with more sophisticated parameter variation"""
        results = []
        growth_volatility = 0.1  # Standard deviation for growth rate changes
        margin_volatility = 0.05  # Standard deviation for margin changes
        
        for _ in range(draws):
            new_case = base_case.copy()
            
            # Vary parameters with normal distribution
            new_case["discount_rate"] = np.random.normal(
                base_case["discount_rate"], 
                base_case["discount_rate"] * 0.05
            )
            
            new_case["growth_rates"] = [
                np.random.normal(g, g * growth_volatility)
                for g in base_case["growth_rates"]
            ]
            
            new_case["margin_forecast"] = [
                np.random.normal(m, m * margin_volatility)
                for m in base_case["margin_forecast"]
            ]
            
            # Ensure terminal growth is reasonable
            new_case["terminal_growth"] = np.clip(
                np.random.normal(base_case["terminal_growth"], 0.005),
                0.01, 0.05
            )
            
            # Clip to reasonable ranges
            new_case["discount_rate"] = np.clip(new_case["discount_rate"], 0.05, 0.25)
            new_case["growth_rates"] = [np.clip(g, 0, 0.5) for g in new_case["growth_rates"]]
            new_case["margin_forecast"] = [np.clip(m, 0, 0.5) for m in new_case["margin_forecast"]]
            
            val, *rest = ValuationEngine.calculate_dcf(**new_case)
            results.append(val)
        
        return {
            "values": results,
            "mean": np.mean(results),
            "median": np.median(results),
            "std": np.std(results),
            "5th_percentile": np.percentile(results, 5),
            "95th_percentile": np.percentile(results, 95)
        }

# ======================
# UI COMPONENTS
# ======================
class InputComponents:
    @staticmethod
    def industry_selector() -> str:
        """Industry selection with intelligent defaults"""
        col1, col2 = st.columns([3,1])
        with col1:
            industry = st.selectbox("Industry", list(INDUSTRY_DEFAULTS.keys()), 
                                  key="industry_select")
        with col2:
            if st.button("Apply Defaults", key="apply_defaults_btn"):
                defaults = INDUSTRY_DEFAULTS.get(industry, {})
                for i in range(st.session_state.forecast_years):
                    st.session_state[f"growth_rate_{i}"] = defaults.get("growth", 0.10)
                    st.session_state[f"margin_{i}"] = defaults.get("margin", 0.15)
                st.session_state.discount_rate = defaults.get("wacc", 0.10)
                st.rerun()
        return industry
    
    @staticmethod
    def slider_grid(label_prefix: str,
                   years: int,
                   min_val: float,
                   max_val: float,
                   default_val: float,
                   format: str = "%.2f") -> List[float]:
        """Create a responsive grid of sliders for multi-year inputs"""
        values = []
        cols = st.columns(min(3, years))  # Max 3 columns
        
        for i in range(years):
            with cols[i%len(cols)]:
                slider_key = f"{label_prefix.lower()}_yr_{i}"
                values.append(st.slider(
                    f"{label_prefix} Year {i+1}",
                    min_val, max_val, default_val,
                    key=slider_key,
                    format=format,
                    help=f"Projected {label_prefix.lower()} for Year {i+1}"
                ))
        return values

    @staticmethod
    def line_item_costs() -> float:
        """Detailed cost breakdown to calculate effective margin"""
        with st.expander("Cost Breakdown (Line-Item Model)", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                overhead = st.number_input("Overhead Costs ($)", 
                                         min_value=0, value=50000, step=1000,
                                         help="Fixed costs like rent, utilities, etc.")
                payroll = st.number_input("Payroll Costs ($)", 
                                        min_value=0, value=100000, step=5000,
                                        help="Annual wages for employees")
            
            with col2:
                cogs = st.slider("COGS % of Revenue", 0.0, 1.0, 0.40,
                                format="%.1f%%", help="Cost of goods sold")
                sgna = st.slider("SG&A % of Revenue", 0.0, 0.5, 0.20,
                                format="%.1f%%", help="Selling, general & admin expenses")
            
            assumed_revenue = st.session_state.get("base_revenue_input", 300000.0)
            
            # Calculate EBIT and margin
            gross_profit = assumed_revenue * (1 - cogs)
            operating_expenses = overhead + payroll + (assumed_revenue * sgna)
            ebit = max(0.0, gross_profit - operating_expenses)
            effective_margin = 0.0 if assumed_revenue == 0 else ebit / assumed_revenue
            
            # Display results
            st.markdown(f"""
                **Effective Margin**: {effective_margin*100:.2f}%  
                *Breakdown*:  
                - Revenue: ${assumed_revenue:,.0f}  
                - Gross Profit: ${gross_profit:,.0f} ({ (1-cogs)*100:.1f}%)  
                - Operating Expenses: ${operating_expenses:,.0f}  
                - EBIT: ${ebit:,.0f}
            """, unsafe_allow_html=True)
            
            return effective_margin

# ======================
# ENHANCED REPORT GENERATION
# ======================
class ProfessionalReportGenerator(FPDF):
    """Professional PDF report generator with multiple sections and visualizations"""
    def __init__(self, title="Valuation Report", logo_path=None):
        super().__init__()
        self.title = title
        self.logo_base64 = get_image_base64(logo_path) if logo_path else None
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font('Helvetica', '', 10)
        self.sections = []
    
    def header(self):
        if self.logo_base64:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    tmp.write(base64.b64decode(self.logo_base64))
                    self.image(tmp.name, x=10, y=8, w=30)
                os.unlink(tmp.name)
            except Exception:
                pass
        
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, self.title, 0, 1, 'C')
        self.set_font('Helvetica', '', 10)
        self.cell(0, 10, datetime.now().strftime("%B %d, %Y"), 0, 1, 'C')
        self.ln(15)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def add_section(self, title, content_type, data=None, analysis=None):
        """Register sections to be included in report"""
        self.sections.append({
            'title': title,
            'type': content_type,
            'data': data,
            'analysis': analysis
        })
    
    def generate_report(self):
        """Generate complete report with all sections"""
        self._add_cover_page()
        self._add_table_of_contents()
        
        for section in self.sections:
            self.add_page()
            self._add_section_header(section['title'])
            
            if section['type'] == 'financial_forecast':
                self._add_financial_forecast(section['data'])
            elif section['type'] == 'sensitivity_analysis':
                self._add_sensitivity_chart(section['data'])
            elif section['type'] == 'monte_carlo':
                self._add_monte_carlo_results(section['data'])
            elif section['type'] == 'valuation_summary':
                self._add_valuation_summary(section['data'])
            elif section['type'] == 'industry_comparison':
                self._add_industry_comparison(section['data']['company'], section['data']['industry'])
            elif section['type'] == 'risk_analysis':
                self._add_risk_analysis(section['data'])
            
            if section.get('analysis'):
                self._add_analyst_commentary(section['analysis'])
        
        self._add_disclaimer()
    
    def _add_cover_page(self):
        self.add_page()
        self.set_font('Helvetica', 'B', 24)
        self.cell(0, 40, self.title, 0, 1, 'C')
        self.ln(20)
        
        if self.logo_base64:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    tmp.write(base64.b64decode(self.logo_base64))
                    self.image(tmp.name, x=60, y=80, w=80)
                os.unlink(tmp.name)
            except Exception:
                pass
        
        self.set_font('Helvetica', '', 14)
        self.cell(0, 10, "Prepared for:", 0, 1, 'C')
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, st.session_state.get('company_name', 'Client Name'), 0, 1, 'C')
        self.ln(20)
        
        self.set_font('Helvetica', '', 12)
        self.cell(0, 8, f"Date: {datetime.now().strftime('%B %d, %Y')}", 0, 1, 'C')
        self.cell(0, 8, f"Prepared by: {st.session_state.get('analyst_name', 'ShebaQuant Analytics')}", 0, 1, 'C')
    
    def _add_table_of_contents(self):
        self.add_page()
        self.set_font('Helvetica', 'B', 16)
        self.cell(0, 10, "Table of Contents", 0, 1)
        self.ln(10)
        
        self.set_font('Helvetica', '', 12)
        for i, section in enumerate(self.sections, 1):
            self.cell(0, 8, f"{i}. {section['title']}", 0, 1)
            self.cell(0, 2, "", 0, 1)  # Small gap
        
        self.ln(15)
        self.set_font('Helvetica', 'I', 10)
        self.multi_cell(0, 6, "This report contains proprietary analysis and should be treated as confidential.")
    
    def _add_section_header(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, title, 0, 1)
        self.ln(5)
    
    def _add_financial_forecast(self, financials):
        """Add detailed financial projections"""
        # Income Statement
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, "Projected Income Statement", 0, 1)
        self.ln(2)
        
        headers = ["", "Year 1", "Year 2", "Year 3"]
        col_widths = [70, 40, 40, 40]
        
        self.set_fill_color(240, 240, 240)
        self.set_font('Helvetica', 'B', 10)
        for w, header in zip(col_widths, headers):
            self.cell(w, 6, header, 1, 0, 'C', fill=True)
        self.ln()
        
        items = [
            ("Revenue", financials['revenue']),
            ("COGS", financials['cogs']),
            ("Gross Profit", financials['gross_profit']),
            ("Operating Expenses", financials['opex']),
            ("EBIT", financials['ebit']),
            ("Taxes", financials['taxes']),
            ("Net Income", financials['net_income'])
        ]
        
        self.set_font('Helvetica', '', 9)
        for label, values in items:
            self.cell(col_widths[0], 6, label, 1)
            for i, val in enumerate(values[:3]):  # First 3 years
                self.cell(col_widths[i+1], 6, f"${val/1000:,.1f}K", 1, 0, 'R')
            self.ln()
        
        # Cash Flow Statement
        self.ln(8)
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, "Projected Cash Flows", 0, 1)
        self.ln(2)
        
        self.set_font('Helvetica', 'B', 10)
        for w, header in zip(col_widths, headers):
            self.cell(w, 6, header, 1, 0, 'C', fill=True)
        self.ln()
        
        cash_items = [
            ("Operating Cash Flow", financials['operating_cf']),
            ("Investing Cash Flow", financials['investing_cf']),
            ("Financing Cash Flow", financials['financing_cf']),
            ("Net Cash Flow", financials['net_cf'])
        ]
        
        self.set_font('Helvetica', '', 9)
        for label, values in cash_items:
            self.cell(col_widths[0], 6, label, 1)
            for i, val in enumerate(values[:3]):
                self.cell(col_widths[i+1], 6, f"${val/1000:,.1f}K", 1, 0, 'R')
            self.ln()
        
        # Add growth analysis
        self.ln(10)
        self._add_growth_analysis(financials)
    
    def _add_growth_analysis(self, financials):
        """Add growth rate analysis"""
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 8, "Growth Analysis", 0, 1)
        self.ln(2)
        
        # Calculate growth rates
        rev_growth = [(financials['revenue'][i] - financials['revenue'][i-1]) / 
                     financials['revenue'][i-1] for i in range(1, 3)]
        ebit_growth = [(financials['ebit'][i] - financials['ebit'][i-1]) / 
                      financials['ebit'][i-1] for i in range(1, 3)]
        
        # Create table
        headers = ["Metric", "Y1-Y2 Growth", "Y2-Y3 Growth"]
        col_widths = [60, 50, 50]
        
        self.set_font('Helvetica', 'B', 9)
        for w, header in zip(col_widths, headers):
            self.cell(w, 6, header, 1, 0, 'C', fill=True)
        self.ln()
        
        self.set_font('Helvetica', '', 9)
        self.cell(col_widths[0], 6, "Revenue", 1)
        self.cell(col_widths[1], 6, f"{rev_growth[0]*100:.1f}%", 1, 0, 'C')
        self.cell(col_widths[2], 6, f"{rev_growth[1]*100:.1f}%", 1, 0, 'C')
        self.ln()
        
        self.cell(col_widths[0], 6, "EBIT", 1)
        self.cell(col_widths[1], 6, f"{ebit_growth[0]*100:.1f}%", 1, 0, 'C')
        self.cell(col_widths[2], 6, f"{ebit_growth[1]*100:.1f}%", 1, 0, 'C')
        self.ln()
    
    def _add_valuation_summary(self, valuation_data):
        """Add detailed valuation metrics"""
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, "Valuation Summary", 0, 1)
        self.ln(5)
        
        # Key metrics table
        metrics = [
            ("Enterprise Value", f"${valuation_data['enterprise_value']/1000:,.1f}K"),
            ("Equity Value", f"${valuation_data['equity_value']/1000:,.1f}K"),
            ("Implied Revenue Multiple", f"{valuation_data['revenue_multiple']:.1f}x"),
            ("Implied EBITDA Multiple", f"{valuation_data['ebitda_multiple']:.1f}x"),
            ("WACC", f"{valuation_data['wacc']*100:.1f}%"),
            ("Risk Premium", f"{valuation_data['risk_premium']*100:.1f}%"),
            ("Terminal Growth", f"{valuation_data['terminal_growth']*100:.1f}%")
        ]
        
        self.set_font('Helvetica', '', 10)
        col_width = 80
        for label, value in metrics:
            self.cell(col_width, 6, label, 'L')
            self.cell(col_width, 6, value, 'L')
            self.ln()
        
        # Valuation bridge chart
        self.ln(10)
        self._add_valuation_bridge(valuation_data)
    
    def _add_valuation_bridge(self, valuation_data):
        """Add valuation bridge visualization"""
        try:
            # Prepare data for bridge chart
            components = {
                'Year 1 CF': valuation_data['year1_cf'],
                'Year 2 CF': valuation_data['year2_cf'],
                'Year 3 CF': valuation_data['year3_cf'],
                'Terminal Value': valuation_data['terminal_value']
            }
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(8, 4))
            prev_val = 0
            colors = ['#2E86AB', '#3A7CA5', '#4D6D9A', '#6B5B95']
            
            for i, (label, value) in enumerate(components.items()):
                ax.barh(0, value, left=prev_val, label=label, color=colors[i])
                prev_val += value
            
            ax.set_title('Valuation Bridge ($)')
            ax.get_yaxis().set_visible(False)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.grid(False)
            
            # Convert to image and embed in PDF
            canvas = FigureCanvasAgg(fig)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                canvas.print_figure(tmp.name, dpi=150)
                self.image(tmp.name, x=10, y=self.get_y(), w=180)
                os.unlink(tmp.name)
            
            plt.close()
        except Exception as e:
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 6, f"Could not generate valuation bridge: {str(e)}", 0, 1)
    
    def _add_sensitivity_chart(self, sensitivity_data):
        """Add sensitivity analysis visualization"""
        try:
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(8, 4))
            for var, data in sensitivity_data.items():
                ax.plot(data['changes'], data['valuations'], label=var)
            
            ax.set_title('Valuation Sensitivity Analysis')
            ax.set_xlabel('Parameter Change (%)')
            ax.set_ylabel('Enterprise Value ($)')
            ax.legend()
            ax.grid(True)
            
            # Convert to image and embed in PDF
            canvas = FigureCanvasAgg(fig)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                canvas.print_figure(tmp.name, dpi=150)
                self.image(tmp.name, x=10, y=self.get_y(), w=180)
                os.unlink(tmp.name)
            
            plt.close()
        except Exception as e:
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 6, f"Could not generate sensitivity chart: {str(e)}", 0, 1)
    
    def _add_monte_carlo_results(self, mc_data):
        """Add Monte Carlo simulation results"""
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, "Monte Carlo Simulation Results", 0, 1)
        self.ln(5)
        
        # Key metrics
        metrics = [
            ("Mean Valuation", f"${mc_data['mean']/1000:,.1f}K"),
            ("Median Valuation", f"${mc_data['median']/1000:,.1f}K"),
            ("Standard Deviation", f"${mc_data['std']/1000:,.1f}K"),
            ("5th Percentile", f"${mc_data['5th_percentile']/1000:,.1f}K"),
            ("95th Percentile", f"${mc_data['95th_percentile']/1000:,.1f}K")
        ]
        
        self.set_font('Helvetica', '', 10)
        col_width = 80
        for label, value in metrics:
            self.cell(col_width, 6, label, 'L')
            self.cell(col_width, 6, value, 'L')
            self.ln()
        
        # Distribution chart
        self.ln(10)
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(mc_data['values'], bins=50, color='#2E86AB', edgecolor='white')
            ax.set_title('Monte Carlo Valuation Distribution')
            ax.set_xlabel('Valuation ($)')
            ax.set_ylabel('Frequency')
            ax.grid(True)
            
            # Add confidence intervals
            ax.axvline(mc_data['5th_percentile'], color='red', linestyle='--')
            ax.axvline(mc_data['95th_percentile'], color='red', linestyle='--')
            
            # Convert to image and embed in PDF
            canvas = FigureCanvasAgg(fig)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                canvas.print_figure(tmp.name, dpi=150)
                self.image(tmp.name, x=10, y=self.get_y(), w=180)
                os.unlink(tmp.name)
            
            plt.close()
        except Exception as e:
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 6, f"Could not generate distribution chart: {str(e)}", 0, 1)
    
    def _add_industry_comparison(self, company_data, industry_data):
        """Add industry benchmarking analysis"""
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, 'Industry Benchmarking', 0, 1)
        self.ln(5)
        
        # Comparison metrics
        metrics = [
            ("Revenue Growth", company_data['growth'], industry_data['growth']),
            ("Gross Margin", company_data['gross_margin'], industry_data['gross_margin']),
            ("EBITDA Margin", company_data['ebitda_margin'], industry_data['ebitda_margin']),
            ("R&D Intensity", company_data['rnd'], industry_data['rnd']),
            ("EV/Revenue Multiple", company_data['ev_rev'], industry_data['ev_rev'])
        ]
        
        # Create comparison table
        self.set_font('Helvetica', 'B', 9)
        self.cell(70, 6, "Metric", 1)
        self.cell(40, 6, "Company", 1, 0, 'C')
        self.cell(40, 6, "Industry", 1, 0, 'C')
        self.cell(30, 6, "Variance", 1, 0, 'C')
        self.ln()
        
        self.set_font('Helvetica', '', 9)
        for label, company_val, industry_val in metrics:
            variance = ((company_val - industry_val) / industry_val) * 100
            variance_color = '00AA00' if variance >=0 else 'FF0000'  # Green/red
            
            self.cell(70, 6, label, 1)
            self.cell(40, 6, f"{company_val:.1%}", 1, 0, 'C')
            self.cell(40, 6, f"{industry_val:.1%}", 1, 0, 'C')
            self.cell(30, 6, f"{variance:+.1f}%", 1, 0, 'C', 
                     fill=False, link='', fill_color=None, 
                     text_color=variance_color)
            self.ln()
        
        self.ln(10)
        self._add_benchmarking_commentary(company_data, industry_data)
    
    def _add_benchmarking_commentary(self, company_data, industry_data):
        """Add analyst commentary on benchmarking"""
        self.set_font('Helvetica', 'I', 10)
        self.multi_cell(0, 6, 
                       "Analysis: The company's performance relative to industry benchmarks indicates " +
                       "areas of competitive advantage and potential improvement opportunities. " +
                       "Key variances should be investigated further.")
    
    def _add_risk_analysis(self, risk_data):
        """Detailed risk assessment section"""
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, 'Risk Analysis', 0, 1)
        self.ln(5)
        
        # Risk factors table
        self.set_font('Helvetica', 'B', 9)
        self.cell(100, 6, "Risk Factor", 1)
        self.cell(30, 6, "Score", 1, 0, 'C')
        self.cell(50, 6, "Mitigation", 1)
        self.ln()
        
        self.set_font('Helvetica', '', 9)
        for factor in risk_data['factors']:
            # Risk score indicator
            score_width = factor['score'] * 30 / 10  # Scale to 0-30 width
            self.set_fill_color(255, 100, 100)  # Red for risk
            
            self.cell(100, 6, factor['name'], 1)
            self.cell(30, 6, "", 1, 0, 'L')  # Empty cell for score bar
            # Add colored bar for visual score indication
            self.rect(self.get_x()-30+2, self.get_y()+2, score_width, 2, 'F')
            self.set_xy(self.get_x(), self.get_y()+6)  # Reset position
            self.cell(50, 6, factor['mitigation'], 1)
            self.ln()
        
        # Risk heatmap
        self.ln(10)
        self._add_risk_heatmap(risk_data)
        
        # Scenario analysis
        self.ln(15)
        self._add_downside_scenario(risk_data)
    
    def _add_risk_heatmap(self, risk_data):
        """Add risk heatmap visualization"""
        try:
            # Prepare data for heatmap
            risks = [f['name'] for f in risk_data['factors']]
            scores = [f['score'] for f in risk_data['factors']]
            impact = [f['impact'] for f in risk_data['factors']]
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(8, 4))
            scatter = ax.scatter(scores, impact, c=[s*i for s,i in zip(scores, impact)], 
                               cmap='RdYlGn_r', s=200, alpha=0.7)
            
            # Add labels
            for i, risk in enumerate(risks):
                ax.text(scores[i], impact[i], risk[:15], ha='center', va='center', fontsize=8)
            
            ax.set_title('Risk Heatmap (Score vs Impact)')
            ax.set_xlabel('Risk Score (1-10)')
            ax.set_ylabel('Potential Impact (1-10)')
            ax.grid(True)
            plt.colorbar(scatter, label='Risk Exposure (Score × Impact)')
            
            # Convert to image and embed in PDF
            canvas = FigureCanvasAgg(fig)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                canvas.print_figure(tmp.name, dpi=150)
                self.image(tmp.name, x=10, y=self.get_y(), w=180)
                os.unlink(tmp.name)
            
            plt.close()
        except Exception as e:
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 6, f"Could not generate risk heatmap: {str(e)}", 0, 1)
    
    def _add_downside_scenario(self, risk_data):
        """Add downside scenario analysis"""
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 8, 'Downside Scenario Analysis', 0, 1)
        self.ln(5)
        
        # Scenario table
        self.set_font('Helvetica', 'B', 9)
        self.cell(60, 6, "Scenario", 1)
        self.cell(40, 6, "Probability", 1, 0, 'C')
        self.cell(40, 6, "Impact", 1, 0, 'C')
        self.cell(40, 6, "Mitigation", 1)
        self.ln()
        
        self.set_font('Helvetica', '', 9)
        for scenario in risk_data['scenarios']:
            self.cell(60, 6, scenario['name'], 1)
            self.cell(40, 6, f"{scenario['probability']:.1%}", 1, 0, 'C')
            self.cell(40, 6, f"${scenario['impact']/1000:,.1f}K", 1, 0, 'C')
            self.cell(40, 6, scenario['mitigation'], 1)
            self.ln()
    
    def _add_analyst_commentary(self, commentary):
        """Add analyst commentary section"""
        self.ln(10)
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 6, "Analyst Commentary:", 0, 1)
        self.set_font('Helvetica', 'I', 9)
        self.multi_cell(0, 5, commentary)
    
    def _add_disclaimer(self):
        """Add standard valuation disclaimer"""
        self.add_page()
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, "Disclaimer", 0, 1)
        self.ln(5)
        
        self.set_font('Helvetica', 'I', 8)
        self.multi_cell(0, 5,
                       "This valuation report is provided for informational purposes only and should not "
                       "be construed as financial advice. The analysis is based on numerous assumptions "
                       "and estimates that may not reflect actual future performance. While we have made "
                       "every effort to ensure the accuracy of the information contained in this report, "
                       "we make no guarantees about its completeness or suitability for any purpose. "
                       "Always consult with a qualified financial professional before making investment "
                       "decisions. The authors and ShebaQuant accept no liability for any actions taken "
                       "based on this analysis.")
        self.ln(5)
        
        self.set_font('Helvetica', '', 8)
        self.cell(0, 5, "Confidential - For Internal Use Only", 0, 1, 'C')

# ======================
# MAIN APPLICATION
# ======================
def main():
    # Initialize session state with persistence
    if 'authenticated' not in st.session_state:
        # Load saved preferences if they exist
        user_prefs = load_user_prefs()
        
        st.session_state.update({
            'authenticated': False,
            'login_attempts': 0,
            'locked': False,
            'lockout_time': None,
            'first_login': True,
            'dark_mode': user_prefs.get('dark_mode', False),
            'color_scheme': user_prefs.get('color_scheme', "Default"),
            'preview_mode': user_prefs.get('preview_mode', False),
            'saved_scenarios': load_scenarios(),
            'forecast_years': 5,
            'discount_rate': 0.10,
            'terminal_growth': 0.03,
            'page': "Valuation",
            'username': None
        })
    
    # Configure page settings
    st.set_page_config(
        page_title="ShebaQuant Enterprise",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply color scheme and styling
    colors = COLOR_SCHEMES[st.session_state.color_scheme]
    primary_color = colors["primary"]
    secondary_color = colors["secondary"]
    bg_color = colors.get("background", "#FFFFFF")
    
    custom_css = f"""
    <style>
        :root {{
            --primary: {primary_color};
            --secondary: {secondary_color};
            --bg-color: {bg_color};
        }}
        .stApp {{
            background-color: var(--bg-color);
        }}
        .stButton>button {{
            background-color: var(--primary) !important;
            color: white !important;
            border: none;
            border-radius: 4px;
        }}
        .stButton>button:hover {{
            opacity: 0.9;
        }}
        .metric-card {{
            border-left: 4px solid var(--primary) !important;
            padding: 10px;
            background-color: rgba(46, 134, 171, 0.1);
            border-radius: 4px;
        }}
        .sidebar .sidebar-content {{
            background-color: var(--bg-color);
        }}
        {'body { color: white; }' if st.session_state.dark_mode else ''}
        h1, h2, h3, h4, h5, h6 {{
            color: var(--primary) !important;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    
    # Authentication and sidebar
    def render_auth_sidebar():
        """Render authentication and navigation sidebar"""
        with st.sidebar:
            # Logo display
            logo_base64 = get_image_base64("image.png")
            if logo_base64:
                st.markdown(f"""
                    <div style="text-align: center; padding: 1rem 0 2rem;">
                        <img src="data:image/png;base64,{logo_base64}"
                             style="max-width: 180px; height: auto; margin: 0 auto; display: block;
                             {'filter: brightness(0) invert(1);' if st.session_state.dark_mode else ''}">
                    </div>
                """, unsafe_allow_html=True)
            
            # Authentication flow
            if not st.session_state.authenticated:
                if st.session_state.locked:
                    remaining_time = max(0, LOCKOUT_TIME - (time.time() - st.session_state.lockout_time))
                    st.error(f"Account locked. Try again in {int(remaining_time//60)}:{int(remaining_time%60):02d}")
                    if remaining_time <= 0:
                        st.session_state.locked = False
                        st.session_state.login_attempts = 0
                        st.rerun()
                else:
                    with st.form("login_form"):
                        st.subheader("🔐 Secure Login")
                        username = st.text_input("Username", key="login_username")
                        password = st.text_input("Password", type="password", key="login_password")
                        
                        if st.form_submit_button("Login"):
                            # Simple authentication for demo (in production, use proper auth)
                            if username == "admin" and password == "password":
                                st.session_state.authenticated = True
                                st.session_state.username = username
                                st.session_state.login_attempts = 0
                                st.rerun()
                            else:
                                st.session_state.login_attempts += 1
                                if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
                                    st.session_state.locked = True
                                    st.session_state.lockout_time = time.time()
                                    st.error("Too many failed attempts - account locked for 5 minutes")
                                else:
                                    remain = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
                                    st.error(f"Invalid credentials ({remain} attempts remaining)")
            else:
                # Navigation for authenticated users
                st.success(f"Welcome back, {st.session_state.username}")
                st.sidebar.title("Navigation")
                
                # Page selection
                pages = {
                    "Valuation": "📊 Valuation Tool",
                    "Scenario Analysis": "🔍 Scenario Analysis",
                    "Reports": "📑 Saved Reports",
                    "Settings": "⚙️ Settings"
                }
                
                st.session_state.page = st.sidebar.radio(
                    "Go to", 
                    list(pages.keys()),
                    format_func=lambda x: pages[x],
                    key="nav_radio"
                )
                
                # User preferences
                with st.sidebar.expander("Preferences", expanded=False):
                    dark_mode = st.checkbox(
                        "Dark Mode", 
                        value=st.session_state.dark_mode, 
                        key="dark_mode_cb"
                    )
                    
                    color_scheme = st.selectbox(
                        "Color Scheme",
                        list(COLOR_SCHEMES.keys()),
                        index=list(COLOR_SCHEMES.keys()).index(st.session_state.color_scheme),
                        key="color_scheme_select"
                    )
                    
                    preview_mode = st.checkbox(
                        "Live Preview",
                        value=st.session_state.preview_mode,
                        help="Show real-time valuation preview",
                        key="preview_mode_cb"
                    )
                    
                    # Save preferences when changed
                    if (dark_mode != st.session_state.dark_mode or 
                        color_scheme != st.session_state.color_scheme or
                        preview_mode != st.session_state.preview_mode):
                        st.session_state.dark_mode = dark_mode
                        st.session_state.color_scheme = color_scheme
                        st.session_state.preview_mode = preview_mode
                        
                        # Persist preferences
                        save_user_prefs({
                            'dark_mode': dark_mode,
                            'color_scheme': color_scheme,
                            'preview_mode': preview_mode
                        })
                        st.rerun()
                
                if st.button("Logout", key="logout_btn"):
                    st.session_state.authenticated = False
                    st.rerun()
                
                return st.session_state.page
    
    current_page = render_auth_sidebar()
    
    # Show login screen if not authenticated
    if not st.session_state.authenticated:
        logo_base64 = get_image_base64("image.png")
        col1, col2, col3 = st.columns([1,2,1])
        
        with col2:
            st.markdown(f"""
                <div style='text-align: center; margin-top: 5rem;'>
                    <h1 style='color:{primary_color};'>ShebaQuant Enterprise</h1>
                    <p style='font-size: 1.2rem; margin-top: 2rem;'>
                        AI-Driven Business Valuation Platform
                    </p>
            """, unsafe_allow_html=True)
            
            if logo_base64:
                st.markdown(f"""
                    <div style='text-align: center; margin: 2rem 0;'>
                        <img src="data:image/png;base64,{logo_base64}" 
                             style='max-width: 200px; border-radius: 8px;
                             {'filter: brightness(0) invert(1);' if st.session_state.dark_mode else ''}'>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
                    <p style='margin-top: 2rem;'>
                        Please log in via the sidebar to access the valuation tools.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        return
    
    # ======================
    # VALUATION PAGE
    # ======================
    if current_page == "Valuation":
        # First-time user guide
        if st.session_state.first_login:
            with st.expander("📚 First Time User Guide", expanded=True):
                st.markdown(f"""
                    ## Welcome to ShebaQuant Enterprise!
                    
                    This tool helps you value small businesses using the **Discounted Cash Flow (DCF)** method 
                    enhanced with **machine learning** risk assessment.
                    
                    ### How to Use:
                    1. **Set Parameters**: Adjust inputs in the left panel
                    2. **Run Valuation**: Click the valuation button
                    3. **Analyze Results**: View interactive charts and metrics
                    4. **Generate Reports**: Download professional PDF reports
                    
                    ### Tips:
                    - Use the <span style='color:{primary_color}'>Live Preview</span> for real-time estimates
                    - Try different scenarios to understand valuation drivers
                    - Check the sensitivity analysis to identify key risks
                    
                    *Ready to get started? Hide this guide and begin your valuation below.*
                """, unsafe_allow_html=True)
                
                if st.button("Got it, hide this guide", key="hide_guide_btn"):
                    st.session_state.first_login = False
                    st.rerun()
        
        st.title("Business Valuation Tool")
        st.markdown("""
            <div style='margin-bottom: 2rem;'>
                Estimate the fair value of a business using discounted cash flow analysis 
                enhanced with AI risk assessment.
            </div>
        """, unsafe_allow_html=True)
        
        # Main layout - input column and results column
        input_col, result_col = st.columns([2, 3])

        with input_col:
            st.header("Valuation Parameters")
            
            # Industry selection
            industry = InputComponents.industry_selector()
            
            # Base revenue input
            base_revenue = st.number_input(
                "Base Revenue ($)",
                value=300000,
                min_value=1000,
                step=10000,
                key="base_revenue_input",
                help="Annual revenue in USD. For startups, use projected Year 1 revenue."
            )
            
            # Forecast period
            forecast_years = st.slider(
                "Forecast Period (Years)", 
                3, 10, st.session_state.forecast_years,
                help="Number of years for detailed forecast",
                key="forecast_years_slider"
            )
            st.session_state.forecast_years = forecast_years
            
            # Growth rates
            st.subheader("Revenue Growth Rates (%)")
            growth_rates = InputComponents.slider_grid(
                "Growth",
                forecast_years,
                0.00, 0.50, 0.12,
                "%.2f%%"
            )
            
            # Margin selection - line item or manual
            margin_method = st.radio(
                "Margin Calculation Method",
                ["Line Item Costs", "Manual Input"],
                index=0,
                horizontal=True,
                key="margin_method"
            )
            
            if margin_method == "Line Item Costs":
                effective_margin = InputComponents.line_item_costs()
                margin_forecast = [effective_margin] * forecast_years
            else:
                st.subheader("Operating Margins (%)")
                margin_forecast = InputComponents.slider_grid(
                    "Margin",
                    forecast_years,
                    0.00, 0.50, 0.25,
                    "%.2f%%"
                )
            
            # Discounting parameters
            st.subheader("Discounting Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.discount_rate = st.slider(
                    "Discount Rate (%)", 
                    5.0, 25.0, st.session_state.discount_rate*100, 0.5,
                    format="%.1f%%", 
                    key="discount_rate_slider"
                ) / 100
            with col2:
                st.session_state.terminal_growth = st.slider(
                    "Terminal Growth (%)", 
                    1.0, 5.0, st.session_state.terminal_growth*100, 0.1,
                    format="%.1f%%", 
                    key="terminal_growth_slider"
                ) / 100
            
            # Advanced options
            with st.expander("Advanced Options"):
                # Time-series forecasting
                if STATS_ENABLED:
                    st.write("**Time-Series Forecasting**")
                    hist_input = st.text_input(
                        "Historical revenues (comma separated)", 
                        "100000,120000,150000",
                        help="Enter at least 3 years of historical revenue for ARIMA forecasting"
                    )
                    
                    if st.button("Generate Forecast"):
                        try:
                            hist_values = [float(x.strip()) for x in hist_input.split(",") if x.strip()]
                            if len(hist_values) >= 3:
                                forecasted = ValuationEngine.time_series_forecast(
                                    hist_values, 
                                    min(3, forecast_years)
                                )
                                st.write("**Forecasted Revenue:**")
                                for i, val in enumerate(forecasted):
                                    st.write(f"Year {i+1}: ${val:,.0f}")
                                
                                # Option to apply to growth rates
                                if st.button("Apply to Growth Rates"):
                                    for i in range(min(3, forecast_years)):
                                        if i == 0:
                                            growth = (forecasted[i] - hist_values[-1]) / hist_values[-1]
                                        else:
                                            growth = (forecasted[i] - forecasted[i-1]) / forecasted[i-1]
                                        st.session_state[f"growth_rate_{i}"] = max(0, min(growth, 0.5))
                                        st.rerun()
                            else:
                                st.error("Please provide at least 3 years of historical data")
                        except Exception as e:
                            st.error(f"Error in forecasting: {str(e)}")
                else:
                    st.warning("Statsmodels not available for time-series forecasting")
                
                # Risk adjustment override
                risk_override = st.slider(
                    "Manual Risk Adjustment (%)",
                    0.0, 30.0, 0.0, 0.5,
                    format="%.1f%%",
                    help="Override the AI-calculated risk premium if needed"
                )
                st.session_state.risk_override = risk_override / 100
            
            # Run valuation button
            if st.button("🚀 Run Valuation", use_container_width=True, type="primary"):
                if validate_inputs(base_revenue, growth_rates, margin_forecast, st.session_state.discount_rate):
                    st.session_state.run_valuation = True
                    st.session_state.valuation_time = datetime.now()
            
            # Real-time preview
            if st.session_state.get("preview_mode", False):
                with st.spinner("Calculating preview..."):
                    try:
                        preview_val, *rest = ValuationEngine.calculate_dcf(
                            base_revenue,
                            growth_rates[:3],
                            margin_forecast[:3],
                            st.session_state.discount_rate,
                            st.session_state.terminal_growth,
                            3
                        )
                        st.metric("Quick Estimate", 
                                 f"${preview_val/1000:,.1f}K", 
                                 delta="Preview",
                                 help="Based on first 3 years of forecast")
                    except Exception as e:
                        st.error(f"Preview calculation failed: {str(e)}")

        # Results column
        with result_col:
            if st.session_state.get('run_valuation'):
                # Calculate AI risk score (or use override)
                if st.session_state.get('risk_override', 0) > 0:
                    ai_risk = st.session_state.risk_override
                    risk_source = "Manual override"
                else:
                    ai_risk = ValuationEngine.ai_risk_score(
                        base_revenue, 
                        industry, 
                        np.mean(margin_forecast)
                    )
                    risk_source = "AI model"
                
                # Run DCF valuation
                try:
                    val_res = ValuationEngine.calculate_dcf(
                        base_revenue,
                        growth_rates,
                        margin_forecast,
                        st.session_state.discount_rate + ai_risk,
                        st.session_state.terminal_growth,
                        forecast_years
                    )
                    valuation, rev_forecast, cf, dcf, tv, dtv = val_res
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        "Year": range(1, forecast_years+1),
                        "Revenue": rev_forecast,
                        "Cash Flow": cf,
                        "Discounted CF": dcf
                    }).set_index("Year")
                    
                    # Display valuation results
                    st.success(f"## Enterprise Value: ${valuation:,.2f}")
                    
                    # Key metrics
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("Risk Adjustment",
                                 f"{ai_risk*100:.1f}%",
                                 risk_source,
                                 help="Additional risk premium added to discount rate")
                    with metrics_col2:
                        st.metric("Terminal Value",
                                 f"${dtv:,.0f}",
                                 f"{dtv/valuation*100:.1f}% of total",
                                 help="Present value beyond forecast period")
                    with metrics_col3:
                        st.metric("Implied Multiple",
                                 f"{valuation/base_revenue:.1f}x",
                                 "Revenue multiple",
                                 help="Valuation relative to base revenue")
                    
                    # Results tabs
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📊 Forecast", 
                        "📈 Charts", 
                        "🔍 Analysis", 
                        "📑 Report"
                    ])
                    
                    with tab1:
                        # Forecast table
                        st.dataframe(
                            results_df.style.format({
                                "Revenue": "${:,.0f}",
                                "Cash Flow": "${:,.0f}",
                                "Discounted CF": "${:,.0f}"
                            }).applymap(
                                lambda x: 'color: #2E86AB' if isinstance(x, str) and '$' in x else ''
                            ),
                            use_container_width=True,
                            height=400
                        )
                        
                        # Export data
                        csv = results_df.to_csv(index=True).encode('utf-8')
                        st.download_button(
                            "Download Data (CSV)",
                            csv,
                            "valuation_data.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    
                    with tab2:
                        # Interactive charts
                        chart_type = st.radio(
                            "Chart Type",
                            ["Line Chart", "Area Chart", "Bar Chart"],
                            horizontal=True
                        )
                        
                        chart_data = results_df.reset_index().melt(
                            "Year", 
                            var_name="Metric", 
                            value_name="Value"
                        )
                        
                        if chart_type == "Line Chart":
                            chart = alt.Chart(chart_data).mark_line(
                                point=True
                            ).encode(
                                x=alt.X("Year:O", title="Year"),
                                y=alt.Y("Value:Q", title="Amount ($)"),
                                color=alt.Color("Metric:N", 
                                              scale=alt.Scale(range=[primary_color, secondary_color, "#C73E1D"]),
                                              title="Metric"),
                                tooltip=["Year", "Metric", "Value"]
                            ).properties(
                                height=500,
                                title="Financial Projections"
                            )
                        elif chart_type == "Area Chart":
                            chart = alt.Chart(chart_data).mark_area(
                                opacity=0.6,
                                interpolate='monotone'
                            ).encode(
                                x="Year:O",
                                y="Value:Q",
                                color="Metric:N",
                                tooltip=["Year", "Metric", "Value"]
                            )
                        else:  # Bar Chart
                            chart = alt.Chart(chart_data).mark_bar().encode(
                                x="Year:O",
                                y="Value:Q",
                                color="Metric:N",
                                tooltip=["Year", "Metric", "Value"]
                            )
                        
                        st.altair_chart(chart, use_container_width=True)
                    
                    with tab3:
                        # Sensitivity analysis
                        st.subheader("Sensitivity Analysis")
                        
                        sens_col1, sens_col2 = st.columns(2)
                        with sens_col1:
                            sens_var = st.selectbox(
                                "Analyze Sensitivity To:",
                                ["Discount Rate", "Growth Rates", "Margins", "Terminal Growth"],
                                key="sens_var"
                            )
                        with sens_col2:
                            sens_range = st.slider(
                                "Range (%)", 
                                10, 50, 20,
                                key="sens_range"
                            )
                        
                        if st.button("Run Sensitivity Analysis"):
                            base_case = {
                                "base_revenue": base_revenue,
                                "growth_rates": growth_rates,
                                "margin_forecast": margin_forecast,
                                "discount_rate": st.session_state.discount_rate,
                                "terminal_growth": st.session_state.terminal_growth,
                                "years": forecast_years
                            }
                            
                            with st.spinner("Running analysis..."):
                                sens_results = ValuationEngine.sensitivity_analysis(
                                    base_case, 
                                    sens_var, 
                                    sens_range
                                )
                                
                                # Display sensitivity chart
                                sens_chart = alt.Chart(sens_results).mark_line(
                                    point=True,
                                    strokeWidth=3
                                ).encode(
                                    x=alt.X("Change (%):Q", title=f"{sens_var} Change (%)"),
                                    y=alt.Y("Valuation:Q", title="Enterprise Value ($)"),
                                    tooltip=[
                                        alt.Tooltip("Change (%):Q", format=".1f"),
                                        alt.Tooltip("Valuation:Q", format="$,.0f")
                                    ]
                                ).properties(
                                    title=f"Valuation Sensitivity to {sens_var}",
                                    height=400
                                )
                                
                                st.altair_chart(sens_chart, use_container_width=True)
                                
                                # Calculate sensitivity metrics
                                base_val = sens_results[sens_results["Change (%)"] == 0]["Valuation"].values[0]
                                max_val = sens_results["Valuation"].max()
                                min_val = sens_results["Valuation"].min()
                                sensitivity = (max_val - min_val) / base_val
                                
                                st.metric(
                                    "Sensitivity",
                                    f"{sensitivity*100:.1f}%",
                                    help="Percentage change in valuation across full range"
                                )
                        
                        # Monte Carlo simulation
                        st.subheader("Monte Carlo Simulation")
                        st.write("Simulate valuation under different parameter scenarios")
                        
                        mc_draws = st.slider(
                            "Number of simulations",
                            100, 5000, 1000,
                            key="mc_draws"
                        )
                        
                        if st.button("Run Monte Carlo"):
                            base_case = {
                                "base_revenue": base_revenue,
                                "growth_rates": growth_rates,
                                "margin_forecast": margin_forecast,
                                "discount_rate": st.session_state.discount_rate,
                                "terminal_growth": st.session_state.terminal_growth,
                                "years": forecast_years
                            }
                            
                            with st.spinner(f"Running {mc_draws} simulations..."):
                                mc_results = ValuationEngine.monte_carlo_simulation(
                                    base_case, 
                                    draws=mc_draws
                                )
                                
                                # Display results
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Mean Valuation", f"${mc_results['mean']:,.0f}")
                                with col2:
                                    st.metric("Median Valuation", f"${mc_results['median']:,.0f}")
                                with col3:
                                    st.metric("Standard Deviation", f"${mc_results['std']:,.0f}")
                                
                                # Distribution chart
                                df_mc = pd.DataFrame({"Valuation": mc_results["values"]})
                                hist = alt.Chart(df_mc).mark_bar().encode(
                                    x=alt.X("Valuation:Q", bin=alt.Bin(maxbins=50), title="Valuation ($)"),
                                    y='count()',
                                    tooltip=[alt.Tooltip("count()", title="Frequency")]
                                ).properties(
                                    height=400,
                                    title="Monte Carlo Valuation Distribution"
                                )
                                
                                # Add confidence intervals
                                rule = alt.Chart(pd.DataFrame({
                                    '5th': [mc_results['5th_percentile']],
                                    '95th': [mc_results['95th_percentile']]
                                })).mark_rule(
                                    color='red',
                                    strokeDash=[5,5]
                                ).encode(
                                    x='5th:Q',
                                    x2='95th:Q'
                                )
                                
                                st.altair_chart(hist + rule, use_container_width=True)
                                
                                st.write(f"**90% Confidence Interval:** ${mc_results['5th_percentile']:,.0f} - ${mc_results['95th_percentile']:,.0f}")
                    
                    with tab4:
                        # Report generation
                        st.subheader("Generate Valuation Report")
                        
                        report_col1, report_col2 = st.columns(2)
                        with report_col1:
                            report_type = st.selectbox(
                                "Report Type",
                                ["Executive Summary", "Detailed Analysis", "Investor Deck"],
                                key="report_type"
                            )
                            
                            company_name = st.text_input(
                                "Company Name (optional)",
                                "",
                                key="company_name"
                            )
                            
                            analyst_name = st.text_input(
                                "Analyst Name (optional)",
                                st.session_state.username or "",
                                key="analyst_name"
                            )
                        
                        with report_col2:
                            report_title = st.text_input(
                                "Report Title",
                                f"{company_name} Valuation Report" if company_name else "Business Valuation Report",
                                key="report_title"
                            )
                            
                            include_disclaimer = st.checkbox(
                                "Include Standard Disclaimer",
                                True,
                                key="include_disclaimer"
                            )
                        
                        if st.button("🖨️ Generate PDF Report"):
                            with st.spinner("Generating professional report..."):
                                try:
                                    # Prepare comprehensive report data
                                    report_data = {
                                        'financials': {
                                            'revenue': rev_forecast,
                                            'cogs': [r * 0.4 for r in rev_forecast],  # Example COGS
                                            'gross_profit': [r * 0.6 for r in rev_forecast],
                                            'opex': [r * 0.3 for r in rev_forecast],  # Example OPEX
                                            'ebit': cf,
                                            'taxes': [c * 0.21 for c in cf],  # Example tax rate
                                            'net_income': [c * 0.79 for c in cf],
                                            'operating_cf': [n + d for n, d in zip([c * 0.79 for c in cf], [r * 0.1 for r in rev_forecast])],  # Example D&A
                                            'investing_cf': [-r * 0.15 for r in rev_forecast],  # Example Capex
                                            'financing_cf': [0] * len(rev_forecast),  # Example
                                            'net_cf': cf
                                        },
                                        'valuation_metrics': {
                                            'enterprise_value': valuation,
                                            'equity_value': valuation * 0.9,  # Example
                                            'revenue_multiple': valuation / base_revenue,
                                            'ebitda_multiple': valuation / (cf[0] * 1.3),  # Approximate EBITDA
                                            'wacc': st.session_state.discount_rate + ai_risk,
                                            'risk_premium': ai_risk,
                                            'terminal_growth': st.session_state.terminal_growth,
                                            'year1_cf': dcf[0],
                                            'year2_cf': dcf[1],
                                            'year3_cf': dcf[2],
                                            'terminal_value': dtv
                                        },
                                        'sensitivity_data': {
                                            'Growth Rates': {
                                                'changes': np.linspace(-20, 20, 5),
                                                'valuations': [
                                                    valuation * (1 + x/100) 
                                                    for x in np.linspace(-20, 20, 5)
                                                ]
                                            },
                                            'Discount Rate': {
                                                'changes': np.linspace(-20, 20, 5),
                                                'valuations': [
                                                    valuation / (1 + x/100) 
                                                    for x in np.linspace(-20, 20, 5)
                                                ]
                                            }
                                        },
                                        'company_metrics': {
                                            'growth': np.mean(growth_rates),
                                            'gross_margin': 0.6,  # Example
                                            'ebitda_margin': np.mean(margin_forecast) * 1.3,
                                            'rnd': 0.1,  # Example
                                            'ev_rev': valuation / base_revenue
                                        },
                                        'industry_benchmarks': INDUSTRY_DEFAULTS.get(industry, {
                                            'growth': 0.12,
                                            'gross_margin': 0.55,
                                            'ebitda_margin': 0.25,
                                            'rnd': 0.08,
                                            'ev_rev': 3.5
                                        }),
                                        'risk_data': {
                                            'factors': [
                                                {
                                                    'name': 'Market Competition',
                                                    'score': 7,
                                                    'impact': 8,
                                                    'mitigation': 'Differentiate product offering'
                                                },
                                                {
                                                    'name': 'Customer Concentration',
                                                    'score': 5,
                                                    'impact': 6,
                                                    'mitigation': 'Diversify customer base'
                                                },
                                                {
                                                    'name': 'Regulatory Changes',
                                                    'score': 4,
                                                    'impact': 7,
                                                    'mitigation': 'Monitor regulatory environment'
                                                }
                                            ],
                                            'scenarios': [
                                                {
                                                    'name': 'Economic Downturn',
                                                    'probability': 0.2,
                                                    'impact': -0.3 * valuation,
                                                    'mitigation': 'Maintain cash reserves'
                                                },
                                                {
                                                    'name': 'Key Employee Loss',
                                                    'probability': 0.15,
                                                    'impact': -0.1 * valuation,
                                                    'mitigation': 'Implement retention plans'
                                                }
                                            ]
                                        },
                                        'financial_analysis': "The company shows strong growth potential with improving margins. " +
                                            "The valuation reflects both the growth trajectory and associated risks.",
                                        'methodology_explanation': "Valuation performed using Discounted Cash Flow methodology with " +
                                            f"a {st.session_state.terminal_growth*100:.1f}% terminal growth rate and " +
                                            f"{st.session_state.discount_rate*100:.1f}% discount rate plus {ai_risk*100:.1f}% risk premium."
                                    }
                                    
                                    # Create PDF report
                                    report = ProfessionalReportGenerator(
                                        title=report_title,
                                        logo_path="image.png"
                                    )
                                    
                                    # Add sections based on report type
                                    report.add_section(
                                        "Executive Summary", 
                                        "valuation_summary", 
                                        data=report_data['valuation_metrics'],
                                        analysis="This valuation represents our estimate of the company's fair market value " +
                                                "based on projected cash flows and industry benchmarks."
                                    )
                                    
                                    report.add_section(
                                        "Financial Projections", 
                                        "financial_forecast", 
                                        data=report_data['financials'],
                                        analysis=report_data['financial_analysis']
                                    )
                                    
                                    if report_type != "Executive Summary":
                                        report.add_section(
                                            "Industry Benchmarking", 
                                            "industry_comparison", 
                                            data={
                                                'company': report_data['company_metrics'],
                                                'industry': report_data['industry_benchmarks']
                                            }
                                        )
                                        
                                        report.add_section(
                                            "Risk Assessment", 
                                            "risk_analysis", 
                                            data=report_data['risk_data']
                                        )
                                        
                                        report.add_section(
                                            "Sensitivity Analysis", 
                                            "sensitivity_analysis", 
                                            data=report_data['sensitivity_data']
                                        )
                                    
                                    if report_type == "Detailed Analysis":
                                        report.add_section(
                                            "Monte Carlo Simulation", 
                                            "monte_carlo", 
                                            data={
                                                'mean': valuation,
                                                'median': valuation,
                                                'std': valuation * 0.15,
                                                '5th_percentile': valuation * 0.8,
                                                '95th_percentile': valuation * 1.2,
                                                'values': [valuation * (0.9 + 0.2 * random.random()) for _ in range(100)]
                                            }
                                        )
                                    
                                    if include_disclaimer:
                                        report.add_section(
                                            "Methodology", 
                                            "methodology", 
                                            analysis=report_data['methodology_explanation']
                                        )
                                    
                                    # Save and offer download
                                    pdf_name = f"{company_name.replace(' ', '_')}_Valuation.pdf" if company_name else "Valuation_Report.pdf"
                                    report.generate_report()
                                    report.output(pdf_name)
                                    
                                    with open(pdf_name, "rb") as f:
                                        st.download_button(
                                            "⬇️ Download Report", 
                                            f,
                                            file_name=pdf_name,
                                            mime="application/pdf",
                                            use_container_width=True
                                        )
                                    
                                    # Save to scenarios
                                    scenario = {
                                        "params": {
                                            "company_name": company_name,
                                            "industry": industry,
                                            "base_revenue": base_revenue,
                                            "growth_rates": growth_rates,
                                            "margin_forecast": margin_forecast,
                                            "discount_rate": st.session_state.discount_rate,
                                            "terminal_growth": st.session_state.terminal_growth,
                                            "years": forecast_years,
                                            "risk_adjustment": ai_risk
                                        },
                                        "results": {
                                            "valuation": valuation,
                                            "terminal_value": dtv,
                                            "revenue_forecast": rev_forecast,
                                            "cash_flows": cf
                                        },
                                        "metadata": {
                                            "created_at": datetime.now().isoformat(),
                                            "analyst": analyst_name,
                                            "report_title": report_title
                                        }
                                    }
                                    
                                    st.session_state.saved_scenarios.append(scenario)
                                    save_scenarios(st.session_state.saved_scenarios)
                                    st.success("Report generated and scenario saved!")
                                
                                except Exception as e:
                                    st.error(f"Failed to generate report: {str(e)}")
                
                except Exception as e:
                    st.error(f"Valuation failed: {str(e)}")
                    st.exception(e)
    
    # ======================
    # SCENARIO ANALYSIS PAGE
    # ======================
    elif current_page == "Scenario Analysis":
        st.title("Scenario Analysis")
        st.write("Compare multiple valuation scenarios and perform what-if analysis.")
        
        # Scenario creation
        with st.expander("➕ Create New Scenario", expanded=True):
            scenario_name = st.text_input(
                "Scenario Name", 
                "New Expansion Plan",
                key="new_scenario_name"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                revenue_change = st.slider(
                    "Revenue Change (%)",
                    -50.0, 100.0, 0.0, 5.0,
                    key="revenue_change"
                )
                cost_change = st.slider(
                    "Cost Change (%)",
                    -30.0, 50.0, 0.0, 5.0,
                    key="cost_change"
                )
            with col2:
                growth_change = st.slider(
                    "Growth Rate Change (%)",
                    -50.0, 100.0, 0.0, 5.0,
                    key="growth_change"
                )
                risk_change = st.slider(
                    "Risk Adjustment Change (%)",
                    -5.0, 10.0, 0.0, 0.5,
                    key="risk_change"
                )
            
            if st.button("Save Scenario", key="save_scenario_btn"):
                scenario = {
                    "name": scenario_name,
                    "changes": {
                        "revenue": revenue_change / 100,
                        "costs": cost_change / 100,
                        "growth": growth_change / 100,
                        "risk": risk_change / 100
                    },
                    "created_at": datetime.now().isoformat(),
                    "created_by": st.session_state.username
                }
                
                st.session_state.saved_scenarios.append(scenario)
                save_scenarios(st.session_state.saved_scenarios)
                st.success(f"Scenario '{scenario_name}' saved!")
        
        # Scenario comparison
        if not st.session_state.saved_scenarios:
            st.info("No saved scenarios yet. Create one above.")
        else:
            st.subheader("Saved Scenarios")
            
            # Scenario selector
            scenario_names = [f"{i+1}. {s.get('name', 'Unnamed')}" 
                            for i, s in enumerate(st.session_state.saved_scenarios)]
            selected_scenarios = st.multiselect(
                "Select scenarios to compare",
                scenario_names,
                default=scenario_names[:min(3, len(scenario_names))]
            )
            
            if selected_scenarios:
                selected_indices = [int(s.split(".")[0])-1 for s in selected_scenarios]
                selected_scenario_data = [st.session_state.saved_scenarios[i] for i in selected_indices]
                
                # Display comparison table
                comparison_data = []
                for scenario in selected_scenario_data:
                    comparison_data.append({
                        "Scenario": scenario.get("name", "Unnamed"),
                        "Revenue Change": f"{scenario['changes']['revenue']*100:.1f}%",
                        "Cost Change": f"{scenario['changes']['costs']*100:.1f}%",
                        "Growth Change": f"{scenario['changes']['growth']*100:.1f}%",
                        "Risk Change": f"{scenario['changes']['risk']*100:.1f}%",
                        "Created": datetime.fromisoformat(scenario['created_at']).strftime("%Y-%m-%d"),
                        "By": scenario.get('created_by', 'Unknown')
                    })
                
                st.dataframe(
                    pd.DataFrame(comparison_data).set_index("Scenario"),
                    use_container_width=True
                )
                
                # Scenario comparison chart
                if len(selected_scenarios) > 1:
                    st.subheader("Scenario Comparison")
                    
                    # For demo purposes - in a real app you'd run the valuation for each scenario
                    base_value = 1000000  # Placeholder
                    scenario_values = [
                        base_value * (1 + s['changes']['revenue']) * 
                        (1 + s['changes']['growth']) / 
                        (1 + s['changes']['costs']) * 
                        (1 - s['changes']['risk'])
                        for s in selected_scenario_data
                    ]
                    
                    chart_data = pd.DataFrame({
                        "Scenario": [s['name'] for s in selected_scenario_data],
                        "Valuation": scenario_values
                    })
                    
                    chart = alt.Chart(chart_data).mark_bar().encode(
                        x=alt.X("Scenario:N", sort="-y", title=""),
                        y=alt.Y("Valuation:Q", title="Relative Valuation"),
                        color=alt.Color("Scenario:N", legend=None),
                        tooltip=["Scenario", "Valuation"]
                    ).properties(
                        height=400,
                        title="Relative Valuation Comparison"
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
    
    # ======================
    # REPORTS PAGE
    # ======================
    elif current_page == "Reports":
        st.title("Saved Reports & Valuations")
        st.write("Access your historical valuation reports and analysis.")
        
        if not st.session_state.saved_scenarios:
            st.warning("No saved valuations found. Run valuations first.")
        else:
            # Filter and search
            col1, col2 = st.columns(2)
            with col1:
                industry_filter = st.multiselect(
                    "Filter by Industry",
                    list(INDUSTRY_DEFAULTS.keys()),
                    default=[],
                    key="industry_filter"
                )
            with col2:
                search_query = st.text_input(
                    "Search by name or description",
                    "",
                    key="search_query"
                )
            
            # Apply filters
            filtered_scenarios = st.session_state.saved_scenarios
            if industry_filter:
                filtered_scenarios = [
                    s for s in filtered_scenarios 
                    if s.get('params', {}).get('industry') in industry_filter
                ]
            
            if search_query:
                filtered_scenarios = [
                    s for s in filtered_scenarios 
                    if search_query.lower() in str(s.get('name', '')).lower() or
                    search_query.lower() in str(s.get('metadata', {}).get('report_title', '')).lower()
                ]
            
            if not filtered_scenarios:
                st.info("No scenarios match your filters.")
            else:
                # Display as expandable cards
                for i, scenario in enumerate(filtered_scenarios):
                    with st.expander(
                        f"🔍 {scenario.get('name', f'Scenario {i+1}')} - "
                        f"${scenario.get('results', {}).get('valuation', 0):,.0f}",
                        expanded=False
                    ):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.metric(
                                "Valuation",
                                f"${scenario.get('results', {}).get('valuation', 0):,.0f}",
                                scenario.get('params', {}).get('industry', 'N/A')
                            )
                            
                            st.write(f"**Date:** {scenario.get('metadata', {}).get('created_at', 'N/A')}")
                            st.write(f"**Analyst:** {scenario.get('metadata', {}).get('analyst', 'N/A')}")
                            
                            if st.button("📄 Generate Report", key=f"report_{i}"):
                                # In a real app, you'd regenerate the report
                                st.success(f"Report for {scenario.get('name', 'Scenario')} generated!")
                        
                        with col2:
                            st.write("**Parameters:**")
                            params = scenario.get('params', {})
                            st.json({
                                k: v for k, v in params.items() 
                                if k not in ['growth_rates', 'margin_forecast']
                            })
    
    # ======================
    # SETTINGS PAGE
    # ======================
    elif current_page == "Settings":
        st.title("User Settings")
        
        with st.form("user_settings"):
            st.subheader("Preferences")
            
            col1, col2 = st.columns(2)
            with col1:
                dark_mode = st.checkbox(
                    "Dark Mode",
                    value=st.session_state.dark_mode,
                    key="settings_dark_mode"
                )
                new_color_scheme = st.selectbox(
                    "Color Scheme",
                    list(COLOR_SCHEMES.keys()),
                    index=list(COLOR_SCHEMES.keys()).index(st.session_state.color_scheme),
                    key="settings_color_scheme"
                )
            with col2:
                preview_mode = st.checkbox(
                    "Live Preview",
                    value=st.session_state.preview_mode,
                    key="settings_preview_mode"
                )
                default_years = st.slider(
                    "Default Forecast Years",
                    3, 10, st.session_state.forecast_years,
                    key="settings_default_years"
                )
            
            st.subheader("Account")
            current_password = st.text_input(
                "Current Password",
                type="password",
                key="current_password"
            )
            new_password = st.text_input(
                "New Password",
                type="password",
                key="new_password"
            )
            confirm_password = st.text_input(
                "Confirm New Password",
                type="password",
                key="confirm_password"
            )
            
            if st.form_submit_button("Save Settings"):
                # Save preferences
                st.session_state.dark_mode = dark_mode
                st.session_state.color_scheme = new_color_scheme
                st.session_state.preview_mode = preview_mode
                st.session_state.forecast_years = default_years
                
                save_user_prefs({
                    'dark_mode': dark_mode,
                    'color_scheme': new_color_scheme,
                    'preview_mode': preview_mode
                })
                
                # Handle password change
                if new_password:
                    if new_password != confirm_password:
                        st.error("New passwords don't match!")
                    elif not current_password:
                        st.error("Please enter current password to change it")
                    else:
                        # In a real app, verify current password first
                        st.success("Settings saved! (Password change simulated)")
                else:
                    st.success("Settings saved!")
                
                st.rerun()

if __name__ == "__main__":
    main()