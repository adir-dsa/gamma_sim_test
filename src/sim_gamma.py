# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import gamma, bernoulli
import plotly.graph_objects as go
import plotly.express as px
import time # To show progress

# --- Constants ---
N_SIMULATIONS = 10000 
CI_LOWER_BOUND = 2.5
CI_UPPER_BOUND = 97.5

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Revenue Impact Simulator (Monte Carlo)", page_icon="🎲")

# --- Custom CSS for Styling ---
st.markdown("""
<style>
/* General Card Style */
.metric-card {
    background-color: #FFFFFF; /* Card background */
    border: 1px solid #E0E0E0; /* Lighter Card border */
    padding: 20px; /* Increased padding */
    border-radius: 8px; /* Slightly less rounded */
    box-shadow: 0 2px 4px rgba(0,0,0,0.05); /* Softer shadow */
    margin-bottom: 20px; /* More space between cards */
    text-align: center; /* Center align text */
    height: 160px; /* Fixed height for alignment */
    display: flex;
    flex-direction: column;
    justify-content: center; /* Vertically center content */
    transition: box-shadow 0.2s ease-in-out, border-color 0.2s ease-in-out;
}
.metric-card:hover {
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    border-color: #C0C0C0;
}

/* Card Label/Title */
.metric-label {
    font-size: 0.95em; /* Slightly smaller label */
    color: #555555; /* Darker grey for label */
    font-weight: 600; /* Semi-bold */
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-transform: uppercase; /* Uppercase label */
    letter-spacing: 0.5px;
}
.metric-label i { /* Style for Font Awesome icons */
    margin-right: 8px;
    font-size: 1.1em; /* Slightly larger icon */
    color: #007bff; /* Default Icon color */
}
/* Specific Icon Colors (Optional examples) */
.fa-sack-dollar { color: #28a745; } /* Green for revenue */
.fa-user-dollar { color: #17a2b8; } /* Teal for ARPU */
.fa-user-slash { color: #ffc107; } /* Amber for zero % */
.fa-chart-line { color: #6f42c1; } /* Purple for active rev */


/* Main Value (Mean) */
.metric-value {
    font-size: 2.0em; /* Slightly smaller main value */
    font-weight: 700; /* Bolder */
    color: #212529; /* Bootstrap dark color */
    line-height: 1.1;
    margin-bottom: 8px; /* Space below value */
}

/* Confidence Interval Text */
.metric-ci {
    font-size: 0.8em;
    color: #6c757d; /* Bootstrap secondary grey */
    margin-top: auto; /* Push CI to bottom if needed */
}

/* Impact Card Specific Styles */
.impact-value-positive {
    color: #28a745 !important; /* Green for positive impact - use !important to override .metric-value color */
}
.impact-value-negative {
    color: #dc3545 !important; /* Red for negative impact */
}
.impact-value-neutral {
    color: #212529 !important; /* Default color if neutral */
}

/* Column Headers */
.column-header {
    font-size: 1.3em; /* Adjusted size */
    font-weight: 600;
    margin-bottom: 20px;
    color: #343a40; /* Bootstrap dark grey */
    text-align: center;
    border-bottom: 2px solid #dee2e6; /* Bootstrap light border */
    padding-bottom: 8px;
}

/* Make Streamlit columns have a bit more gap */
div[data-testid="stHorizontalBlock"] > div[data-testid^="stVerticalBlock"] {
    padding: 0 12px; /* Add slightly more horizontal padding */
}

</style>
""", unsafe_allow_html=True)

# Add Font Awesome link in header (requires internet)
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css">', unsafe_allow_html=True)


# --- Helper Functions ---

def generate_zig_data(n_customers, p_zero, k_shape, theta_scale):
    """Generates data from a Zero-Inflated Gamma distribution."""
    # Basic parameter validation
    if not (0 <= p_zero <= 1): return np.zeros(n_customers) # Error case: invalid p_zero
    if k_shape <= 0 or theta_scale <= 0:
        return np.zeros(n_customers)

    is_gamma = bernoulli.rvs(1 - p_zero, size=n_customers)
    n_gamma = np.sum(is_gamma)
    data = np.zeros(n_customers)
    if n_gamma > 0:
        gamma_samples = gamma.rvs(a=k_shape, scale=theta_scale, size=n_gamma)
        # Ensure samples are non-negative (can happen with tiny k/theta)
        gamma_samples[gamma_samples < 0] = 0
        data[is_gamma == 1] = gamma_samples
    return data

def format_currency(value):
    """Formats a number as currency."""
    if not np.isfinite(value): return "N/A"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"

def format_percentage(value):
    """Formats a number as percentage."""
    if not np.isfinite(value): return "N/A"
    return f"{value:,.1f}" # Use comma for thousands separator if needed, 1 decimal place

def run_monte_carlo(n_sims, n_cust, base_p, base_k, base_th, pol_p, pol_k, pol_th):
    """Runs the Monte Carlo simulation N times."""
    baseline_metrics = {
        'total_revenue': [], 'arpu': [], 'zero_perc': [], 'avg_active_revenue': [], 'n_zeros': [], 'n_active': []
    }
    initiative_metrics = {
        'total_revenue': [], 'arpu': [], 'zero_perc': [], 'avg_active_revenue': [], 'n_zeros': [], 'n_active': []
    }
    first_run_data = {'baseline': None, 'initiative': None}

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    start_time = time.time()

    for i in range(n_sims):
        # --- Baseline Simulation ---
        b_data = generate_zig_data(n_cust, base_p, base_k, base_th)
        b_active_data = b_data[b_data > 0]
        b_n_zeros = np.sum(b_data == 0)
        b_n_active = len(b_active_data)

        baseline_metrics['total_revenue'].append(np.sum(b_data))
        baseline_metrics['arpu'].append(np.mean(b_data) if n_cust > 0 else 0)
        baseline_metrics['n_zeros'].append(b_n_zeros)
        baseline_metrics['n_active'].append(b_n_active)
        baseline_metrics['zero_perc'].append(b_n_zeros / n_cust * 100 if n_cust > 0 else 0)
        baseline_metrics['avg_active_revenue'].append(np.mean(b_active_data) if b_n_active > 0 else 0)

        # --- Scenario Simulation ---
        p_data = generate_zig_data(n_cust, pol_p, pol_k, pol_th)
        p_active_data = p_data[p_data > 0]
        p_n_zeros = np.sum(p_data == 0)
        p_n_active = len(p_active_data)

        initiative_metrics['total_revenue'].append(np.sum(p_data))
        initiative_metrics['arpu'].append(np.mean(p_data) if n_cust > 0 else 0)
        initiative_metrics['n_zeros'].append(p_n_zeros)
        initiative_metrics['n_active'].append(p_n_active)
        initiative_metrics['zero_perc'].append(p_n_zeros / n_cust * 100 if n_cust > 0 else 0)
        initiative_metrics['avg_active_revenue'].append(np.mean(p_active_data) if p_n_active > 0 else 0)

        if i == 0:
            first_run_data['baseline'] = b_data
            first_run_data['initiative'] = p_data

        # Update progress less frequently for performance
        if (i + 1) % max(1, n_sims // 100) == 0 or (i + 1) == n_sims:
             progress = (i + 1) / n_sims
             progress_bar.progress(progress)
             status_text.text(f"Running Simulation {i+1}/{n_sims}")

    progress_bar.empty()
    end_time = time.time()
    status_text.text(f"Simulations Complete ({n_sims} runs in {end_time - start_time:.2f} seconds)")

    for key in baseline_metrics: baseline_metrics[key] = np.array(baseline_metrics[key])
    for key in initiative_metrics: initiative_metrics[key] = np.array(initiative_metrics[key])

    return baseline_metrics, initiative_metrics, first_run_data

def calculate_ci_and_mean(data_array):
    """Calculates mean and 95% CI using percentiles."""
    if len(data_array) == 0: # Handle empty array case
        return np.nan, np.nan, np.nan
    mean_val = np.mean(data_array)
    # Ensure percentiles are calculated only on finite values if necessary
    finite_data = data_array[np.isfinite(data_array)]
    if len(finite_data) < 2: # Need at least 2 points for percentile
         return mean_val, np.nan, np.nan
    ci_low = np.percentile(finite_data, CI_LOWER_BOUND)
    ci_high = np.percentile(finite_data, CI_UPPER_BOUND)
    return mean_val, ci_low, ci_high

# --- Card Generating Functions ---

def create_metric_card(icon_class, label, mean_value, ci_low, ci_high, format_func, unit="", bg_color="#FFFFFF", border_color="#E0E0E0"):
    """Generates HTML for a KPI metric card."""
    formatted_mean = format_func(mean_value)
    formatted_ci = f"{format_func(ci_low)} - {format_func(ci_high)}"
    if formatted_mean == "N/A":
        formatted_ci = "N/A"

    card_html = f"""
    <div class="metric-card" style="background-color:{bg_color}; border-color:{border_color};">
        <div class="metric-label">
            <i class="{icon_class}"></i>
            {label}
        </div>
        <div class="metric-value">
            {formatted_mean}{unit}
        </div>
        <div class="metric-ci">
            95% CI: {formatted_ci}
        </div>
    </div>
    """
    return card_html

def create_impact_card(icon_class, label, mean_lift, ci_low, ci_high, format_func, unit="", positive_is_good=True):
    """Generates HTML for an impact metric card with color coding."""
    formatted_mean = format_func(mean_lift)
    formatted_ci = f"{format_func(ci_low)} - {format_func(ci_high)}"

    color_class = "impact-value-neutral"
    if np.isfinite(mean_lift) and abs(mean_lift) > 1e-9 : # Check for non-zero finite values
        is_positive = mean_lift > 0
        is_significantly_positive = ci_low > 1e-9 # Check if lower bound is also positive
        is_significantly_negative = ci_high < -1e-9 # Check if upper bound is also negative

        if positive_is_good:
            if is_significantly_positive: color_class = "impact-value-positive"
            elif is_significantly_negative: color_class = "impact-value-negative"
            # else: neutral (CI includes zero or value is near zero)
        else: # Negative change is good
             if is_significantly_negative: color_class = "impact-value-positive" # Good outcome = green
             elif is_significantly_positive: color_class = "impact-value-negative" # Bad outcome = red
             # else: neutral

    if formatted_mean == "N/A":
        formatted_ci = "N/A"
        color_class = "impact-value-neutral"

    card_html = f"""
    <div class="metric-card">
        <div class="metric-label">
           <i class="{icon_class}"></i>
           {label}
        </div>
        <div class="metric-value {color_class}">
            {formatted_mean}{unit}
        </div>
        <div class="metric-ci">
            95% CI: {formatted_ci}
        </div>
    </div>
    """
    return card_html


# --- Title and Introduction ---
st.title("🎲 Revenue Impact Simulator")
st.markdown(f"""
Analyze the potential revenue impact of customer growth initiatives using **Monte Carlo simulation** ({N_SIMULATIONS:,} runs).
Results include **95% confidence intervals** to quantify outcome uncertainty.
Uses a **Zero-Inflated Gamma** model for customer revenue. Adjust parameters and simulate below.
""")
st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Simulation Controls")

# --- Baseline Parameters ---
st.sidebar.subheader("1. Baseline Customer Profile")
n_customers = st.sidebar.slider("Number of Customers", 1000, 1000000, 10000, 1000, help="Total customers per simulation run.")

st.sidebar.markdown("**Distribution Settings (Current State):**")
current_p_zero_perc = st.sidebar.slider("Percentage of Zero-Revenue Customers (%)", 0.0, 100.0, 60.0, 0.5,
                                   help="Proportion of customers currently generating no revenue (p_zero).")
current_p_zero = current_p_zero_perc / 100.0

current_mean_active = st.sidebar.slider("Average Revenue per *Active* Customer ($)", 1.0, 1000.0, 150.0, 1.0, # Min value 1.0
                                        help="Current average revenue ONLY from active customers (k*θ > 0).")
current_k_shape = st.sidebar.slider("Revenue Consistency (Shape k)", 0.1, 10.0, 2.0, 0.1, # Min value 0.1
                                     help="Consistency of active revenue (higher k = less relative variation, k > 0).")
current_theta_scale = current_mean_active / current_k_shape if current_k_shape > 0 else 0

valid_params = True
if current_theta_scale <= 0 and current_p_zero < 1:
     st.sidebar.error("Baseline Scale (θ) is non-positive. Adjust Avg Active Revenue or Consistency.")
     valid_params = False

st.sidebar.info(f"""
*Implied Baseline Parameters:*
- `p_zero`: {current_p_zero:.3f}, `k`: {current_k_shape:.2f}, `θ`: {current_theta_scale:.2f}
""")

# --- Initiative Selection and Impact ---
st.sidebar.subheader("2. Initiative Scenario")
initiative_options = {
    "None": "No initiative applied.",
    "Engagement Campaign": "Aims to convert inactive customers to active ones.",
    "Premium Service Tier": "Aims to increase spending among currently active customers.",
    "Fee Optimization": "Aims to slightly increase activity *and* average spend.",
    "Loyalty Program": "Aims to increase spending consistency and amount for active users."
}
selected_initiative = st.sidebar.selectbox("Select Initiative to Simulate", options=list(initiative_options.keys()), index=0,
                                       help="Choose an initiative whose potential impact you want to model.")
st.sidebar.markdown(f"**Description:** *{initiative_options[selected_initiative]}*")

# --- Initiative Impact Levers (Conditional) ---
st.sidebar.markdown("**Initiative Impact Levers (Hypothesized Effects):**")
impact_p_reduction = 0.0
impact_mean_increase = 0.0
impact_k_increase_perc = 0.0

if selected_initiative == "Engagement Campaign":
    impact_p_reduction_perc = st.sidebar.slider("Reduction in Zero-Revenue % (Target)", 0.0, 100.0, 10.0, 0.5)
    impact_p_reduction = impact_p_reduction_perc / 100.0
elif selected_initiative == "Premium Service Tier":
    impact_mean_increase_perc = st.sidebar.slider("Increase in Avg. Active Revenue % (Target)", 0.0, 100.0, 15.0, 0.5)
    impact_mean_increase = impact_mean_increase_perc / 100.0
elif selected_initiative == "Fee Optimization":
    impact_p_reduction_perc = st.sidebar.slider("Reduction in Zero-Revenue % (Target)", 0.0, 50.0, 5.0, 0.5)
    impact_p_reduction = impact_p_reduction_perc / 100.0
    impact_mean_increase_perc = st.sidebar.slider("Increase in Avg. Active Revenue % (Target)", 0.0, 50.0, 5.0, 0.5)
    impact_mean_increase = impact_mean_increase_perc / 100.0
elif selected_initiative == "Loyalty Program":
    impact_mean_increase_perc = st.sidebar.slider("Increase in Avg. Active Revenue % (Target)", 0.0, 75.0, 10.0, 0.5)
    impact_mean_increase = impact_mean_increase_perc / 100.0
    impact_k_increase_perc = st.sidebar.slider("Increase in Revenue Consistency (k) % ", 0.0, 50.0, 5.0, 0.5) / 100.0

# Calculate initiative parameters
initiative_p_zero = max(0, min(1, current_p_zero * (1 - impact_p_reduction)))
initiative_k_shape = max(0.01, current_k_shape * (1 + impact_k_increase_perc)) # Ensure k > 0
initiative_mean_active = max(0.01, current_mean_active * (1 + impact_mean_increase)) # Ensure mean > 0
initiative_theta_scale = initiative_mean_active / initiative_k_shape if initiative_k_shape > 0 else 0

if initiative_k_shape <= 0 and initiative_p_zero < 1 and selected_initiative != "None":
     st.sidebar.error("Initiative impact results in non-positive Shape (k). Adjust impacts.")
     valid_params = False
if initiative_theta_scale <= 0 and initiative_p_zero < 1 and selected_initiative != "None":
     st.sidebar.error("Initiative impact results in non-positive Scale (θ). Adjust impacts.")
     valid_params = False

st.sidebar.info(f"""
*Implied Initiative Parameters:*
- `p_zero`: {initiative_p_zero:.3f}, `k`: {initiative_k_shape:.2f}, `θ`: {initiative_theta_scale:.2f}
- *Implied Avg Active Rev*: ${initiative_mean_active:,.2f}
""")

# --- Simulation Execution ---
st.sidebar.subheader("3. Run Simulation")
run_simulation = st.sidebar.button(f"🚀 Run {N_SIMULATIONS} Simulations", disabled=not valid_params, type="primary")

# --- Main Panel Results ---
if 'simulation_results' not in st.session_state:
    st.session_state['simulation_results'] = None
if 'simulation_run_complete' not in st.session_state:
     st.session_state['simulation_run_complete'] = False

if run_simulation and valid_params:
    # Clear previous plots/state if needed before starting a new run
    st.session_state['simulation_run_complete'] = False
    st.session_state['simulation_results'] = None

    spinner_placeholder = st.empty() # Placeholder for spinner message
    with spinner_placeholder:
         with st.spinner(f"Running {N_SIMULATIONS} simulations... This may take a moment."):
              baseline_metrics_mc, initiative_metrics_mc, first_run_data = run_monte_carlo(
                   N_SIMULATIONS, n_customers,
                   current_p_zero, current_k_shape, current_theta_scale,
                   initiative_p_zero, initiative_k_shape, initiative_theta_scale
              )
              st.session_state['simulation_results'] = {
                   'baseline': baseline_metrics_mc,
                   'initiative': initiative_metrics_mc,
                   'first_run': first_run_data,
                   'initiative_name': selected_initiative if selected_initiative != "None" else "Baseline" # Store initiative name
              }
              st.session_state['simulation_run_complete'] = True
              spinner_placeholder.empty() # Clear spinner message on completion


if st.session_state.get('simulation_run_complete', False):
    results = st.session_state['simulation_results']
    if results is None:
         st.warning("Simulation results are not available. Please run the simulation.")
         st.stop() # Stop execution if results somehow became None after flag set

    baseline_mc = results['baseline']
    initiative_mc = results['initiative']
    first_run = results['first_run']
    # Handle case where initiative is "None" more gracefully
    initiative_name = results['initiative_name'] if results['initiative_name'] != 'Baseline' else 'No Initiative'

    st.header(f"📊 Monte Carlo Simulation Results ({N_SIMULATIONS} runs)")
    st.markdown(f"Comparing **Baseline** vs. **Initiative: {initiative_name}**")
    st.markdown("*(Showing Mean & 95% Confidence Interval)*")

    # --- Calculate CIs for all metrics ---
    results_summary = {}
    for scenario, data in [('Baseline', baseline_mc), ('initiative', initiative_mc)]:
        results_summary[scenario] = {}
        for metric, values in data.items():
            mean_val, ci_low, ci_high = calculate_ci_and_mean(values)
            results_summary[scenario][metric] = {'mean': mean_val, 'ci_low': ci_low, 'ci_high': ci_high}

    # --- Calculate CIs for Lift Metrics ---
    revenue_lift_dist = initiative_mc['total_revenue'] - baseline_mc['total_revenue']
    mean_lift, lift_ci_low, lift_ci_high = calculate_ci_and_mean(revenue_lift_dist)

    valid_baseline_rev = baseline_mc['total_revenue'][np.isfinite(baseline_mc['total_revenue']) & (baseline_mc['total_revenue'] != 0)]
    valid_initiative_rev_match = initiative_mc['total_revenue'][np.isfinite(baseline_mc['total_revenue']) & (baseline_mc['total_revenue'] != 0)]

    if len(valid_baseline_rev) > 0:
        # Ensure lengths match if filtering initiative based on baseline non-zero
        indices = np.where(np.isfinite(baseline_mc['total_revenue']) & (baseline_mc['total_revenue'] != 0))[0]
        valid_initiative_rev_match = initiative_mc['total_revenue'][indices]

        revenue_lift_perc_dist = (valid_initiative_rev_match - valid_baseline_rev) / valid_baseline_rev * 100
        # Remove potential inf/-inf if initiative rev is huge and baseline was tiny
        revenue_lift_perc_dist = revenue_lift_perc_dist[np.isfinite(revenue_lift_perc_dist)]
        mean_lift_perc, lift_perc_ci_low, lift_perc_ci_high = calculate_ci_and_mean(revenue_lift_perc_dist)

    else:
        mean_lift_perc, lift_perc_ci_low, lift_perc_ci_high = (np.nan, np.nan, np.nan)

    zero_perc_change_dist = initiative_mc['zero_perc'] - baseline_mc['zero_perc']
    mean_zero_change, zero_change_ci_low, zero_change_ci_high = calculate_ci_and_mean(zero_perc_change_dist)

    avg_active_change_dist = initiative_mc['avg_active_revenue'] - baseline_mc['avg_active_revenue']
    mean_avg_active_change, avg_active_change_ci_low, avg_active_change_ci_high = calculate_ci_and_mean(avg_active_change_dist)

    # --- Display KPIs with CIs using Custom Cards ---
    st.subheader("Key Performance Indicators")
    col_kpi1, col_kpi2 = st.columns(2)

    with col_kpi1:
        st.markdown("<div class='column-header'>Baseline Scenario</div>", unsafe_allow_html=True)
        bs = results_summary['Baseline']
        html = create_metric_card("fa-solid fa-sack-dollar", "TOTAL REVENUE", bs['total_revenue']['mean'], bs['total_revenue']['ci_low'], bs['total_revenue']['ci_high'], format_currency, bg_color="#F8F9FA") # Lighter grey bg
        st.markdown(html, unsafe_allow_html=True)
        html = create_metric_card("fa-solid fa-user-dollar", "ARPU", bs['arpu']['mean'], bs['arpu']['ci_low'], bs['arpu']['ci_high'], format_currency, bg_color="#F8F9FA")
        st.markdown(html, unsafe_allow_html=True)
        html = create_metric_card("fa-solid fa-user-slash", "% ZERO REVENUE", bs['zero_perc']['mean'], bs['zero_perc']['ci_low'], bs['zero_perc']['ci_high'], format_percentage, unit="%", bg_color="#F8F9FA")
        st.markdown(html, unsafe_allow_html=True)
        html = create_metric_card("fa-solid fa-chart-line", "AVG. ACTIVE REV", bs['avg_active_revenue']['mean'], bs['avg_active_revenue']['ci_low'], bs['avg_active_revenue']['ci_high'], format_currency, bg_color="#F8F9FA")
        st.markdown(html, unsafe_allow_html=True)

    with col_kpi2:
        initiative_title = initiative_name if initiative_name != 'No Inititative' else 'Scenario Result'
        st.markdown(f"<div class='column-header'>Initiative: {initiative_title}</div>", unsafe_allow_html=True)
        ps = results_summary['initiative']
        html = create_metric_card("fa-solid fa-sack-dollar", "TOTAL REVENUE", ps['total_revenue']['mean'], ps['total_revenue']['ci_low'], ps['total_revenue']['ci_high'], format_currency, bg_color="#FEFBF4") # Light yellow/orange bg
        st.markdown(html, unsafe_allow_html=True)
        html = create_metric_card("fa-solid fa-user-dollar", "ARPU", ps['arpu']['mean'], ps['arpu']['ci_low'], ps['arpu']['ci_high'], format_currency, bg_color="#FEFBF4")
        st.markdown(html, unsafe_allow_html=True)
        html = create_metric_card("fa-solid fa-user-slash", "% ZERO REVENUE", ps['zero_perc']['mean'], ps['zero_perc']['ci_low'], ps['zero_perc']['ci_high'], format_percentage, unit="%", bg_color="#FEFBF4")
        st.markdown(html, unsafe_allow_html=True)
        html = create_metric_card("fa-solid fa-chart-line", "AVG. ACTIVE REV", ps['avg_active_revenue']['mean'], ps['avg_active_revenue']['ci_low'], ps['avg_active_revenue']['ci_high'], format_currency, bg_color="#FEFBF4")
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Estimated Initiative Impact")
    col_imp1, col_imp2, col_imp3, col_imp4 = st.columns(4)

    with col_imp1:
        html = create_impact_card("fa-solid fa-arrow-up-right-dots", "Revenue Lift ($)", mean_lift, lift_ci_low, lift_ci_high, format_currency, positive_is_good=True)
        st.markdown(html, unsafe_allow_html=True)
    with col_imp2:
        html = create_impact_card("fa-solid fa-percent", "Revenue Lift (%)", mean_lift_perc, lift_perc_ci_low, lift_perc_ci_high, format_percentage, unit="%", positive_is_good=True)
        st.markdown(html, unsafe_allow_html=True)
    with col_imp3:
        html = create_impact_card("fa-solid fa-person-falling-burst", "Change in % Zeros", mean_zero_change, zero_change_ci_low, zero_change_ci_high, format_percentage, unit=" pts", positive_is_good=False) # Negative change is good
        st.markdown(html, unsafe_allow_html=True)
    with col_imp4:
        html = create_impact_card("fa-solid fa-dollar-sign", "Change in Avg Active Rev", mean_avg_active_change, avg_active_change_ci_low, avg_active_change_ci_high, format_currency, positive_is_good=True)
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Visualizations")
    st.markdown("<p style='font-size:0.9em; color: #6c757d;'><i>Note: Plots below show results from one representative simulation run to illustrate distribution shapes. Aggregate uncertainty is captured in the KPI confidence intervals above.</i></p>", unsafe_allow_html=True)

    # --- Bar Chart: Use AVERAGE counts from simulations ---
    st.markdown("##### Average Breakdown: Zero vs. Active Customers")
    avg_baseline_zeros = results_summary['Baseline']['n_zeros']['mean']
    avg_baseline_active = results_summary['Baseline']['n_active']['mean']
    avg_initiative_zeros = results_summary['initiative']['n_zeros']['mean']
    avg_initiative_active = results_summary['initiative']['n_active']['mean']

    plot_data_avg = {
        'Scenario': ['Baseline', 'Baseline', f'Initiative: {initiative_name}', f'Initiative: {initiative_name}'],
        'Customer Type': ['Zero Revenue', 'Active Revenue', 'Zero Revenue', 'Active Revenue'],
        'Average Number of Customers': [avg_baseline_zeros, avg_baseline_active, avg_initiative_zeros, avg_initiative_active]
    }
    df_counts_avg = pd.DataFrame(plot_data_avg)

    fig_bar_avg = px.bar(df_counts_avg,
                         x='Scenario',
                         y='Average Number of Customers',
                         color='Customer Type',
                         barmode='group',
                         title='Average Customer Counts: Zero vs. Active',
                         labels={'Average Number of Customers': 'Avg. Count'},
                         color_discrete_map={'Zero Revenue':'#EF553B', 'Active Revenue':'#636EFA'},
                         height=350) # Slightly smaller height
    fig_bar_avg.update_layout(yaxis_title='Average Number of Customers', title_x=0.5)
    st.plotly_chart(fig_bar_avg, use_container_width=True)

    # --- Density Plot: Use data from the FIRST simulation run ---
    st.markdown("##### Example Distribution: Active Customer Revenue")
    baseline_data_first = first_run['baseline']
    initiative_data_first = first_run['initiative']

    if baseline_data_first is None or initiative_data_first is None:
        st.warning("Sample run data for plotting is not available.")
    else:
        non_zero_baseline_first = baseline_data_first[baseline_data_first > 0]
        non_zero_initiative_first = initiative_data_first[initiative_data_first > 0]

        max_rev = 1
        percentile_cap = 99
        baseline_perc = np.percentile(non_zero_baseline_first, percentile_cap) if len(non_zero_baseline_first)>0 else 0
        initiative_perc = np.percentile(non_zero_initiative_first, percentile_cap) if len(non_zero_initiative_first)>0 else 0
        max_rev = max(baseline_perc, initiative_perc)

        # Use mean active revenue if percentile is too small or zero
        mean_active_base = results_summary['Baseline']['avg_active_revenue']['mean']
        if max_rev <= 1 : max_rev = mean_active_base * 3 if np.isfinite(mean_active_base) and mean_active_base > 0 else 10

        fig_density_first = go.Figure()
        plot_ok = False # Flag to check if any data was plotted

        if len(non_zero_baseline_first) > 0:
            fig_density_first.add_trace(go.Histogram(x=non_zero_baseline_first, name='Baseline (Active) - Sample Run',
                                       marker_color='#1f77b4', nbinsx=50, histnorm='probability density', opacity=0.7))
            plot_ok = True

        if len(non_zero_initiative_first) > 0 and initiative_name != 'No Initiative': # Only plot initiative if it exists
            fig_density_first.add_trace(go.Histogram(x=non_zero_initiative_first, name=f'Initiative: {initiative_name} (Active) - Sample Run',
                                    marker_color='#ff7f0e', nbinsx=50, histnorm='probability density', opacity=0.7))
            plot_ok = True

        if plot_ok:
            fig_density_first.update_layout(
                title='Example Density Distribution for ACTIVE Customers (Single Run)',
                xaxis_title='Revenue per Customer ($)',
                yaxis_title='Density',
                barmode='overlay',
                legend_title_text='Scenario (Sample)',
                xaxis_range=[0, max_rev * 1.05],
                height=400,
                title_x=0.5
            )
            fig_density_first.update_traces(opacity=0.75)
            st.plotly_chart(fig_density_first, use_container_width=True)
            st.caption(f"*Note: Illustrative plot from one simulation run. X-axis capped near {percentile_cap}th percentile for clarity.*")
        else:
             st.info("No active customers in the sample simulation run to display density distribution.")

    # --- Explanation Expander ---
    with st.expander("ℹ️ Detailed Explanations & Caveats"):
        st.markdown(f"""
        #### Understanding the Monte Carlo Simulation ({N_SIMULATIONS:,} Runs)

        1.  **Multiple Scenarios:** We simulated `{N_SIMULATIONS:,}` possible futures based on the input parameters. Each run generated `{n_customers:,}` customers for both Baseline and Initiative scenarios.
        2.  **Quantifying Uncertainty:** This approach captures inherent randomness. The **95% Confidence Interval (CI)** shows the range where we expect the true average metric to lie 95% of the time, given the model and assumptions.
        3.  **Interpreting Cards:**
            *   **KPI Cards:** Show the average (mean) result across all runs and the 95% CI range.
            *   **Impact Cards:** Show the average difference between Initiative and Baseline. Color indicates the direction of impact (Green = Favorable, Red = Unfavorable, based on context). Check if the CI includes zero – if it does, the observed average impact might not be statistically significant.
        4.  **Visualizations:** The **Bar Chart** shows average counts. The **Density Plot** uses *one sample run* to illustrate the *type* of distributional shift (useful for understanding *how* things change, not the aggregate uncertainty).

        #### Key Insights from this Scenario:

        *   The initiative **`{initiative_name}`** resulted in a mean projected revenue lift of **`{format_currency(mean_lift)}`** (95% CI: `{format_currency(lift_ci_low)}` to `{format_currency(lift_ci_high)}`).
        *   Assess Significance: Does the **Revenue Lift CI** exclude zero? If yes, the impact is likely real (under these assumptions). If it includes zero, the effect is uncertain.
        *   Identify Drivers: Look at the CIs for **Change in % Zeros** and **Change in Avg Active Rev**. Which impact seems more certain (CI excludes zero) or larger in magnitude?

        #### Important Caveats:

        *   **Model Accuracy:** Results depend on the Zero-Inflated Gamma model accurately reflecting customer behavior and the initiative impact assumptions being realistic.
        *   **Parameter Certainty:** Assumes input parameters are known; in reality, they carry their own uncertainty. Sensitivity analysis on inputs could be a next step.
        *   **Simulation, Not Forecast:** This tool aids strategic decision-making and scenario comparison, not precise financial prediction. External factors are omitted.
        """)


elif not valid_params:
     st.error("Simulation cannot run due to invalid baseline or initiative parameters (e.g., negative Scale/Shape). Please check sidebar inputs.")
else:
    st.info(f"Adjust parameters and click '🚀 Run {N_SIMULATIONS} Simulations' to start.")

# Add a footer or separator
st.markdown("---")
st.caption("Monte Carlo Initiative Simulator v1.1")
