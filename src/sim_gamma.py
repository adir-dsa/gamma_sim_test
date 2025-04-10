# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import gamma, bernoulli
import plotly.graph_objects as go
import plotly.express as px
import time # To show progress

# --- Constants ---
CI_LOWER_BOUND = 2.5
CI_UPPER_BOUND = 97.5

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Revenue Impact Simulator (Monte Carlo)", page_icon="🎲")

# --- Custom CSS for Styling ---
# Added style for adjacent cards and adjusted card height
st.markdown("""
<style>
/* General Card Style */
.metric-card {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    padding: 15px; /* Slightly reduced padding */
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    margin-bottom: 15px; /* Reduced margin */
    text-align: center;
    height: 145px; /* Adjusted height */
    display: flex;
    flex-direction: column;
    justify-content: center;
    transition: box-shadow 0.2s ease-in-out, border-color 0.2s ease-in-out;
}
.metric-card:hover {
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    border-color: #C0C0C0;
}
/* Style for side-by-side cards */
.metric-row {
    display: flex;
    justify-content: space-between; /* Distribute space */
    margin-bottom: 10px; /* Space between rows */
}
.metric-row .metric-card {
    width: 48%; /* Approx width for two cards side-by-side with gap */
    margin-bottom: 0; /* Remove bottom margin as row handles spacing */
}


/* Card Label/Title */
.metric-label {
    font-size: 0.90em; /* Further reduced label size */
    color: #555555;
    font-weight: 600;
    margin-bottom: 6px; /* Reduced margin */
    display: flex;
    align-items: center;
    justify-content: center;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-label i {
    margin-right: 6px; /* Reduced space */
    font-size: 1.0em; /* Slightly smaller icon */
    color: #007bff;
}
/* Specific Icon Colors */
.fa-sack-dollar { color: #28a745; }
.fa-user-dollar { color: #17a2b8; }
.fa-user-slash { color: #ffc107; }
.fa-chart-line { color: #6f42c1; }


/* Main Value (Mean) */
.metric-value {
    font-size: 1.8em; /* Reduced value size */
    font-weight: 700;
    color: #212529;
    line-height: 1.1;
    margin-bottom: 6px; /* Reduced margin */
}

/* Confidence Interval Text */
.metric-ci {
    font-size: 0.75em; /* Reduced CI size */
    color: #6c757d;
    margin-top: auto;
}

/* Impact Card Specific Styles */
.impact-value-positive { color: #28a745 !important; }
.impact-value-negative { color: #dc3545 !important; }
.impact-value-neutral { color: #212529 !important; }


/* Column Headers (Removed as structure changed) */

/* Make Streamlit columns have standard gap (overriding previous custom padding) */
/* div[data-testid="stHorizontalBlock"] > div[data-testid^="stVerticalBlock"] {
    padding: 0 10px;
} */

</style>
""", unsafe_allow_html=True)

# Add Font Awesome link
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css">', unsafe_allow_html=True)


# --- Helper Functions ---
# generate_zig_data, format_currency, format_percentage remain the same
def generate_zig_data(n_customers, p_zero, k_shape, theta_scale):
    """Generates data from a Zero-Inflated Gamma distribution."""
    if not (0 <= p_zero <= 1): return np.zeros(n_customers)
    if k_shape <= 0 or theta_scale <= 0: return np.zeros(n_customers)
    is_gamma = bernoulli.rvs(1 - p_zero, size=n_customers)
    n_gamma = np.sum(is_gamma)
    data = np.zeros(n_customers)
    if n_gamma > 0:
        gamma_samples = gamma.rvs(a=k_shape, scale=theta_scale, size=n_gamma)
        gamma_samples[gamma_samples < 0] = 0
        data[is_gamma == 1] = gamma_samples
    return data

def format_currency(value):
    if not np.isfinite(value): return "N/A"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"

def format_percentage(value):
    if not np.isfinite(value): return "N/A"
    return f"{value:,.1f}"

# run_monte_carlo remains mostly the same
#@st.cache_data # Consider caching
def run_monte_carlo(n_sims, n_cust, base_p, base_k, base_th, pol_p, pol_k, pol_th):
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
        b_data = generate_zig_data(n_cust, base_p, base_k, base_th)
        b_active_data = b_data[b_data > 0]; b_n_zeros = n_cust - len(b_active_data); b_n_active = len(b_active_data)
        baseline_metrics['total_revenue'].append(np.sum(b_data))
        baseline_metrics['arpu'].append(np.mean(b_data) if n_cust > 0 else 0)
        baseline_metrics['n_zeros'].append(b_n_zeros); baseline_metrics['n_active'].append(b_n_active)
        baseline_metrics['zero_perc'].append(b_n_zeros / n_cust * 100 if n_cust > 0 else 0)
        baseline_metrics['avg_active_revenue'].append(np.mean(b_active_data) if b_n_active > 0 else 0)

        p_data = generate_zig_data(n_cust, pol_p, pol_k, pol_th)
        p_active_data = p_data[p_data > 0]; p_n_zeros = n_cust - len(p_active_data); p_n_active = len(p_active_data)
        initiative_metrics['total_revenue'].append(np.sum(p_data))
        initiative_metrics['arpu'].append(np.mean(p_data) if n_cust > 0 else 0)
        initiative_metrics['n_zeros'].append(p_n_zeros); initiative_metrics['n_active'].append(p_n_active)
        initiative_metrics['zero_perc'].append(p_n_zeros / n_cust * 100 if n_cust > 0 else 0)
        initiative_metrics['avg_active_revenue'].append(np.mean(p_active_data) if p_n_active > 0 else 0)
        if i == 0:
            first_run_data['baseline'] = b_data; first_run_data['initiative'] = p_data
        if (i + 1) % max(1, n_sims // 100) == 0 or (i + 1) == n_sims:
             progress = (i + 1) / n_sims; progress_bar.progress(progress); status_text.text(f"Running Simulation {i+1}/{n_sims}")
    progress_bar.empty(); end_time = time.time()
    status_text.text(f"Simulations Complete ({n_sims:,} runs in {end_time - start_time:.2f} seconds)")
    for key in baseline_metrics: baseline_metrics[key] = np.array(baseline_metrics[key])
    for key in initiative_metrics: initiative_metrics[key] = np.array(initiative_metrics[key])
    return baseline_metrics, initiative_metrics, first_run_data

# calculate_ci_and_mean remains the same
def calculate_ci_and_mean(data_array):
    if data_array is None or len(data_array) == 0: return np.nan, np.nan, np.nan
    mean_val = np.mean(data_array)
    finite_data = data_array[np.isfinite(data_array)]
    if len(finite_data) < 2: return mean_val, np.nan, np.nan
    ci_low = np.percentile(finite_data, CI_LOWER_BOUND)
    ci_high = np.percentile(finite_data, CI_UPPER_BOUND)
    return mean_val, ci_low, ci_high

# Card generating functions remain the same
def create_metric_card(icon_class, label, mean_value, ci_low, ci_high, format_func, unit="", bg_color="#FFFFFF", border_color="#E0E0E0"):
    formatted_mean = format_func(mean_value)
    formatted_ci = f"{format_func(ci_low)} - {format_func(ci_high)}"
    if formatted_mean == "N/A": formatted_ci = "N/A"
    card_html = f"""<div class="metric-card" style="background-color:{bg_color}; border-color:{border_color};"><div class="metric-label"><i class="{icon_class}"></i> {label}</div><div class="metric-value">{formatted_mean}{unit}</div><div class="metric-ci">95% CI: {formatted_ci}</div></div>"""
    return card_html

def create_impact_card(icon_class, label, mean_lift, ci_low, ci_high, format_func, unit="", positive_is_good=True):
    formatted_mean = format_func(mean_lift)
    formatted_ci = f"{format_func(ci_low)} - {format_func(ci_high)}"
    color_class = "impact-value-neutral"
    if np.isfinite(mean_lift) and abs(mean_lift) > 1e-9 :
        is_significantly_positive = np.isfinite(ci_low) and ci_low > 1e-9
        is_significantly_negative = np.isfinite(ci_high) and ci_high < -1e-9
        if positive_is_good:
            if is_significantly_positive: color_class = "impact-value-positive"
            elif is_significantly_negative: color_class = "impact-value-negative"
        else:
             if is_significantly_negative: color_class = "impact-value-positive"
             elif is_significantly_positive: color_class = "impact-value-negative"
    if formatted_mean == "N/A": formatted_ci = "N/A"; color_class = "impact-value-neutral"
    card_html = f"""<div class="metric-card"><div class="metric-label"><i class="{icon_class}"></i> {label}</div><div class="metric-value {color_class}">{formatted_mean}{unit}</div><div class="metric-ci">95% CI: {formatted_ci}</div></div>"""
    return card_html


# --- Title ---
st.title("🎲 Revenue Impact Simulator") # Static title is fine

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Simulation Controls")
# Simulation Run Count Selector
simulation_options = { "Fast (1,000 runs)": 1000, "Medium (5,000 runs)": 5000, "Standard (10,000 runs)": 10000, "High Precision (20,000 runs)": 20000 }
selected_sim_option = st.sidebar.selectbox("Select Simulation Precision:", options=list(simulation_options.keys()), index=2 )
N_SIMULATIONS = simulation_options[selected_sim_option]
st.sidebar.caption(f"Will perform {N_SIMULATIONS:,} simulations.")

# Baseline Parameters
st.sidebar.subheader("1. Baseline Customer Profile")
n_customers = st.sidebar.slider( "Number of Customers", min_value=1000, max_value=1000000, value=150000, step=1000, help="Total customers used in each simulation run." )
st.sidebar.markdown("**Distribution Settings (Current State):**")
current_p_zero_perc = st.sidebar.slider("Percentage of Zero-Revenue Customers (%)", 0.0, 100.0, 60.0, 0.5, help="Proportion of customers currently generating no revenue (p_zero).")
current_p_zero = current_p_zero_perc / 100.0
current_mean_active = st.sidebar.slider("Average Revenue per *Active* Customer ($)", 1.0, 1000.0, 150.0, 1.0, help="Current average revenue ONLY from active customers (k*θ > 0).")
# MODIFIED: Enhanced tooltip for k
current_k_shape = st.sidebar.slider("Revenue Consistency (Shape k)", 0.1, 10.0, 2.0, 0.1, help="Controls active revenue spread & skewness. Higher 'k' = less spread, more symmetric (less skewed). Lower 'k' = more variation & skewness (k > 0).")
current_theta_scale = current_mean_active / current_k_shape if current_k_shape > 0 else 0
valid_params = True
if current_theta_scale <= 0 and current_p_zero < 1: st.sidebar.error("Baseline Scale (θ) is non-positive."); valid_params = False
st.sidebar.info(f"*Implied:* `p_zero`: {current_p_zero:.3f}, `k`: {current_k_shape:.2f}, `θ`: {current_theta_scale:.2f}")

# Initiative Selection and Impact
st.sidebar.subheader("2. Initiative Scenario")
initiative_options = { "None": "No initiative.", "Engagement Campaign": "Activate inactive customers.", "Premium Service Tier": "Increase active spend.", "Fee Optimization": "Increase activity & avg spend.", "Loyalty Program": "Increase active spend & consistency." }
selected_initiative = st.sidebar.selectbox("Select Initiative:", options=list(initiative_options.keys()), index=0, help="Choose initiative to model.")
st.sidebar.markdown(f"**Goal:** *{initiative_options[selected_initiative]}*") # Shortened description
st.sidebar.markdown("**Hypothesized Effects:**")
impact_p_reduction=0.0; impact_mean_increase=0.0; impact_k_increase_perc=0.0 # Initialize impacts
# Impact sliders (simplified labels if possible)
if selected_initiative == "Engagement Campaign": impact_p_reduction = st.sidebar.slider("Reduce Zero Rev % By", 0.0, 100.0, 10.0, 0.5) / 100.0
elif selected_initiative == "Premium Service Tier": impact_mean_increase = st.sidebar.slider("Increase Avg Active Rev % By", 0.0, 100.0, 15.0, 0.5) / 100.0
elif selected_initiative == "Fee Optimization":
    impact_p_reduction = st.sidebar.slider("Reduce Zero Rev % By", 0.0, 50.0, 5.0, 0.5) / 100.0
    impact_mean_increase = st.sidebar.slider("Increase Avg Active Rev % By", 0.0, 50.0, 5.0, 0.5) / 100.0
elif selected_initiative == "Loyalty Program":
    impact_mean_increase = st.sidebar.slider("Increase Avg Active Rev % By", 0.0, 75.0, 10.0, 0.5) / 100.0
    impact_k_increase_perc = st.sidebar.slider("Increase Consistency (k) % By", 0.0, 50.0, 5.0, 0.5) / 100.0

# Calculate initiative parameters
initiative_p_zero = max(0, min(1, current_p_zero * (1 - impact_p_reduction)))
initiative_k_shape = max(0.01, current_k_shape * (1 + impact_k_increase_perc))
initiative_mean_active = max(0.01, current_mean_active * (1 + impact_mean_increase))
initiative_theta_scale = initiative_mean_active / initiative_k_shape if initiative_k_shape > 0 else 0
if initiative_k_shape <= 0 and initiative_p_zero < 1 and selected_initiative!="None": st.sidebar.error("Initiative k <= 0."); valid_params = False
if initiative_theta_scale <= 0 and initiative_p_zero < 1 and selected_initiative!="None": st.sidebar.error("Initiative θ <= 0."); valid_params = False
st.sidebar.info(f"*Implied:* `p`: {initiative_p_zero:.3f}, `k`: {initiative_k_shape:.2f}, `θ`: {initiative_theta_scale:.2f}, *AvgRev*: ${initiative_mean_active:,.2f}")

# Run Button
st.sidebar.subheader("3. Run Simulation")
run_simulation = st.sidebar.button(f"🚀 Run {N_SIMULATIONS:,} Simulations", disabled=not valid_params, type="primary")

# --- Main Panel ---
# Dynamic intro text (shows selected runs)
st.markdown(f"""
Analyze the potential revenue impact of customer growth initiatives using **Monte Carlo simulation** ({N_SIMULATIONS:,} runs).
Results include **95% confidence intervals**. Adjust parameters and simulate below.
""")
# Removed model description here, assumed understood or in expander

# --- Session State Init ---
if 'simulation_results' not in st.session_state: st.session_state['simulation_results'] = None
if 'simulation_run_complete' not in st.session_state: st.session_state['simulation_run_complete'] = False

# --- Simulation Execution ---
if run_simulation and valid_params:
    st.session_state['simulation_run_complete'] = False; st.session_state['simulation_results'] = None
    spinner_placeholder = st.empty()
    with spinner_placeholder, st.spinner(f"Running {N_SIMULATIONS:,} simulations..."):
        baseline_metrics_mc, initiative_metrics_mc, first_run_data = run_monte_carlo( N_SIMULATIONS, n_customers, current_p_zero, current_k_shape, current_theta_scale, initiative_p_zero, initiative_k_shape, initiative_theta_scale )
        st.session_state['simulation_results'] = { 'baseline': baseline_metrics_mc, 'initiative': initiative_metrics_mc, 'first_run': first_run_data, 'initiative_name': selected_initiative if selected_initiative != "None" else "Baseline", 'n_sims_run': N_SIMULATIONS }
        st.session_state['simulation_run_complete'] = True
        spinner_placeholder.empty()

# --- Results Display Area ---
if st.session_state.get('simulation_run_complete', False):
    results = st.session_state['simulation_results']
    if results is None: st.warning("Sim results missing."); st.stop()

    baseline_mc = results['baseline']; initiative_mc = results['initiative']; first_run = results['first_run']
    initiative_name = results['initiative_name'] if results['initiative_name'] != 'Baseline' else 'No Initiative'
    n_sims_run = results.get('n_sims_run', N_SIMULATIONS)

    st.markdown("---")
    st.header(f"📊 Simulation Results ({n_sims_run:,} runs)")

    # --- NEW: Visualizations Moved Up ---
    st.subheader("Distribution Comparison")

    # Plot 1: Sample Run Histogram (Faceted, includes zeros)
    baseline_data_first = first_run['baseline']
    initiative_data_first = first_run['initiative']

    if baseline_data_first is not None and initiative_data_first is not None:
        df_base = pd.DataFrame({'Revenue': baseline_data_first, 'Scenario': 'Baseline'})
        # Only include initiative if one was selected
        if initiative_name != 'No Initiative':
             df_init = pd.DataFrame({'Revenue': initiative_data_first, 'Scenario': f'Initiative: {initiative_name}'})
             df_plot = pd.concat([df_base, df_init])
        else:
             df_plot = df_base # Only plot baseline if 'None' selected

        # Determine a reasonable upper limit for x-axis based on non-zero data percentiles
        non_zero_rev = df_plot[df_plot['Revenue'] > 1e-6]['Revenue'] # Use small threshold
        x_max = np.percentile(non_zero_rev, 98) if not non_zero_rev.empty else 1000 # Cap at 98th percentile
        x_max = max(x_max, current_mean_active * 2) # Ensure it's at least 2x baseline mean active

        fig_sample_hist = px.histogram(df_plot, x='Revenue', color='Scenario',
                                       facet_col='Scenario', # Use facets
                                       #barmode='overlay', # No longer needed with facets
                                       histnorm='', # Use counts, not density
                                       nbins=60, # Adjust bin count as needed
                                       title='Example Revenue Distribution (Single Simulation Run)',
                                       opacity=0.8,
                                       color_discrete_map={'Baseline': '#1f77b4', f'Initiative: {initiative_name}': '#ff7f0e'}
                                       )
        fig_sample_hist.update_xaxes(range=[0, x_max * 1.05]) # Apply range limit
        fig_sample_hist.update_layout(showlegend=False, height=350, title_x=0.5)
        # Make y-axes independent for facets
        fig_sample_hist.update_yaxes(matches=None)
        st.plotly_chart(fig_sample_hist, use_container_width=True)
        st.caption(f"Illustrative histogram including zero-revenue customers (X-axis capped near {int(x_max)} for clarity).")

    else:
        st.warning("Sample run data for plotting histogram is not available.")


    # Plot 2: Distribution of Simulated Mean Outcomes (CLT illustration)
    # Choose a key metric to plot, e.g., Total Revenue
    metric_to_plot = 'total_revenue'
    metric_label = 'Total Revenue per Simulation'

    df_means_base = pd.DataFrame({metric_label: baseline_mc[metric_to_plot], 'Scenario': 'Baseline'})
    # Only include initiative if one was selected
    if initiative_name != 'No Initiative':
         df_means_init = pd.DataFrame({metric_label: initiative_mc[metric_to_plot], 'Scenario': f'Initiative: {initiative_name}'})
         df_means_plot = pd.concat([df_means_base, df_means_init])
    else:
         df_means_plot = df_means_base

    fig_clt = px.histogram(df_means_plot, x=metric_label, color='Scenario',
                           barmode='overlay', # Overlay is good here
                           marginal="rug", # Show rug plot
                           histnorm='probability density', # Normalize for shape comparison
                           nbins=50,
                           title=f'Distribution of Simulated Mean Outcomes ({metric_label})',
                           opacity=0.7,
                           color_discrete_map={'Baseline': '#1f77b4', f'Initiative: {initiative_name}': '#ff7f0e'}
                          )
    fig_clt.update_layout(height=350, title_x=0.5, legend_title_text='Scenario')
    st.plotly_chart(fig_clt, use_container_width=True)
    st.caption("Shows the distribution of mean results across all simulation runs (illustrates Central Limit Theorem).")


    st.markdown("---")
    # --- KPIs and Impact Cards (Revised Layout) ---
    st.subheader("Summary Metrics (Mean & 95% CI)")

    # Calculate summaries (moved here for cleaner access)
    results_summary = {}
    for scenario, data in [('Baseline', baseline_mc), ('Initiative', initiative_mc)]:
        results_summary[scenario] = {}
        if data is None: continue
        for metric, values in data.items():
            mean_val, ci_low, ci_high = calculate_ci_and_mean(values)
            results_summary[scenario][metric] = {'mean': mean_val, 'ci_low': ci_low, 'ci_high': ci_high}

    # Check if summaries exist
    if 'Baseline' not in results_summary or 'Initiative' not in results_summary:
         st.error("Failed to calculate summary statistics."); st.stop()
    bs = results_summary['Baseline']
    ps = results_summary['Initiative']

    # Define metrics to display side-by-side
    metric_pairs = [
        ("fa-solid fa-sack-dollar", "TOTAL REVENUE", 'total_revenue', format_currency),
        ("fa-solid fa-user-dollar", "ARPU", 'arpu', format_currency),
        ("fa-solid fa-user-slash", "% ZERO REVENUE", 'zero_perc', format_percentage, "%"),
        ("fa-solid fa-chart-line", "AVG. ACTIVE REV", 'avg_active_revenue', format_currency)
    ]

    # NEW Layout: Rows of paired cards
    for icon, label, metric_key, format_f, *unit_arg in metric_pairs:
        unit = unit_arg[0] if unit_arg else ""
        # Create cards for baseline and initiative for this metric
        html_bs = create_metric_card(icon, f"Baseline {label}", bs[metric_key]['mean'], bs[metric_key]['ci_low'], bs[metric_key]['ci_high'], format_f, unit, bg_color="#F8F9FA")
        # Handle 'No Initiative' case - show baseline again or empty card? Show baseline is less confusing.
        if initiative_name == 'No Initiative':
             html_ps = create_metric_card(icon, f"N/A {label}", bs[metric_key]['mean'], bs[metric_key]['ci_low'], bs[metric_key]['ci_high'], format_f, unit, bg_color="#FFFFFF", border_color="#F8F9FA") # Use baseline data, diff style
        else:
             html_ps = create_metric_card(icon, f"Initiative {label}", ps[metric_key]['mean'], ps[metric_key]['ci_low'], ps[metric_key]['ci_high'], format_f, unit, bg_color="#FEFBF4")

        # Display side-by-side using columns or metric-row div
        col1, col2 = st.columns(2)
        with col1: st.markdown(html_bs, unsafe_allow_html=True)
        with col2: st.markdown(html_ps, unsafe_allow_html=True)
        # Alternative using div:
        # st.markdown(f'<div class="metric-row">{html_bs}{html_ps}</div>', unsafe_allow_html=True)


    st.markdown("---")
    st.subheader("Estimated Initiative Impact")
    if initiative_name == 'No Initiative':
        st.info("No initiative selected to estimate impact.")
    else:
        # Calculate Lift metrics (moved here for cleaner access)
        revenue_lift_dist = initiative_mc['total_revenue'] - baseline_mc['total_revenue']
        mean_lift, lift_ci_low, lift_ci_high = calculate_ci_and_mean(revenue_lift_dist)
        # ... (rest of lift calculations as before) ...
        valid_baseline_rev = baseline_mc['total_revenue'][np.isfinite(baseline_mc['total_revenue']) & (baseline_mc['total_revenue'] != 0)]
        if len(valid_baseline_rev) > 0:
            indices = np.where(np.isfinite(baseline_mc['total_revenue']) & (baseline_mc['total_revenue'] != 0))[0]
            if max(indices) < len(initiative_mc['total_revenue']):
                valid_initiative_rev_match = initiative_mc['total_revenue'][indices]
                finite_mask = np.isfinite(valid_initiative_rev_match)
                valid_initiative_rev_match = valid_initiative_rev_match[finite_mask]
                valid_baseline_rev = valid_baseline_rev[finite_mask]
                if len(valid_baseline_rev) > 0:
                    revenue_lift_perc_dist = (valid_initiative_rev_match - valid_baseline_rev) / valid_baseline_rev * 100
                    revenue_lift_perc_dist = revenue_lift_perc_dist[np.isfinite(revenue_lift_perc_dist)]
                    mean_lift_perc, lift_perc_ci_low, lift_perc_ci_high = calculate_ci_and_mean(revenue_lift_perc_dist)
                else: mean_lift_perc, lift_perc_ci_low, lift_perc_ci_high = (np.nan, np.nan, np.nan)
            else: mean_lift_perc, lift_perc_ci_low, lift_perc_ci_high = (np.nan, np.nan, np.nan)
        else: mean_lift_perc, lift_perc_ci_low, lift_perc_ci_high = (np.nan, np.nan, np.nan)
        zero_perc_change_dist = initiative_mc['zero_perc'] - baseline_mc['zero_perc']
        mean_zero_change, zero_change_ci_low, zero_change_ci_high = calculate_ci_and_mean(zero_perc_change_dist)
        avg_active_change_dist = initiative_mc['avg_active_revenue'] - baseline_mc['avg_active_revenue']
        mean_avg_active_change, avg_active_change_ci_low, avg_active_change_ci_high = calculate_ci_and_mean(avg_active_change_dist)

        # Display Impact Cards
        col_imp1, col_imp2, col_imp3, col_imp4 = st.columns(4)
        with col_imp1: html = create_impact_card("fa-solid fa-arrow-up-right-dots", "Revenue Lift ($)", mean_lift, lift_ci_low, lift_ci_high, format_currency, positive_is_good=True); st.markdown(html, unsafe_allow_html=True)
        with col_imp2: html = create_impact_card("fa-solid fa-percent", "Revenue Lift (%)", mean_lift_perc, lift_perc_ci_low, lift_perc_ci_high, format_percentage, unit="%", positive_is_good=True); st.markdown(html, unsafe_allow_html=True)
        with col_imp3: html = create_impact_card("fa-solid fa-user-plus", "Change in % Zeros", mean_zero_change, zero_change_ci_low, zero_change_ci_high, format_percentage, unit=" pts", positive_is_good=False); st.markdown(html, unsafe_allow_html=True) # Changed icon
        with col_imp4: html = create_impact_card("fa-solid fa-dollar-sign", "Change in Avg Active Rev", mean_avg_active_change, avg_active_change_ci_low, avg_active_change_ci_high, format_currency, positive_is_good=True); st.markdown(html, unsafe_allow_html=True)


    # --- Explanation Expander ---
    # (Keep expander as is, it contains valuable detailed info)
    with st.expander("ℹ️ Detailed Explanations & Caveats"):
        st.markdown(f"""
        #### Understanding the Monte Carlo Simulation ({n_sims_run:,} Runs)

        1.  **Multiple Scenarios:** We simulated `{n_sims_run:,}` possible futures based on the input parameters. Each run generated `{n_customers:,}` customers for both Baseline and Initiative scenarios.
        2.  **Quantifying Uncertainty:** This approach captures inherent randomness. The **95% Confidence Interval (CI)** shows the range where we expect the true average metric to lie 95% of the time, given the model and assumptions.
        3.  **Interpreting Visualizations:**
            *   **Sample Histogram (Top):** Shows the distribution shape (including zeros) from *one* typical simulation run for Baseline vs. Initiative. Useful for seeing *how* the distribution might change. Uses facets (side-by-side plots) for clarity.
            *   **Distribution of Means (Middle):** Shows how the *average* outcome (like mean Total Revenue) varied across *all* `{n_sims_run:,}` simulations. This distribution is typically narrower and more bell-shaped (Central Limit Theorem) than the single run histogram.
            *   **Summary Cards:** Provide the overall average metric and its 95% CI based on all simulations.
            *   **Impact Cards:** Show the average *difference* between Initiative and Baseline. Color indicates impact direction (Green=Favorable). Check if the CI excludes zero for statistical significance.
        #### Key Insights Example:
        *   The initiative **`{initiative_name}`** resulted in a mean projected revenue lift of **`{format_currency(mean_lift)}`** (95% CI: `{format_currency(lift_ci_low)}` to `{format_currency(lift_ci_high)}`). Does the CI exclude zero?
        *   Identify primary drivers by comparing the CIs for **Change in % Zeros** and **Change in Avg Active Rev**.
        #### Important Caveats:
        *   Model accuracy depends on assumptions. Simulation != Forecast. Parameter uncertainty exists. External factors omitted.
        """)

elif not valid_params:
     st.error("Simulation cannot run due to invalid parameters. Check sidebar.")
else:
    st.info(f"Adjust parameters and click '🚀 Run {N_SIMULATIONS:,} Simulations'.")

# Footer
st.markdown("---")
st.caption("Monte Carlo Revenue Simulator v1.3")
