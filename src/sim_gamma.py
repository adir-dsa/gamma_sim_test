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
def generate_zig_data(n_customers, p_zero, k_shape, theta_scale):
    """Generates data from a Zero-Inflated Gamma distribution."""
    # Basic validation to prevent errors in stats functions
    p_zero = max(0.0, min(1.0, p_zero if np.isfinite(p_zero) else 0.0))
    k_shape = max(0.01, k_shape if np.isfinite(k_shape) and k_shape > 0 else 0.01) # Ensure k>0
    theta_scale = max(0.01, theta_scale if np.isfinite(theta_scale) and theta_scale > 0 else 0.01) # Ensure theta>0

    if n_customers <= 0: return np.array([]) # Handle zero customers case

    try:
        is_gamma = bernoulli.rvs(1 - p_zero, size=n_customers)
        n_gamma = np.sum(is_gamma)
        data = np.zeros(n_customers)
        if n_gamma > 0:
            gamma_samples = gamma.rvs(a=k_shape, scale=theta_scale, size=n_gamma)
            gamma_samples[gamma_samples < 0] = 0 # Ensure non-negative
            data[is_gamma == 1] = gamma_samples
        return data
    except Exception as e:
        # If stats generation fails for some reason, return zeros
        st.warning(f"Error generating ZIG data (p={p_zero:.2f}, k={k_shape:.2f}, θ={theta_scale:.2f}): {e}. Returning zeros.")
        return np.zeros(n_customers)


def format_currency(value):
    if not np.isfinite(value) or value is None: return "N/A" # Added None check
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"

def format_percentage(value):
    if not np.isfinite(value) or value is None: return "N/A" # Added None check
    # Distinguish between percentage points and % sign
    if abs(value) > 100: # Likely a raw percentage like 60.0
         return f"{value:,.1f}"
    else: # Likely a change or requires a % sign formatting implicitly
         return f"{value:,.1f}" # Let the unit add the '%' or ' pts'

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
        # Generate Baseline Data for this run
        b_data = generate_zig_data(n_cust, base_p, base_k, base_th)
        b_active_data = b_data[b_data > 0]; b_n_zeros = n_cust - len(b_active_data); b_n_active = len(b_active_data)
        baseline_metrics['total_revenue'].append(np.sum(b_data))
        baseline_metrics['arpu'].append(np.mean(b_data) if n_cust > 0 else 0)
        baseline_metrics['n_zeros'].append(b_n_zeros); baseline_metrics['n_active'].append(b_n_active)
        baseline_metrics['zero_perc'].append(b_n_zeros / n_cust * 100 if n_cust > 0 else 0)
        baseline_metrics['avg_active_revenue'].append(np.mean(b_active_data) if b_n_active > 0 else 0)

        # Generate Initiative Data for this run
        p_data = generate_zig_data(n_cust, pol_p, pol_k, pol_th)
        p_active_data = p_data[p_data > 0]; p_n_zeros = n_cust - len(p_active_data); p_n_active = len(p_active_data)
        initiative_metrics['total_revenue'].append(np.sum(p_data))
        initiative_metrics['arpu'].append(np.mean(p_data) if n_cust > 0 else 0)
        initiative_metrics['n_zeros'].append(p_n_zeros); initiative_metrics['n_active'].append(p_n_active)
        initiative_metrics['zero_perc'].append(p_n_zeros / n_cust * 100 if n_cust > 0 else 0)
        initiative_metrics['avg_active_revenue'].append(np.mean(p_active_data) if p_n_active > 0 else 0)

        # Store data from the very first simulation run for plotting
        if i == 0:
            first_run_data['baseline'] = b_data; first_run_data['initiative'] = p_data

        # Update progress
        if (i + 1) % max(1, n_sims // 100) == 0 or (i + 1) == n_sims:
             progress = (i + 1) / n_sims; progress_bar.progress(progress); status_text.text(f"Running Simulation {i+1}/{n_sims}")

    progress_bar.empty(); end_time = time.time()
    status_text.text(f"Simulations Complete ({n_sims:,} runs in {end_time - start_time:.2f} seconds)")

    # Convert lists to numpy arrays
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
             if is_significantly_negative: color_class = "impact-value-positive" # Negative change is good (e.g., fewer zeros)
             elif is_significantly_positive: color_class = "impact-value-negative"
    if formatted_mean == "N/A": formatted_ci = "N/A"; color_class = "impact-value-neutral"
    # Add unit directly after value in the card
    card_html = f"""<div class="metric-card"><div class="metric-label"><i class="{icon_class}"></i> {label}</div><div class="metric-value {color_class}">{formatted_mean}{unit}</div><div class="metric-ci">95% CI: {formatted_ci}</div></div>"""
    return card_html


# --- Title ---
st.title("🎲 Revenue Impact Simulator")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Simulation Controls")
# Simulation Run Count Selector
simulation_options = { "Fast (1,000 runs)": 1000, "Medium (5,000 runs)": 5000, "Standard (10,000 runs)": 10000, "High Precision (20,000 runs)": 20000 }
selected_sim_option = st.sidebar.selectbox("Select Simulation Precision:", options=list(simulation_options.keys()), index=2 )
N_SIMULATIONS = simulation_options[selected_sim_option]
st.sidebar.caption(f"Will perform {N_SIMULATIONS:,} simulations.")

# Baseline Parameters
st.sidebar.subheader("1. Baseline Customer Profile")
# Use a smaller number for the preview plot for responsiveness, full number for simulation
N_CUSTOMERS_FULL = st.sidebar.slider( "Number of Customers (for Simulation)", min_value=1000, max_value=1000000, value=150000, step=1000, help="Total customers used in each full simulation run." )
N_CUSTOMERS_PREVIEW = min(N_CUSTOMERS_FULL, 5000) # Cap preview size for speed
st.sidebar.markdown("**Distribution Settings (Current State):**")
current_p_zero_perc = st.sidebar.slider("Percentage of Zero-Revenue Customers (%)", 0.0, 100.0, 60.0, 0.5, help="Proportion of customers currently generating no revenue (p_zero).")
current_p_zero = current_p_zero_perc / 100.0
current_mean_active = st.sidebar.slider("Average Revenue per *Active* Customer ($)", 1.0, 1000.0, 150.0, 1.0, help="Current average revenue ONLY from active customers (k*θ > 0).")
current_k_shape = st.sidebar.slider("Revenue Consistency (Shape k)", 0.1, 10.0, 2.0, 0.1, help="Controls active revenue spread & skewness. Higher 'k' = less spread, more symmetric (less skewed). Lower 'k' = more variation & skewness (k > 0).")
# Ensure k_shape is positive before division
current_theta_scale = current_mean_active / current_k_shape if current_k_shape > 0 else np.nan # Use NaN if k=0

# Check baseline validity (for info and disabling run button)
valid_base_params = True
if not (0 <= current_p_zero <= 1):
    st.sidebar.warning("Baseline Zero % must be between 0 and 100.")
    valid_base_params = False
if current_k_shape <= 0 and current_p_zero < 1:
    st.sidebar.warning("Baseline Shape (k) must be > 0 if not all customers are zero-revenue.")
    # Don't set valid_base_params to False here, allow preview with default theta=0.01 maybe?
    current_theta_scale = 0.01 # Assign a minimal scale for preview if k is bad
if current_theta_scale <= 0 and current_p_zero < 1:
    st.sidebar.warning("Baseline Scale (θ) must be > 0 if not all customers are zero-revenue.")
    valid_base_params = False

# Calculate theta string separately to handle NaN case before f-string formatting
theta_baseline_str = f"{current_theta_scale:.2f}" if np.isfinite(current_theta_scale) else "N/A"
st.sidebar.info(f"*Implied Baseline:* `p`: {current_p_zero:.3f}, `k`: {current_k_shape:.2f}, `θ`: {theta_baseline_str}")

# Initiative Selection and Impact
st.sidebar.subheader("2. Initiative Scenario")
initiative_options = { "None": "No initiative.", "Engagement Campaign": "Activate inactive customers.", "Premium Service Tier": "Increase active spend.", "Fee Optimization": "Increase activity & avg spend.", "Loyalty Program": "Increase active spend & consistency." }
selected_initiative = st.sidebar.selectbox("Select Initiative:", options=list(initiative_options.keys()), index=0, help="Choose initiative to model.")
st.sidebar.markdown(f"**Goal:** *{initiative_options[selected_initiative]}*")
st.sidebar.markdown("**Hypothesized Effects:**")
impact_p_reduction_perc=0.0; impact_mean_increase_perc=0.0; impact_k_increase_perc=0.0 # Use percentage inputs

if selected_initiative == "Engagement Campaign": impact_p_reduction_perc = st.sidebar.slider("Reduce Zero Rev % By (relative %)", 0.0, 100.0, 10.0, 0.5, help="E.g., 10% reduction on 60% zeros -> 54% zeros.")
elif selected_initiative == "Premium Service Tier": impact_mean_increase_perc = st.sidebar.slider("Increase Avg Active Rev % By", 0.0, 100.0, 15.0, 0.5)
elif selected_initiative == "Fee Optimization":
    impact_p_reduction_perc = st.sidebar.slider("Reduce Zero Rev % By (relative %)", 0.0, 50.0, 5.0, 0.5)
    impact_mean_increase_perc = st.sidebar.slider("Increase Avg Active Rev % By", 0.0, 50.0, 5.0, 0.5)
elif selected_initiative == "Loyalty Program":
    impact_mean_increase_perc = st.sidebar.slider("Increase Avg Active Rev % By", 0.0, 75.0, 10.0, 0.5)
    impact_k_increase_perc = st.sidebar.slider("Increase Consistency (k) % By", 0.0, 50.0, 5.0, 0.5)

# Calculate initiative parameters
impact_p_reduction = impact_p_reduction_perc / 100.0
impact_mean_increase = impact_mean_increase_perc / 100.0
impact_k_increase = impact_k_increase_perc / 100.0

initiative_p_zero = max(0, min(1, current_p_zero * (1 - impact_p_reduction))) if selected_initiative != "None" else current_p_zero
initiative_k_shape = max(0.01, current_k_shape * (1 + impact_k_increase)) if selected_initiative != "None" else current_k_shape
initiative_mean_active = max(0.01, current_mean_active * (1 + impact_mean_increase)) if selected_initiative != "None" else current_mean_active
initiative_theta_scale = initiative_mean_active / initiative_k_shape if initiative_k_shape > 0 else np.nan # Use NaN if k=0

# Check initiative validity
valid_initiative_params = True
if selected_initiative != "None":
    if initiative_k_shape <= 0 and initiative_p_zero < 1:
        st.sidebar.warning("Initiative Shape (k) must be > 0 if not all customers are zero-revenue.")
        initiative_theta_scale = 0.01 # Assign minimal scale for preview
    if initiative_theta_scale <= 0 and initiative_p_zero < 1:
        st.sidebar.warning("Initiative Scale (θ) must be > 0 if not all customers are zero-revenue.")
        valid_initiative_params = False
    # Calculate initiative theta string separately
    theta_initiative_str = f"{initiative_theta_scale:.2f}" if np.isfinite(initiative_theta_scale) else "N/A"
    # Format initiative mean active separately as well for clarity
    formatted_initiative_mean_active = f"{initiative_mean_active:,.2f}" if np.isfinite(initiative_mean_active) else "N/A"
    st.sidebar.info(f"*Implied Initiative:* `p`: {initiative_p_zero:.3f}, `k`: {initiative_k_shape:.2f}, `θ`: {theta_initiative_str}, *AvgRev*: ${formatted_initiative_mean_active}")
else:
    # For "None" initiative, params are same as baseline
    initiative_p_zero = current_p_zero
    initiative_k_shape = current_k_shape
    initiative_theta_scale = current_theta_scale


# Determine overall parameter validity for running the simulation
valid_params_for_run = valid_base_params and valid_initiative_params

# Run Button
st.sidebar.subheader("3. Run Simulation")
run_simulation = st.sidebar.button(f"🚀 Run {N_SIMULATIONS:,} Simulations", disabled=not valid_params_for_run, type="primary")
if not valid_params_for_run:
     st.sidebar.error("Cannot run simulation due to invalid parameters. Check warnings above.")


# --- Main Panel ---
# Dynamic intro text (shows selected runs)
st.markdown(f"""
Analyze the potential revenue impact of customer growth initiatives using **Monte Carlo simulation** ({N_SIMULATIONS:,} runs).
Adjust parameters in the sidebar and see the assumed **initial distribution** below. Click Run to see simulated outcomes.
""")

# --- NEW: Initial Distribution Display ---
st.subheader("Assumed Customer Distribution (Based on Current Settings)")
st.caption(f"This graph shows a sample ({N_CUSTOMERS_PREVIEW:,} customers) based on sidebar parameters and updates live.")

try:
    # Generate sample data based *current* sidebar values using N_CUSTOMERS_PREVIEW
    initial_baseline_data = generate_zig_data(N_CUSTOMERS_PREVIEW, current_p_zero, current_k_shape, current_theta_scale)
    initial_initiative_data = generate_zig_data(N_CUSTOMERS_PREVIEW, initiative_p_zero, initiative_k_shape, initiative_theta_scale)

    # Create DataFrame for plotting
    df_initial_base = pd.DataFrame({'Revenue': initial_baseline_data, 'Scenario': 'Baseline'})
    scenario_dfs = [df_initial_base]
    initiative_label = f'Initiative: {selected_initiative}' if selected_initiative != "None" else 'Baseline' # Use 'Baseline' if None selected

    # Only add initiative data if an initiative is actually selected
    if selected_initiative != "None":
        df_initial_init = pd.DataFrame({'Revenue': initial_initiative_data, 'Scenario': initiative_label})
        scenario_dfs.append(df_initial_init)
        df_initial_plot = pd.concat(scenario_dfs)
        color_map = {'Baseline': '#1f77b4', initiative_label: '#ff7f0e'}
    else:
        # If no initiative, just plot baseline
        df_initial_plot = df_initial_base
        color_map = {'Baseline': '#1f77b4'}


    # Determine x-axis limit for the initial plot - based on PREVIEW data
    non_zero_rev_initial = df_initial_plot[df_initial_plot['Revenue'] > 1e-6]['Revenue']
    if not non_zero_rev_initial.empty:
        # Use 98th percentile of non-zero data, but ensure it's at least 2x the mean(s)
        x_max_initial = np.percentile(non_zero_rev_initial, 98)
        max_mean = current_mean_active
        if selected_initiative != "None" and np.isfinite(initiative_mean_active):
             max_mean = max(max_mean, initiative_mean_active)
        x_max_initial = max(x_max_initial, max_mean * 2.5) # Make limit a bit higher than 2x mean
    else:
        x_max_initial = max(10, current_mean_active * 2.5) # Fallback if only zeros


    # Create and display the plot using Plotly Express histogram
    fig_initial_dist = px.histogram(
        df_initial_plot,
        x='Revenue',
        color='Scenario',
        facet_col='Scenario', # Separate plots side-by-side
        histnorm='', # Use counts
        nbins=50, # Adjust bins as needed
        title='Example Input Revenue Distribution (Single Sample)',
        opacity=0.8,
        color_discrete_map=color_map
    )
    fig_initial_dist.update_xaxes(range=[0, x_max_initial * 1.05]) # Apply range limit + 5% margin
    fig_initial_dist.update_layout(
        showlegend=False,
        height=350,
        title_x=0.5,
        yaxis_title="Number of Customers" # Add Y axis label
        )
    # Make y-axes independent for facets to handle potentially different scales
    fig_initial_dist.update_yaxes(matches=None, title_text="Number of Customers", row=1, col=1)
    fig_initial_dist.update_yaxes(matches=None, title_text="Number of Customers", row=1, col=2) # Assuming max 2 cols
    # Explicitly set titles for facets
    # Get scenario names (handles 'None' case where only 'Baseline' exists)
    scenario_names = df_initial_plot['Scenario'].unique()
    for i, scenario_name in enumerate(scenario_names):
        fig_initial_dist.update_xaxes(title_text="Revenue per Customer", row=1, col=i+1)


    st.plotly_chart(fig_initial_dist, use_container_width=True)
    st.caption(f"Illustrative histogram using current parameters, including zero-revenue customers (X-axis capped near {format_currency(x_max_initial)} for clarity).")

except Exception as e:
    st.error(f"Error generating initial distribution plot: {e}")
    st.warning("Please ensure parameters in the sidebar are valid (e.g., Shape k > 0, Avg Active Rev > 0).")


# --- Session State Init ---
if 'simulation_results' not in st.session_state: st.session_state['simulation_results'] = None
if 'simulation_run_complete' not in st.session_state: st.session_state['simulation_run_complete'] = False

# --- Simulation Execution ---
if run_simulation and valid_params_for_run:
    st.session_state['simulation_run_complete'] = False # Reset flag
    st.session_state['simulation_results'] = None # Clear previous results

    # Display spinner and status text DURING the simulation run
    spinner_placeholder = st.empty()
    # Note: The run_monte_carlo function now handles the progress bar internally
    with spinner_placeholder, st.spinner(f"Preparing {N_SIMULATIONS:,} simulations..."):
         # Now run with the FULL number of customers
        baseline_metrics_mc, initiative_metrics_mc, first_run_data = run_monte_carlo(
            N_SIMULATIONS,
            N_CUSTOMERS_FULL, # Use the full number here
            current_p_zero, current_k_shape, current_theta_scale,
            initiative_p_zero, initiative_k_shape, initiative_theta_scale
            )
        st.session_state['simulation_results'] = {
             'baseline': baseline_metrics_mc,
             'initiative': initiative_metrics_mc,
             'first_run': first_run_data,
             'initiative_name': selected_initiative, # Store the name used
             'n_sims_run': N_SIMULATIONS,
             'n_cust_run': N_CUSTOMERS_FULL
             }
        st.session_state['simulation_run_complete'] = True
        spinner_placeholder.empty() # Clear spinner area
        # Optionally force a rerun to ensure the results section displays immediately
        # st.experimental_rerun() # Use st.rerun() in newer versions

# --- Results Display Area ---
# This section only appears *after* a simulation has successfully run
st.markdown("---") # Separator before results
if st.session_state.get('simulation_run_complete', False):
    results = st.session_state['simulation_results']
    if results is None:
        st.warning("Simulation results are not available. Please run the simulation.")
        st.stop()

    # Unpack results
    baseline_mc = results['baseline']; initiative_mc = results['initiative']; first_run = results['first_run']
    initiative_name = results['initiative_name'] if results['initiative_name'] != "None" else "Baseline" # Display 'Baseline' if None was run
    n_sims_run = results.get('n_sims_run', N_SIMULATIONS)
    n_cust_run = results.get('n_cust_run', N_CUSTOMERS_FULL)


    st.header(f"📊 Simulation Results ({n_sims_run:,} runs)")
    st.caption(f"Based on {n_cust_run:,} customers per run.")

    # --- Visualizations Moved Up ---
    st.subheader("Distribution Comparison (Post-Simulation)")

    # Plot 1: Sample Run Histogram (Faceted, includes zeros) - FROM SIMULATION
    baseline_data_first = first_run.get('baseline') # Use .get for safety
    initiative_data_first = first_run.get('initiative')

    if baseline_data_first is not None and initiative_data_first is not None:
        df_base = pd.DataFrame({'Revenue': baseline_data_first, 'Scenario': 'Baseline'})
        sim_plot_dfs = [df_base]
        sim_initiative_label = f'Initiative: {initiative_name}' if initiative_name != 'Baseline' else 'Baseline'

        if initiative_name != 'Baseline':
             df_init = pd.DataFrame({'Revenue': initiative_data_first, 'Scenario': sim_initiative_label})
             sim_plot_dfs.append(df_init)
             df_sim_plot = pd.concat(sim_plot_dfs)
             sim_color_map = {'Baseline': '#1f77b4', sim_initiative_label: '#ff7f0e'}
        else:
             df_sim_plot = df_base
             sim_color_map = {'Baseline': '#1f77b4'}


        # Determine a reasonable upper limit for x-axis based on non-zero data percentiles - FROM SIMULATION DATA
        non_zero_rev_sim = df_sim_plot[df_sim_plot['Revenue'] > 1e-6]['Revenue']
        if not non_zero_rev_sim.empty:
             x_max_sim = np.percentile(non_zero_rev_sim, 98)
             # Get means used in the simulation run (can recalculate or fetch from inputs used)
             # Re-calculate theta to be sure, as it might have been NaN before run if k=0 initially
             base_theta_run = current_mean_active / current_k_shape if current_k_shape > 0 else 0
             init_mean_run = max(0.01, current_mean_active * (1 + impact_mean_increase)) if results['initiative_name'] != "None" else current_mean_active
             max_mean_run = max(current_mean_active, init_mean_run if np.isfinite(init_mean_run) else 0)
             x_max_sim = max(x_max_sim, max_mean_run * 2.5)
        else:
             x_max_sim = 1000 # Fallback


        fig_sample_hist = px.histogram(df_sim_plot, x='Revenue', color='Scenario',
                                       facet_col='Scenario',
                                       histnorm='', # counts
                                       nbins=60,
                                       title='Example Revenue Distribution (First Simulation Run)',
                                       opacity=0.8,
                                       color_discrete_map=sim_color_map
                                       )
        fig_sample_hist.update_xaxes(range=[0, x_max_sim * 1.05])
        fig_sample_hist.update_layout(showlegend=False, height=350, title_x=0.5)
        # Make y-axes independent for facets
        fig_sample_hist.update_yaxes(matches=None, title_text="Number of Customers", row=1, col=1)
        fig_sample_hist.update_yaxes(matches=None, title_text="Number of Customers", row=1, col=2)
        # Explicitly set titles for facets
        sim_scenario_names = df_sim_plot['Scenario'].unique()
        for i, scenario_name in enumerate(sim_scenario_names):
             fig_sample_hist.update_xaxes(title_text="Revenue per Customer", row=1, col=i+1)

        st.plotly_chart(fig_sample_hist, use_container_width=True)
        st.caption(f"Illustrative histogram from the *first* simulation run ({n_cust_run:,} customers), including zeros (X-axis capped near {format_currency(x_max_sim)}).")

    else:
        st.warning("Sample run data for plotting simulation histogram is not available.")


    # Plot 2: Distribution of Simulated Mean Outcomes (CLT illustration)
    metric_to_plot = 'total_revenue'
    metric_label = 'Total Revenue per Simulation'

    df_means_base = pd.DataFrame({metric_label: baseline_mc[metric_to_plot], 'Scenario': 'Baseline'})
    clt_plot_dfs = [df_means_base]
    if initiative_name != 'Baseline':
         df_means_init = pd.DataFrame({metric_label: initiative_mc[metric_to_plot], 'Scenario': sim_initiative_label})
         clt_plot_dfs.append(df_means_init)
         df_means_plot = pd.concat(clt_plot_dfs)
    else:
         df_means_plot = df_means_base


    fig_clt = px.histogram(df_means_plot, x=metric_label, color='Scenario',
                           barmode='overlay',
                           marginal="rug", # Show rug plot
                           histnorm='probability density', # Normalize for shape comparison
                           nbins=50,
                           title=f'Distribution of Simulated Outcomes ({metric_label})',
                           opacity=0.7,
                           color_discrete_map=sim_color_map
                          )
    fig_clt.update_layout(height=350, title_x=0.5, legend_title_text='Scenario', xaxis_title=metric_label, yaxis_title="Density")
    st.plotly_chart(fig_clt, use_container_width=True)
    st.caption("Shows the distribution of results across all simulation runs (illustrates Central Limit Theorem).")


    # --- KPIs and Impact Cards (Revised Layout) ---
    st.markdown("---")
    st.subheader("Summary Metrics (Mean & 95% CI)")

    # Calculate summaries
    results_summary = {}
    for scenario_key, data in [('Baseline', baseline_mc), ('Initiative', initiative_mc)]:
        results_summary[scenario_key] = {}
        if data is None: continue
        for metric, values in data.items():
            mean_val, ci_low, ci_high = calculate_ci_and_mean(values)
            results_summary[scenario_key][metric] = {'mean': mean_val, 'ci_low': ci_low, 'ci_high': ci_high}

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

    # Layout: Rows of paired cards
    for icon, label, metric_key, format_f, *unit_arg in metric_pairs:
        unit = unit_arg[0] if unit_arg else ""
        # Create cards for baseline and initiative for this metric
        html_bs = create_metric_card(icon, f"Baseline {label}", bs[metric_key]['mean'], bs[metric_key]['ci_low'], bs[metric_key]['ci_high'], format_f, unit, bg_color="#F8F9FA")
        # Handle 'No Initiative' case - show baseline again but styled differently
        if initiative_name == 'Baseline': # Changed check here
             # Show baseline data again, but visually distinct or labelled as N/A
             html_ps = create_metric_card(icon, f"N/A {label}", bs[metric_key]['mean'], bs[metric_key]['ci_low'], bs[metric_key]['ci_high'], format_f, unit, bg_color="#FFFFFF", border_color="#F8F9FA")
        else:
             html_ps = create_metric_card(icon, f"Initiative {label}", ps[metric_key]['mean'], ps[metric_key]['ci_low'], ps[metric_key]['ci_high'], format_f, unit, bg_color="#FEFBF4") # Light yellow for initiative

        # Display side-by-side using columns
        col1, col2 = st.columns(2)
        with col1: st.markdown(html_bs, unsafe_allow_html=True)
        with col2: st.markdown(html_ps, unsafe_allow_html=True)


    # --- Estimated Initiative Impact ---
    st.markdown("---")
    st.subheader("Estimated Initiative Impact")
    if initiative_name == 'Baseline': # Changed check here
        st.info("No initiative selected to estimate impact (showing baseline vs baseline).")
    else:
        # Calculate Lift metrics
        revenue_lift_dist = initiative_mc['total_revenue'] - baseline_mc['total_revenue']
        mean_lift, lift_ci_low, lift_ci_high = calculate_ci_and_mean(revenue_lift_dist)

        # Calculate Percentage Lift (handle division by zero or near-zero baseline)
        mean_lift_perc, lift_perc_ci_low, lift_perc_ci_high = np.nan, np.nan, np.nan # Default to NaN
        # Ensure baseline revenue mean is positive and finite before calculating % lift
        if bs['total_revenue']['mean'] is not None and np.isfinite(bs['total_revenue']['mean']) and bs['total_revenue']['mean'] > 1e-6:
             valid_baseline_rev = baseline_mc['total_revenue'][np.isfinite(baseline_mc['total_revenue']) & (baseline_mc['total_revenue'] != 0)]
             if len(valid_baseline_rev) > 0:
                 indices = np.where(np.isfinite(baseline_mc['total_revenue']) & (baseline_mc['total_revenue'] != 0))[0]
                 # Ensure indices are within bounds of initiative results
                 if len(indices) > 0 and max(indices) < len(initiative_mc['total_revenue']):
                     valid_initiative_rev_match = initiative_mc['total_revenue'][indices]
                     # Further filter based on initiative values being finite
                     finite_mask = np.isfinite(valid_initiative_rev_match)
                     valid_initiative_rev_match = valid_initiative_rev_match[finite_mask]
                     valid_baseline_rev = valid_baseline_rev[finite_mask] # Match baseline filtering

                     if len(valid_baseline_rev) > 0: # Check again after filtering
                         revenue_lift_perc_dist = (valid_initiative_rev_match - valid_baseline_rev) / valid_baseline_rev * 100
                         revenue_lift_perc_dist = revenue_lift_perc_dist[np.isfinite(revenue_lift_perc_dist)] # Filter out NaNs/Infs from division
                         if len(revenue_lift_perc_dist) > 0:
                             mean_lift_perc, lift_perc_ci_low, lift_perc_ci_high = calculate_ci_and_mean(revenue_lift_perc_dist)


        # Calculate Change in % Zeros (Absolute points change)
        zero_perc_change_dist = initiative_mc['zero_perc'] - baseline_mc['zero_perc']
        mean_zero_change, zero_change_ci_low, zero_change_ci_high = calculate_ci_and_mean(zero_perc_change_dist)

        # Calculate Change in Avg Active Revenue
        avg_active_change_dist = initiative_mc['avg_active_revenue'] - baseline_mc['avg_active_revenue']
        mean_avg_active_change, avg_active_change_ci_low, avg_active_change_ci_high = calculate_ci_and_mean(avg_active_change_dist)

        # Display Impact Cards
        col_imp1, col_imp2, col_imp3, col_imp4 = st.columns(4)
        with col_imp1: html = create_impact_card("fa-solid fa-arrow-up-right-dots", "Revenue Lift ($)", mean_lift, lift_ci_low, lift_ci_high, format_currency, positive_is_good=True); st.markdown(html, unsafe_allow_html=True)
        with col_imp2: html = create_impact_card("fa-solid fa-percent", "Revenue Lift (%)", mean_lift_perc, lift_perc_ci_low, lift_perc_ci_high, format_percentage, unit="%", positive_is_good=True); st.markdown(html, unsafe_allow_html=True)
        with col_imp3: html = create_impact_card("fa-solid fa-user-plus", "Change in % Zeros", mean_zero_change, zero_change_ci_low, zero_change_ci_high, format_percentage, unit=" pts", positive_is_good=False); st.markdown(html, unsafe_allow_html=True)
        with col_imp4: html = create_impact_card("fa-solid fa-dollar-sign", "Change in Avg Active Rev", mean_avg_active_change, avg_active_change_ci_low, avg_active_change_ci_high, format_currency, positive_is_good=True); st.markdown(html, unsafe_allow_html=True)


    # --- Explanation Expander ---
    with st.expander("ℹ️ Detailed Explanations & Caveats"):
        st.markdown(f"""
        #### Understanding the Monte Carlo Simulation ({n_sims_run:,} Runs)

        1.  **Multiple Scenarios:** We simulated `{n_sims_run:,}` possible futures based on the input parameters. Each run generated `{n_cust_run:,}` customers for both Baseline and Initiative scenarios.
        2.  **Quantifying Uncertainty:** This approach captures inherent randomness. The **95% Confidence Interval (CI)** shows the range where we expect the true average metric to lie 95% of the time, given the model and assumptions.
        3.  **Interpreting Visualizations:**
            *   **Assumed Distribution (Top):** Shows the theoretical distribution shape (based on parameters) used as *input* to the simulation. Updates live as you change sidebar controls.
            *   **Example Distribution (Post-Sim):** Shows the distribution from *one* typical simulation run. Useful for seeing *how* the distribution looked in a single instance.
            *   **Distribution of Outcomes (Post-Sim):** Shows how the *average* outcome (like mean Total Revenue) varied across *all* `{n_sims_run:,}` simulations. This distribution is typically narrower and more bell-shaped (Central Limit Theorem) than the single run histogram.
            *   **Summary Cards:** Provide the overall average metric and its 95% CI based on all simulations.
            *   **Impact Cards:** Show the average *difference* between Initiative and Baseline. Color indicates impact direction (Green=Favorable). Check if the CI excludes zero for statistical significance (at the 95% level).
        #### Key Insights Example:
        *   If an initiative was run, the mean projected revenue lift is **`{format_currency(mean_lift) if initiative_name != 'Baseline' else 'N/A'}`** (95% CI: `{format_currency(lift_ci_low) if initiative_name != 'Baseline' else 'N/A'}` to `{format_currency(lift_ci_high) if initiative_name != 'Baseline' else 'N/A'}`). Does the CI exclude zero?
        *   Identify primary drivers by comparing the CIs for **Change in % Zeros** and **Change in Avg Active Rev**.
        #### Important Caveats:
        *   Model accuracy depends heavily on the chosen parameters (`p_zero`, `k`, `θ`) accurately reflecting reality.
        *   Simulation provides a range of possibilities based on assumptions; it is not a crystal ball forecast.
        *   External market factors, competitor actions, and implementation quality are not modeled here.
        """)

elif not run_simulation and not valid_params_for_run:
     # This message is now mainly handled by the sidebar error and button disabling
     # st.warning("Adjust parameters in the sidebar. Simulation cannot run with current settings.")
     pass # Keep the main area clean, sidebar has the error
elif not run_simulation and valid_params_for_run:
    # Only show the prompt to run if parameters are valid and run hasn't been clicked
    st.info(f"Parameters are set. Click '🚀 Run {N_SIMULATIONS:,} Simulations' in the sidebar to see detailed results.")


# Footer
st.markdown("---")
st.caption("Monte Carlo Revenue Simulator v1.4")
