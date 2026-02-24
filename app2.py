import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --- 1. ENHANCED NUMBER FORMATTER (xx.xxx.xxx ₫) ---
def format_vnd(amount):
    if amount is None or amount == 0:
        return "0 ₫"
    return "{:,.0f} ₫".format(amount).replace(",", ".")

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="Emergency Fund Adequate Calculator", layout="wide")

st.title("Smart Emergency Fund Adequate Calculator")

# --- 3. CALLBACKS FOR SYNCHRONIZATION ---
def update_slider(key_input, key_slider):
    st.session_state[key_slider] = st.session_state[key_input]

def update_input(key_input, key_slider):
    st.session_state[key_input] = st.session_state[key_slider]

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("📋 Essential Monthly Expenses")
    housing = st.number_input("Housing (Rent/Mortgage)", min_value=0, step=500000, value=0)
    utilities = st.number_input("Utilities (Elec/Water/Internet)", min_value=0, step=100000, value=0)
    transport = st.number_input("Transportation", min_value=0, step=100000, value=0)
    food = st.number_input("Food & Groceries", min_value=0, step=500000, value=0)
    health = st.number_input("Health & Insurance", min_value=0, step=100000, value=0)
    debt = st.number_input("Non-mortgage debt payments", min_value=0, step=500000, value=0)
    childcare = st.number_input("Childcare", min_value=0, step=500000, value=0)
    education = st.number_input("Education", min_value=0, step=500000, value=0)
    alimony = st.number_input("Child support and alimony", min_value=0, step=500000, value=0)
    other_ess = st.number_input("Other essential expenses", min_value=0, step=100000, value=0)
    
    exp_dict = {
        "Housing": housing, "Utilities": utilities, "Transport": transport,
        "Food": food, "Health": health, "Debt": debt, "Childcare": childcare,
        "Education": education, "Alimony": alimony, "Other": other_ess
    }
    e_basic = sum(exp_dict.values())

    st.divider()
    st.header("💰 Income & Savings Info")
    
    if 'inc_val' not in st.session_state: st.session_state['inc_val'] = 0
    st.number_input("Monthly Gross Income (₫)", min_value=0, max_value=500000000, step=500000, key='inc_val', on_change=update_slider, args=('inc_val', 'inc_sld'))
    st.slider("Income Slider", min_value=0, max_value=500000000, step=500000, key='inc_sld', on_change=update_input, args=('inc_val', 'inc_sld'), label_visibility="collapsed")
    income = st.session_state['inc_val']

    if 'bal_val' not in st.session_state: st.session_state['bal_val'] = 0
    st.number_input("Current emergency funds available (₫)", min_value=0, max_value=1000000000, step=500000, key='bal_val', on_change=update_slider, args=('bal_val', 'bal_sld'))
    st.slider("Fund Slider", min_value=0, max_value=1000000000, step=500000, key='bal_sld', on_change=update_input, args=('bal_val', 'bal_sld'), label_visibility="collapsed")
    current_bal = st.session_state['bal_val']

    if 'sav_val' not in st.session_state: st.session_state['sav_val'] = 0
    st.number_input("Amount you can save monthly (₫)", min_value=0, max_value=max(income, 1), step=500000, key='sav_val', on_change=update_slider, args=('sav_val', 'sav_sld'))
    st.slider("Savings Slider", min_value=0, max_value=max(income, 1), step=500000, key='sav_sld', on_change=update_input, args=('sav_val', 'sav_sld'), label_visibility="collapsed")
    monthly_savings = st.session_state['sav_val']

    rate_earn = st.number_input("Rate you earn on savings (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)

    st.divider()
    run_calc = st.button("🚀 Calculate Analysis", use_container_width=True, type="primary")

# --- 5. MAIN CONTENT AREA ---
if run_calc:
    # GLOBAL TARGET DEFINITION (Prevents NameError)
    target_total = e_basic * 6

    # 5.1 SUMMARY
    st.header("📝 Summary of Your Financial Profile")
    col_s1, col_s2, col_s3 = st.columns([1, 1, 1.5])

    with col_s1:
        st.markdown("**Monthly Income:**")
        st.markdown(f"## `{format_vnd(income)}`")
        st.markdown("**Total Essential Expenses:**")
        st.markdown(f"## `{format_vnd(e_basic)}`")

    with col_s2:
        essential_ratio = (e_basic / income) * 100 if income > 0 else 0
        st.markdown(f"**Budget Load:** `{essential_ratio:.1f}%` of income.")
        if income > 0:
            if essential_ratio > 70: st.warning("High essential expenses!")
            elif essential_ratio < 40: st.success("Healthy spending ratio.")

    with col_s3:
        df_exp = pd.DataFrame(list(exp_dict.items()), columns=['Category', 'Amount'])
        df_exp = df_exp[df_exp['Amount'] > 0]
        if not df_exp.empty:
            fig_pie = px.pie(df_exp, values='Amount', names='Category', title="Expense Breakdown", hole=0.4)
            fig_pie.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

    # 5.2 ROADMAP
    st.divider()
    st.header("🎯 Strategic Wealth Roadmap & Interpretation")

    if e_basic > 0:
        monthly_interest = (rate_earn / 100) / 12
        max_limit = 120
        months_display = 24
        
        # Determine projection length
        if monthly_savings > 0:
            temp_calc = current_bal
            for m in range(max_limit + 1):
                if temp_calc >= target_total:
                    months_display = max(24, m)
                    break
                temp_calc = (temp_calc + monthly_savings) * (1 + monthly_interest)

        # 5-State Thresholds (increments of 25% of the 6-month goal)
        s2_t = target_total * 0.25
        s3_t = target_total * 0.50
        s4_t = target_total * 0.75
        s5_t = target_total
        
        history_bal = []
        current_temp = current_bal
        reached = { "S2": None, "S3": None, "S4": None, "S5": None }
        
        for m in range(months_display + 1):
            color_status = "Secure" if current_temp >= s5_t else "Inadequate"
            history_bal.append({"Month": m, "Balance": current_temp, "Status": color_status})
            
            # Identify month of crossing for each state transition
            if reached["S2"] is None and current_temp >= s2_t: reached["S2"] = m
            if reached["S3"] is None and current_temp >= s3_t: reached["S3"] = m
            if reached["S4"] is None and current_temp >= s4_t: reached["S4"] = m
            if reached["S5"] is None and current_temp >= s5_t: reached["S5"] = m
            
            # Accumulate for next month
            current_temp = (current_temp + monthly_savings) * (1 + monthly_interest)
        
        df_growth = pd.DataFrame(history_bal)
        col_plot, col_coach = st.columns([2, 1])
        
        with col_plot:
            fig_growth = px.bar(df_growth, x="Month", y="Balance", title=f"Accumulation Projection ({months_display} Months)", color="Status", color_discrete_map={"Inadequate": "#EF553B", "Secure": "#00CC96"})
            # Add all state threshold lines
            for t, label in zip([s2_t, s3_t, s4_t, s5_t], ["S2 (25%)", "S3 (50%)", "S4 (75%)", "S5 (Goal)"]):
                fig_growth.add_hline(y=t, line_dash="dash", line_color="black", annotation_text=label)
            st.plotly_chart(fig_growth, use_container_width=True)

        with col_coach:
            st.subheader("🤝Interpretation")
            if reached["S2"] is not None: st.write(f"🔸 **S2 (Emerging):** Month {reached['S2']}")
            if reached["S3"] is not None: st.write(f"🔸 **S3 (Resilient):** Month {reached['S3']}")
            if reached["S4"] is not None: st.write(f"🔸 **S4 (Secure):** Month {reached['S4']}")
            
            if reached["S5"] is not None:
                st.success(f"🚀 **S5 (Optimal):** Reached at Month {reached['S5']}")
                st.info("**Advice:** Goal achieved. Maintain your discipline and start investing surplus.")
            elif monthly_savings > 0:
                gap = s5_t - history_bal[-1]["Balance"]
                st.warning(f"⚠️ **S5 Gap:** You are still **{format_vnd(gap)}** short of the goal by Month 24.")
            else:
                st.error("🛑 **Growth Stalled:** Zero savings detected. State S5 is currently unreachable.")

    # 5.3 MARKOV LOGIC
    st.divider()
    st.subheader("🎲 Financial State Probability (Markov Model)")
    
    lambda_shock = 0.073
    step_val = target_total / 4
    p_up = min(0.85, monthly_savings / step_val) if step_val > 0 else 0
    
    # 5x5 Transition Matrix P
    P = np.zeros((5, 5))
    for i in range(5):
        if i > 0: P[i, i-1] = lambda_shock
        if i < 4: P[i, i+1] = p_up
        row_sum = P[i, :].sum()
        if row_sum > 1: P[i, :] /= row_sum
        P[i, i] = 1 - P[i, :].sum()

    col_m1, col_m2 = st.columns([1, 1])
    with col_m1:
        st.write("**Transition Matrix (P):**")
        st.table(pd.DataFrame(P, index=["S1", "S2", "S3", "S4", "S5"], columns=["S1", "S2", "S3", "S4", "S5"]).style.format("{:.1%}"))
        
        evals, evecs = np.linalg.eig(P.T)
        steady = evecs[:, np.isclose(evals, 1.0)].real
        steady = (steady / steady.sum()).flatten()
        st.metric("Long-run Safety (S4+S5 Probability)", f"{(steady[3] + steady[4]):.1%}")

    with col_m2:
        ratio = current_bal / target_total if target_total > 0 else 0
        curr_idx = 0 if ratio < 0.25 else 1 if ratio < 0.5 else 2 if ratio < 0.75 else 3 if ratio < 1 else 4
        v = np.zeros(5); v[curr_idx] = 1
        history_m = [v]
        for _ in range(24):
            v = np.dot(v, P)
            history_m.append(v)
        st.plotly_chart(px.area(pd.DataFrame(history_m, columns=["S1", "S2", "S3", "S4", "S5"]), title="24-Month Probability Forecast"), use_container_width=True)

else:
    st.info("👈 Please enter your financial details in the sidebar and click 'Calculate Analysis' to begin.")