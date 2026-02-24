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
st.set_page_config(page_title="Emergency Fund Calculator", layout="wide")

st.title("🛡️ Smart Emergency Fund Calculator")

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
    # ANALYSIS BUTTON
    run_calc = st.button("🚀 Calculate Analysis", use_container_width=True, type="primary")

# --- 5. MAIN CONTENT AREA ---
if run_calc:
    # 5.1 SUMMARIZE USER DATA
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

    # 5.2 STRATEGIC ROADMAP (WITH DYNAMIC MONTHS)
    st.divider()
    st.header("🎯 Strategic Wealth Roadmap & Interpretation")

    if e_basic > 0:
        s2_target = e_basic * 3
        s3_target = e_basic * 6
        monthly_interest = (rate_earn / 100) / 12
        
        # Logic to calculate months needed for S3 (cap at 120)
        max_limit = 120
        months_display = 24
        if monthly_savings > 0:
            temp_calc = current_bal
            for m in range(max_limit + 1):
                if temp_calc >= s3_target:
                    months_display = max(24, m)
                    break
                temp_calc = (temp_calc + monthly_savings) * (1 + monthly_interest)
                if m == max_limit: months_display = 120

        history_bal = []
        current_temp = current_bal
        reached_s2 = None
        reached_s3 = None
        
        for m in range(months_display + 1):
            color_status = "Secure" if current_temp >= s2_target else "Inadequate"
            history_bal.append({"Month": m, "Balance": current_temp, "Status": color_status})
            
            if reached_s2 is None and current_temp >= s2_target: reached_s2 = m
            if reached_s3 is None and current_temp >= s3_target: reached_s3 = m
            current_temp = (current_temp + monthly_savings) * (1 + monthly_interest)
        
        df_growth = pd.DataFrame(history_bal)

        col_plot, col_coach = st.columns([2, 1])
        
        with col_plot:
            fig_growth = px.bar(df_growth, x="Month", y="Balance", 
                                title=f"Accumulation Projection ({months_display} Months)",
                                color="Status", 
                                color_discrete_map={"Inadequate": "#EF553B", "Secure": "#00CC96"})
            
            fig_growth.add_hline(y=s2_target, line_dash="dash", line_color="black", annotation_text="S2 Target")
            fig_growth.add_hline(y=s3_target, line_dash="dash", line_color="black", annotation_text="S3 Target")
            fig_growth.update_layout(yaxis=dict(tickformat=',.0f'), showlegend=True)
            st.plotly_chart(fig_growth, use_container_width=True)

        with col_coach:
            st.subheader("Interpretation")
            st.write(f"**Current Balance:** {format_vnd(current_bal)}")
            st.write(f"**Monthly Savings:** {format_vnd(monthly_savings)}")
            
            st.markdown("---")
            if monthly_savings > 0:
                if reached_s2 is not None:
                    st.success(f"✅ **S2 (Adequate):** Reached at **Month {reached_s2}**")
                else:
                    st.warning(f"⚠️ **S2 (Adequate):** Not reachable within {months_display} months.")
                
                if reached_s3 is not None:
                    st.success(f"🚀 **S3 (Strong):** Reached at **Month {reached_s3}**")
                else:
                    gap_s3 = max(0, s3_target - df_growth['Balance'].iloc[-1])
                    st.info(f"ℹ️ **S3 Gap:** You are still `{format_vnd(gap_s3)}` short after this period.")
            else:
                st.error("🛑 **Growth Stalled:** Zero savings means the roadmap is frozen.")

    # 5.3 MARKOV LOGIC
    st.divider()
    st.subheader("🎲 Financial State Probability (Markov Model)")
    p_up = min(0.9, (monthly_savings / e_basic) * 0.35 + (rate_earn/100)) if e_basic > 0 else 0.02
    states = ["S₀: Depleted", "S₁: Inadequate", "S₂: Adequate", "S₃: Strong"]

    matrix_data = np.array([
        [1-p_up, p_up, 0, 0], [0.05, 1-0.05-p_up, p_up, 0],
        [0, 0.05, 1-0.05-p_up, p_up], [0, 0, 0.05, 1-0.05]
    ])
    matrix_data = matrix_data / matrix_data.sum(axis=1)[:, None]

    col_m1, col_m2 = st.columns([1, 1])
    with col_m1:
        st.table(pd.DataFrame(matrix_data, index=states, columns=states).style.format("{:.1%}"))

    with col_m2:
        ratio = current_bal / e_basic if e_basic > 0 else 0
        curr_idx = 0 if ratio < 1 else 1 if ratio < 3 else 2 if ratio < 6 else 3
        v = np.zeros(4); v[curr_idx] = 1
        history = [v]
        for _ in range(24):
            v = np.dot(v, matrix_data)
            history.append(v)
        st.plotly_chart(px.area(pd.DataFrame(history, columns=states)), use_container_width=True)

    st.caption("This calculator is for general education purposes only ")

else:
    # DISPLAY WHEN NO BUTTON IS PRESSED
    st.info("Please enter your financial details in the sidebar and click 'Calculate Analysis' to begin.")