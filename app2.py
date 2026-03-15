import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def fmt_money(amount):
    if amount is None or amount == 0:
        return "0 ₫"
    return "{:,.0f} ₫".format(amount).replace(",", ".")

TIER_WIDTHS = [1, 2, 3, 3]

STATE_LABELS = ["🔴 Distressed", "🟠 Fragile", "🔵 Stable", "🟢 Flourishing"]
STATE_KEYS   = ["Distressed", "Fragile", "Stable", "Flourishing"]

COLOR_MAP = {
    "🔴 Distressed":  "#EF553B",
    "🟠 Fragile":     "#FFA15A",
    "🔵 Stable":      "#636EFA",
    "🟢 Flourishing": "#00CC96",
}

STATE_DESCRIPTIONS = {
    "🔴 Distressed":  "Your emergency fund covers less than 1 month of expenses. A single unexpected event could put you in serious financial trouble.",
    "🟠 Fragile":     "Your fund covers 1–3 months of expenses. You have some cushion, but a major event like job loss could still wipe it out quickly.",
    "🔵 Stable":      "Your fund covers 3–6 months of expenses. You're in a good position to handle most unexpected events.",
    "🟢 Flourishing": "Your fund covers 6+ months of expenses. You're well protected. Even a prolonged income disruption won't immediately threaten your finances.",
}

def classify_state(fund_balance, monthly_expenditure):
    if monthly_expenditure <= 0:
        return STATE_LABELS[0], 0
    coverage = fund_balance / monthly_expenditure
    if coverage < 1:   return STATE_LABELS[0], 0
    elif coverage < 3: return STATE_LABELS[1], 1
    elif coverage < 6: return STATE_LABELS[2], 2
    else:              return STATE_LABELS[3], 3

def build_P(monthly_surplus, monthly_expenditure):
    """
    Build transition matrix with:
    - Upward pressure from positive surplus (savings driving fund growth)
    - Downward pressure from negative surplus (deficit draining fund)
    - A small baseline slip probability even when saving, because in real
      life unexpected months happen (illness, car repair, etc.) that can
      temporarily set back even a disciplined saver.
      Baseline slip scales inversely with surplus size — the more you save
      above your expenses, the smaller the slip risk.
    """
    n = 4
    P = np.zeros((n, n))
    if monthly_expenditure <= 0:
        np.fill_diagonal(P, 1.0)
        return P

    # Baseline downward slip: even with positive surplus there is always
    # some chance of a bad month. Ranges from ~5% (strong saver) to ~15%
    # (barely positive). Scales with how thin the surplus margin is.
    if monthly_surplus > 0:
        margin_ratio = min(monthly_surplus / monthly_expenditure, 1.0)
        # Thin margin → higher slip, thick margin → lower slip
        baseline_slip = 0.15 * (1.0 - margin_ratio) + 0.05 * margin_ratio
    elif monthly_surplus < 0:
        baseline_slip = 0.0   # handled by explicit downward rate below
    else:
        baseline_slip = 0.10  # break-even: moderate slip risk

    for i in range(n):
        tier_amount = monthly_expenditure * TIER_WIDTHS[i]
        rate = min(abs(monthly_surplus) / tier_amount, 0.40) if tier_amount > 0 else 0.0

        if monthly_surplus > 0:
            p_up   = rate if i < n - 1 else 0.0
            p_down = baseline_slip if i > 0 else 0.0
        elif monthly_surplus < 0:
            p_up   = 0.0
            p_down = rate if i > 0 else 0.0
        else:
            p_up   = 0.0
            p_down = baseline_slip if i > 0 else 0.0

        p_stay = max(1.0 - p_up - p_down, 0.0)

        P[i, i]     = p_stay
        if i < n-1: P[i, i+1] = p_up
        if i > 0:   P[i, i-1] = p_down

        row_sum = P[i].sum()
        if row_sum > 0: P[i] /= row_sum

    return P

def steady_state(P):
    pi = np.ones(4) / 4.0
    for _ in range(200):
        pi = pi @ P
    pi /= pi.sum()
    return pi

def mfpt_matrix(P, pi):
    n = P.shape[0]
    W = np.tile(pi, (n, 1))
    try:
        Z = np.linalg.inv(np.eye(n) - P + W)
    except np.linalg.LinAlgError:
        return np.full((n, n), np.inf)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (Z[j,j] - Z[i,j]) / pi[j] if pi[j] > 1e-10 else np.inf
    return M

def compute_pci(coverage, event_duration):
    total = coverage + event_duration
    return coverage / total if total > 0 else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Emergency Fund Planner", layout="wide", page_icon="🛡️")

st.title("🛡️ Emergency Fund Planner")
st.markdown("*Find out where you stand financially and exactly what you need to do to stay safe.*")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Tell us about your finances")
    st.caption("All amounts are monthly.")

    st.subheader("🏠 Essential spending")
    housing   = st.number_input("Rent / Mortgage",        min_value=0, step=500_000, value=0)
    food      = st.number_input("Food & Groceries",       min_value=0, step=100_000, value=0)
    utilities = st.number_input("Utilities (electric, water, internet)", min_value=0, step=100_000, value=0)
    transport = st.number_input("Transport",              min_value=0, step=100_000, value=0)
    health    = st.number_input("Health & Insurance",     min_value=0, step=100_000, value=0)
    education = st.number_input("Education",              min_value=0, step=100_000, value=0)
    debt      = st.number_input("Loan / Debt repayments", min_value=0, step=100_000, value=0)
    eb = housing + food + utilities + transport + health + education + debt

    st.subheader("🎭 Regular non-essential spending")
    entertainment = st.number_input("Entertainment & dining out", min_value=0, step=100_000, value=0)
    household     = st.number_input("Household goods",            min_value=0, step=100_000, value=0)
    er = entertainment + household

    st.subheader("✈️ Lifestyle spending")
    holidays = st.number_input("Travel & holidays",       min_value=0, step=100_000, value=0)
    luxury   = st.number_input("Luxury items / services", min_value=0, step=100_000, value=0)
    el = holidays + luxury

    total_expenditure = eb + er + el

    st.divider()
    st.subheader("💵 Income & savings")
    income          = st.number_input("Monthly take-home income", min_value=0, step=500_000, value=0)
    current_balance = st.number_input("Emergency fund you have now", min_value=0, step=500_000, value=0)
    monthly_savings = st.number_input("How much you can save each month", min_value=0, step=100_000, value=0)
    rate_earn       = st.number_input("Interest rate on savings (% per year)", min_value=0.0, max_value=20.0, step=0.1, value=0.0)

    st.divider()
    st.subheader("⚡ What situation are you preparing for?")
    event_duration = st.slider(
        "If you lost your income, how many months do you want to be covered?",
        min_value=1, max_value=24, value=6
    )

    st.divider()
    run = st.button("📊 Show My Results", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# LANDING
# ─────────────────────────────────────────────────────────────────────────────
if not run:
    st.markdown("""
    ### How it works

    1. **Enter your monthly expenses and income** in the sidebar
    2. **Tell us how much you've already saved** for emergencies
    3. **Set your savings goal** — how many months of expenses you want covered
    4. Hit **Show My Results** to get your personal financial health report

    ---
    **What you'll find out:**
    - 🔴 Whether your current situation is dangerous or safe
    - ⏱️ How long it will take you to reach a safe level at your current savings pace
    - 💡 Exactly how much you should be saving each month
    - 📈 How your situation is likely to look 2 years from now
    """)
    st.stop()

if total_expenditure == 0:
    st.warning("⚠️ Please enter at least one spending item in the sidebar.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────
monthly_interest  = (rate_earn / 100.0) / 12.0
monthly_surplus   = income - total_expenditure
coverage_months   = current_balance / total_expenditure if total_expenditure > 0 else 0.0
curr_label, curr_idx = classify_state(current_balance, total_expenditure)
pci               = compute_pci(coverage_months, event_duration)

monthly_surplus   = income - total_expenditure
effective_surplus = monthly_surplus + monthly_savings  # savings actively grow the fund

P      = build_P(effective_surplus, total_expenditure)
pi     = steady_state(P)
M_mfpt = mfpt_matrix(P, pi)

p_safe     = float(pi[2] + pi[3])
p_at_risk  = float(pi[0] + pi[1])

deficit = total_expenditure - income
min_save_needed = max(deficit + 1, 0) if deficit > 0 else 0

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — YOUR SITUATION RIGHT NOW
# ─────────────────────────────────────────────────────────────────────────────
st.header("📍 Your Situation Right Now")

col_status, col_detail = st.columns([1, 2])

with col_status:
    # Big state card
    state_bg = {"Distressed": "#FFEEEE", "Fragile": "#FFF3E0", "Stable": "#EEF0FF", "Flourishing": "#E8FFF5"}
    state_border = {"Distressed": "#EF553B", "Fragile": "#FFA15A", "Stable": "#636EFA", "Flourishing": "#00CC96"}
    raw_key = STATE_KEYS[curr_idx]
    bg   = state_bg[raw_key]
    bord = state_border[raw_key]

    st.markdown(f"""
    <div style="background:{bg}; border-left: 6px solid {bord};
                padding: 20px; border-radius: 8px; margin-bottom:12px;">
        <div style="font-size:2rem; font-weight:700; color:{bord};">{curr_label}</div>
        <div style="font-size:1rem; margin-top:8px; color:#333;">
            {STATE_DESCRIPTIONS[curr_label]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.metric("Your emergency fund covers", f"{coverage_months:.1f} months of expenses")

with col_detail:
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Monthly income",      fmt_money(income))
        st.metric("Monthly spending",    fmt_money(total_expenditure))
        surplus_label = "Operating surplus / month" if monthly_surplus >= 0 else "Shortfall each month"
        st.metric(surplus_label,
                  fmt_money(abs(monthly_surplus)),
                  delta="✅ Positive" if monthly_surplus > 0 else ("⚠️ Break-even" if monthly_surplus == 0 else "🚨 Negative"),
                  delta_color="normal" if monthly_surplus > 0 else "inverse")
    with c2:
        st.metric("Emergency fund balance", fmt_money(current_balance))
        st.metric("Monthly savings set aside", fmt_money(monthly_savings))
        st.metric("Effective monthly fund growth", fmt_money(effective_surplus))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CAN YOU SURVIVE YOUR SCENARIO?
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header(f"⚡ If You Lost Your Income for {event_duration} Months...")

pci_col, gauge_col = st.columns([2, 1])

with pci_col:
    if pci >= 0.5:
        st.success(f"""
        ✅ **You're covered.**

        Your current emergency fund of **{fmt_money(current_balance)}** 
        can support **{coverage_months:.1f} months** of expenses — 
        enough to get through a **{event_duration}-month** disruption.

        You have a **{coverage_months - event_duration:.1f}-month buffer** beyond your target.
        """)
    else:
        shortfall_months = event_duration - coverage_months
        shortfall_amount = shortfall_months * total_expenditure
        st.error(f"""
        🚨 **Your fund is not enough.**

        You can only cover **{coverage_months:.1f} months** but need **{event_duration} months**.
        
        You are short by **{shortfall_months:.1f} months** of expenses — 
        that's about **{fmt_money(shortfall_amount)}** that you do not currently have.

        If something happened today, you would run out of money after {coverage_months:.1f} months 
        and would need to borrow, sell assets, or rely on others.
        """)

with gauge_col:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(pci, 2),
        number={"suffix": "", "font": {"size": 36}},
        title={"text": "Readiness Score (PCI)", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 1], "tickvals": [0, 0.25, 0.5, 0.75, 1],
                     "ticktext": ["0", "0.25", "0.50", "0.75", "1.0"]},
            "bar":  {"color": "#00CC96" if pci >= 0.5 else "#EF553B"},
            "steps": [
                {"range": [0, 0.5],  "color": "#FFEEEE"},
                {"range": [0.5, 1.0], "color": "#EEFFEE"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75, "value": 0.5
            }
        }
    ))
    fig_gauge.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Dynamic verdict based on actual PCI value
    if pci >= 0.5:
        st.success(f"✅ **Covered** — PCI {pci:.2f} ≥ 0.50")
    elif pci >= 0.25:
        st.warning(f"⚠️ **Partially covered** — PCI {pci:.2f} < 0.50")
    else:
        st.error(f"🚨 **Not covered** — PCI {pci:.2f} far below 0.50")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — HOW LONG TO GET TO SAFETY?
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("⏱️ How Long Until You're Safe?")

time_cols = st.columns(4)
state_targets = [
    ("🔴 Less than 1 month covered",  0, "Distressed"),
    ("🟠 1–3 months covered",          1, "Fragile"),
    ("🔵 3–6 months covered",          2, "Stable"),
    ("🟢 6+ months covered",           3, "Flourishing"),
]

for col, (label, idx, key) in zip(time_cols, state_targets):
    mfpt_val = M_mfpt[curr_idx, idx]
    bord = {"Distressed":"#EF553B","Fragile":"#FFA15A","Stable":"#636EFA","Flourishing":"#00CC96"}[key]
    with col:
        if idx == curr_idx:
            st.markdown(f"""
            <div style="border:2px solid {bord}; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.85rem; color:#555;">{label}</div>
                <div style="font-size:1.3rem; font-weight:700; color:{bord}; margin-top:6px;">📍 You are here</div>
            </div>
            """, unsafe_allow_html=True)
        elif idx < curr_idx:
            st.markdown(f"""
            <div style="border:2px solid #ccc; border-radius:8px; padding:12px; text-align:center; opacity:0.5;">
                <div style="font-size:0.85rem; color:#555;">{label}</div>
                <div style="font-size:1.3rem; font-weight:700; color:#aaa; margin-top:6px;">✅ Already passed</div>
            </div>
            """, unsafe_allow_html=True)
        elif np.isinf(mfpt_val) or mfpt_val > 999:
            st.markdown(f"""
            <div style="border:2px solid {bord}; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.85rem; color:#555;">{label}</div>
                <div style="font-size:1.3rem; font-weight:700; color:#EF553B; margin-top:6px;">⚠️ Not reachable</div>
                <div style="font-size:0.8rem; color:#888; margin-top:4px;">Save more each month</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            years = mfpt_val / 12
            time_str = f"{mfpt_val:.0f} months" if years < 1 else f"{years:.1f} years"
            st.markdown(f"""
            <div style="border:2px solid {bord}; border-radius:8px; padding:12px; text-align:center;">
                <div style="font-size:0.85rem; color:#555;">{label}</div>
                <div style="font-size:1.6rem; font-weight:700; color:{bord}; margin-top:6px;">{time_str}</div>
                <div style="font-size:0.8rem; color:#888; margin-top:4px;">at your current savings pace</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — WHAT YOUR FUTURE LOOKS LIKE
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("📈 What Your Next 24 Months Look Like")

forecast_col, info_col = st.columns([2, 1])

with forecast_col:
    v = np.zeros(4)
    v[curr_idx] = 1.0
    history_m = [v.copy()]
    for _ in range(24):
        v = v @ P
        history_m.append(v.copy())

    df_forecast = pd.DataFrame(history_m, columns=STATE_LABELS)
    df_forecast.index.name = "Month"
    df_forecast = df_forecast.reset_index()

    df_lines = pd.DataFrame({
        "Month": df_forecast["Month"],
        "🟢 Chance of being safe (3+ months covered)": df_forecast["🔵 Stable"] + df_forecast["🟢 Flourishing"],
        "🔴 Chance of being at risk (less than 3 months)": df_forecast["🔴 Distressed"] + df_forecast["🟠 Fragile"],
    })

    fig_lines = px.line(
        df_lines, x="Month",
        y=["🟢 Chance of being safe (3+ months covered)",
           "🔴 Chance of being at risk (less than 3 months)"],
        color_discrete_map={
            "🟢 Chance of being safe (3+ months covered)": "#00CC96",
            "🔴 Chance of being at risk (less than 3 months)": "#EF553B",
        },
        title="Over the next 24 months — how likely are you to be safe vs at risk?",
    )
    fig_lines.add_hline(
        y=0.5, line_dash="dot", line_color="#aaa",
        annotation_text="50/50", annotation_position="right"
    )
    fig_lines.update_layout(
        yaxis_tickformat=".0%",
        yaxis_title="Probability",
        xaxis_title="Month from now",
        legend_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=-0.40, xanchor="left", x=0),
        height=340,
        margin=dict(t=40, b=70)
    )
    st.plotly_chart(fig_lines, use_container_width=True)
    st.caption(
        "**Safe** = your fund covers 3 or more months of expenses. "
        "**At risk** = your fund covers less than 3 months. "
        "The two lines always add up to 100%."
    )

month_6_safe  = float(df_lines["🟢 Chance of being safe (3+ months covered)"].iloc[6])
month_24_safe = float(df_lines["🟢 Chance of being safe (3+ months covered)"].iloc[24])

with info_col:
    st.markdown("**What this means for you:**")
    st.markdown("<br>", unsafe_allow_html=True)

    m6_icon  = "🟢" if month_6_safe  >= 0.6 else "🔴"
    m24_icon = "🟢" if month_24_safe >= 0.6 else "🔴"
    st.markdown(f"{m6_icon} **In 6 months:** {month_6_safe:.0%} chance of being safe")
    st.markdown(f"{m24_icon} **In 24 months:** {month_24_safe:.0%} chance of being safe")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "**What does 'safe' mean here?**\n\n"
        "It means your emergency fund covers at least 3 months of your expenses — "
        "enough to survive a job loss, a medical bill, or another major unexpected event "
        "without having to borrow money."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if month_24_safe >= 0.70:
        st.success("✅ You are on a strong path. Keep saving at this pace.")
    elif month_24_safe >= 0.50:
        st.warning("⚠️ You have a moderate chance of being safe in 2 years. Saving a bit more each month would make a big difference.")
    else:
        st.error("🚨 At your current pace, you are more likely to still be at risk in 2 years. See the recommendation below for what to change.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — WHAT SHOULD YOU DO?
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("💡 What Should You Do?")

# Build savings scenarios
scenario_savings = []
if min_save_needed > 0:
    scenario_savings = [0, min_save_needed, int(min_save_needed * 1.5), int(min_save_needed * 2)]
else:
    base = max(monthly_savings, int(total_expenditure * 0.05))
    scenario_savings = [0, base, base * 2, base * 3]

if monthly_savings not in scenario_savings:
    scenario_savings.append(monthly_savings)
scenario_savings = sorted(set([max(0, int(x)) for x in scenario_savings]))

rows = []
for s_test in scenario_savings:
    effective_test = monthly_surplus + s_test
    P_test  = build_P(effective_test, total_expenditure)
    pi_test = steady_state(P_test)
    M_test  = mfpt_matrix(P_test, pi_test)
    mfpt_s3 = M_test[curr_idx, 2]
    mfpt_s4 = M_test[curr_idx, 3]
    p_safe_test = float(pi_test[2] + pi_test[3])

    if effective_test > 0:
        drift = "📈 Growing"
    elif effective_test == 0:
        drift = "➡️ Flat"
    else:
        drift = "📉 Shrinking"

    rows.append({
        "If you save each month":    fmt_money(s_test),
        "Your fund will be":         drift,
        "Months to reach Stable 🔵": f"{mfpt_s3:.0f} months" if not np.isinf(mfpt_s3) and curr_idx < 2 else ("Already there ✅" if curr_idx >= 2 else "Not reachable ⚠️"),
        "Months to reach Flourishing 🟢": f"{mfpt_s4:.0f} months" if not np.isinf(mfpt_s4) and curr_idx < 3 else ("Already there ✅" if curr_idx >= 3 else "Not reachable ⚠️"),
        "Long-run chance of being safe": f"{p_safe_test:.0%}",
    })

df_scenarios = pd.DataFrame(rows)

# Highlight the user's current savings row
def highlight_current(row):
    if row["If you save each month"] == fmt_money(monthly_savings):
        return ["background-color: #fffde7"] * len(row)
    return [""] * len(row)

st.markdown("**Here's how your outcome changes based on how much you save:**")
st.caption(f"Your current savings is highlighted — {fmt_money(monthly_savings)}/month")
st.dataframe(df_scenarios.style.apply(highlight_current, axis=1), use_container_width=True, hide_index=True)

# Plain-English recommendation
st.markdown("### 🎯 Our Recommendation")

if monthly_surplus > 0 and curr_idx >= 2:
    st.success(f"""
    **You're in good shape.** Your fund is growing and you're already in a stable or flourishing state.

    Keep saving **{fmt_money(monthly_savings)}/month** and you'll maintain this strong position.
    
    If you want extra security, consider putting any extra money into a higher-yield savings account 
    to make your fund work harder for you.
    """)
elif monthly_surplus > 0 and curr_idx < 2:
    months_to_stable = M_mfpt[curr_idx, 2]
    st.info(f"""
    **You're on the right track**, but not safe yet.
    
    At your current savings pace of **{fmt_money(monthly_savings)}/month**, 
    you should reach a stable safety level in about **{months_to_stable:.0f} months**.
    
    If you can save a little more each month, you'll get there faster — 
    see the table above to find the right number for you.
    """)
elif monthly_surplus == 0:
    st.warning(f"""
    **Your income and spending are balanced — but you're not building any cushion.**
    
    Any unexpected expense right now would hurt you because your fund isn't growing.
    
    Try to find **{fmt_money(int(total_expenditure * 0.05))}–{fmt_money(int(total_expenditure * 0.10))}/month** 
    to set aside. Even a small amount will start moving you toward safety.
    """)
else:
    st.error(f"""
    **You're spending more than you earn each month.** This means your emergency fund 
    is slowly being used up, leaving you more vulnerable over time.
    
    **The most important step right now:** reduce monthly spending by at least 
    **{fmt_money(abs(monthly_surplus))}**, or increase your income by that amount.
    
    Once your income covers your expenses, any additional savings will start rebuilding your fund.
    The table above shows what happens when you do.
    """)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — FUND GROWTH PROJECTION
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("📊 Fund Growth Projection")

if monthly_savings > 0 or current_balance > 0:
    proj_months = 36
    balance_history = []
    bal = float(current_balance)

    for m in range(proj_months + 1):
        label, _ = classify_state(bal, total_expenditure)
        coverage = bal / total_expenditure if total_expenditure > 0 else 0
        balance_history.append({
            "Month": m,
            "Balance": bal,
            "Coverage (months)": round(coverage, 2),
            "State": label,
        })
        monthly_addition = monthly_surplus + monthly_savings
        bal = (bal + monthly_addition) * (1 + monthly_interest)

    df_proj = pd.DataFrame(balance_history)

    fig_proj = px.bar(
        df_proj, x="Month", y="Balance",
        color="State", color_discrete_map=COLOR_MAP,
        title="Your emergency fund balance over the next 3 years",
        labels={"Balance": "Fund Balance", "Month": "Month from now"}
    )

    # Add coverage lines
    for months_cov, label_text, color in [
        (1, "1 month covered",  "#FFA15A"),
        (3, "3 months covered", "#636EFA"),
        (6, "6 months covered", "#00CC96"),
    ]:
        fig_proj.add_hline(
            y=months_cov * total_expenditure,
            line_dash="dash", line_color=color,
            annotation_text=label_text,
            annotation_position="right"
        )

    fig_proj.update_layout(
        yaxis_title="Fund Balance",
        xaxis_title="Month from now",
        height=350,
        legend_title="Financial State"
    )
    st.plotly_chart(fig_proj, use_container_width=True)

    # Milestone callouts
    milestones = {1: None, 3: None, 6: None}
    for row in balance_history:
        cov = row["Coverage (months)"]
        for target in milestones:
            if milestones[target] is None and cov >= target:
                milestones[target] = row["Month"]

    m1, m2, m3 = st.columns(3)
    for col, (target, month_reached) in zip([m1, m2, m3], milestones.items()):
        with col:
            if month_reached is not None:
                if month_reached == 0:
                    col.metric(f"{target}-month coverage", "Already there ✅")
                else:
                    col.metric(f"{target}-month coverage", f"Month {month_reached}", f"≈ {month_reached/12:.1f} years")
            else:
                col.metric(f"{target}-month coverage", "Not reached in 3 years", delta="Save more", delta_color="inverse")
else:
    st.info("Enter a savings amount or current balance to see your fund growth projection.")