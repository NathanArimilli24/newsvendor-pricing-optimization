"""
Optimization Project 3 - Non-Linear Programming (Newsvendor Model)
==================================================================

Pricing and production decisions for a publishing company, solved as a sequence
of optimization models in Gurobi.

Pipeline:
    Part 1  Fit a linear regression of demand on price.
    Part 2  Generate demand scenarios at p=1 from the regression residuals.
    Part 3  Solve the optimal production quantity at p=1 (a linear program).
    Part 4  Make price a decision variable and solve the joint price-quantity
            problem (a concave quadratic program).
    Part 6  One bootstrap resample: refit and re-solve Part 4.
    Part 7  Repeat the bootstrap many times; summarize the distributions of the
            optimal price, quantity, and profit.
    Part 8  Build the boss's baseline (the standard lost-sales newsvendor model)
            and compare it to our model on a common, real-world profit metric.

Cost structure (the "true" operating environment):
    c = 0.50  regular per-unit production cost
    g = 0.75  rush-order cost per unit when production falls short of demand
    t = 0.15  disposal cost per unit of unsold inventory

With rush orders available, all demand is met, so revenue is always p * D.
Running this file regenerates every figure and the results table in ../output/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.linear_model import LinearRegression
import gurobipy as gp
from gurobipy import GRB

# ----------------------------------------------------------------------------
# Paths (relative to this file, so the script runs from anywhere)
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "price_demand_data.csv"
OUT = HERE.parent / "output"
OUT.mkdir(exist_ok=True)

# Cost parameters
c = 0.50   # regular production cost per unit
g = 0.75   # rush-order cost per unit (g > c)
t = 0.15   # disposal cost per unit of excess

# ============================================================================
# Part 1: Linear regression of demand on price
# ============================================================================
df = pd.read_csv(DATA)
X = df[["price"]].values
y = df["demand"].values

reg = LinearRegression().fit(X, y)
beta_0 = reg.intercept_     # intercept
beta_1 = reg.coef_[0]       # slope (coefficient on price)
residuals = y - reg.predict(X)   # epsilon_i = D_i - (beta_0 + beta_1 * p_i)

print("=" * 60)
print("Part 1: Linear Regression")
print("=" * 60)
print(f"Observations           : {len(df)}")
print(f"beta_0 (intercept)     : {beta_0:.4f}")
print(f"beta_1 (slope)         : {beta_1:.4f}")
print(f"Demand = {beta_0:.4f} + ({beta_1:.4f}) * Price")
print(f"Residual std           : {residuals.std():.4f}")

# Figure: regression fit + residual distribution
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(df["price"], df["demand"], alpha=0.6, label="Data")
grid = np.linspace(df["price"].min(), df["price"].max(), 100)
axes[0].plot(grid, beta_0 + beta_1 * grid, "r-", lw=2, label="Fitted line")
axes[0].set(xlabel="Price", ylabel="Demand", title="Price vs Demand")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].hist(residuals, bins=20, edgecolor="black", alpha=0.7)
axes[1].set(xlabel="Residual", ylabel="Frequency", title="Distribution of Residuals")
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "part1_regression.png", dpi=150)
plt.close(fig)

# ============================================================================
# Part 2: Generate demand scenarios at p = 1
# ============================================================================
# The residuals are the source of randomness. At a fixed price we get one
# demand scenario per residual: D_i(p) = beta_0 + beta_1 * p + epsilon_i.
p = 1.0
demand_at_p1 = beta_0 + beta_1 * p + residuals
n = len(demand_at_p1)

print("\n" + "=" * 60)
print("Part 2: Demand scenarios at p = 1")
print("=" * 60)
print(f"Base demand (beta_0 + beta_1*p): {beta_0 + beta_1 * p:.4f}")
print(f"Mean demand : {demand_at_p1.mean():.4f}")
print(f"Std demand  : {demand_at_p1.std():.4f}")
print(f"Range       : [{demand_at_p1.min():.2f}, {demand_at_p1.max():.2f}]")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(demand_at_p1, bins=20, edgecolor="black", alpha=0.7)
ax.axvline(demand_at_p1.mean(), color="r", ls="--", lw=2,
           label=f"Mean = {demand_at_p1.mean():.2f}")
ax.set(xlabel="Demand", ylabel="Frequency", title="Demand distribution at price = $1")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "part2_demand_p1.png", dpi=150)
plt.close(fig)


# ============================================================================
# Part 3: Optimal quantity at p = 1 (Linear Program)
# ============================================================================
# For a fixed quantity q and scenario i:
#   profit_i = p*D_i - c*q - g*(D_i - q)^+ - t*(q - D_i)^+
# We linearize the two positive-part terms with shortage (s) and excess (e):
#   s_i >= D_i - q,  e_i >= q - D_i,  s_i, e_i >= 0
# Because s and e carry positive costs in a maximization, they settle at the
# tight values s_i = (D_i - q)^+ and e_i = (q - D_i)^+.
def solve_fixed_price_lp(D, price):
    m = gp.Model()
    m.Params.OutputFlag = 0
    q = m.addVar(lb=0.0, name="q")
    s = m.addMVar(len(D), lb=0.0, name="shortage")
    e = m.addMVar(len(D), lb=0.0, name="excess")
    for i in range(len(D)):
        m.addConstr(s[i] >= D[i] - q)
        m.addConstr(e[i] >= q - D[i])
    m.setObjective(
        (1 / len(D)) * gp.quicksum(price * D[i] - c * q - g * s[i] - t * e[i]
                                   for i in range(len(D))),
        GRB.MAXIMIZE,
    )
    m.optimize()
    return q.X, m.ObjVal, s.X, e.X


q3, profit3, s3, e3 = solve_fixed_price_lp(demand_at_p1, p)
print("\n" + "=" * 60)
print("Part 3: Optimal quantity at p = 1 (LP)")
print("=" * 60)
print(f"Optimal quantity q* : {q3:.4f}")
print(f"Expected profit     : ${profit3:.4f}")
print(f"Days short / over   : {(s3 > 1e-3).sum()} / {(e3 > 1e-3).sum()} of {n}")

# Value-add figure: expected profit as a function of q, with the optimum marked.
q_grid = np.linspace(demand_at_p1.min() - 50, demand_at_p1.max() + 50, 400)
profit_curve = [
    np.mean(p * demand_at_p1 - c * qq
            - g * np.maximum(demand_at_p1 - qq, 0)
            - t * np.maximum(qq - demand_at_p1, 0))
    for qq in q_grid
]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(q_grid, profit_curve, lw=2)
ax.axvline(q3, color="r", ls="--", label=f"q* = {q3:.1f}")
ax.scatter([q3], [profit3], color="r", zorder=5)
ax.set(xlabel="Production quantity q", ylabel="Expected profit ($)",
       title="Expected profit vs quantity at p = $1")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "part3_profit_vs_quantity.png", dpi=150)
plt.close(fig)


# ============================================================================
# Part 4: Joint price-quantity optimization (concave QP)
# ============================================================================
# Now price is a decision variable and demand depends on it:
#   D_i(p) = beta_0 + beta_1 * p + epsilon_i
# Revenue p*D_i(p) contains the term beta_1 * p^2. Because beta_1 < 0 this term
# is concave, so maximizing expected profit is a convex (concave-maximization)
# QP that Gurobi solves directly. The shortage/excess constraints stay linear.
# Price is bounded to the observed data range to avoid extrapolating the fit.
def solve_joint_qp(b0, b1, eps, p_min, p_max):
    m = gp.Model()
    m.Params.OutputFlag = 0
    nn = len(eps)
    p_var = m.addVar(lb=p_min, ub=p_max, name="price")
    q_var = m.addVar(lb=0.0, name="quantity")
    s = m.addMVar(nn, lb=0.0, name="shortage")
    e = m.addMVar(nn, lb=0.0, name="excess")
    profit = 0
    for i in range(nn):
        D_i = b0 + b1 * p_var + eps[i]
        m.addConstr(s[i] >= D_i - q_var)
        m.addConstr(e[i] >= q_var - D_i)
        profit += p_var * D_i - c * q_var - g * s[i] - t * e[i]
    m.setObjective(profit / nn, GRB.MAXIMIZE)
    m.optimize()
    return p_var.X, q_var.X, m.ObjVal


p_lo, p_hi = df["price"].min(), df["price"].max()
p4, q4, profit4 = solve_joint_qp(beta_0, beta_1, residuals, p_lo, p_hi)
print("\n" + "=" * 60)
print("Part 4: Joint price-quantity optimization (QP)")
print("=" * 60)
print(f"Optimal price p*    : {p4:.4f}")
print(f"Optimal quantity q* : {q4:.4f}")
print(f"Expected profit     : ${profit4:.4f}")


# ============================================================================
# Part 6: Single bootstrap resample
# ============================================================================
def fit_and_solve_bootstrap():
    """Resample the data, refit the regression, and re-solve the joint QP.

    Uses the legacy np.random stream so results match a single, reproducible
    seed set just before the call (np.random.seed)."""
    idx = np.random.choice(len(df), size=len(df), replace=True)
    Xb = df["price"].values[idx].reshape(-1, 1)
    yb = df["demand"].values[idx]
    rb = LinearRegression().fit(Xb, yb)
    eps_b = yb - rb.predict(Xb)
    # Price bounds stay on the original observed range for comparability.
    return rb.intercept_, rb.coef_[0], *solve_joint_qp(
        rb.intercept_, rb.coef_[0], eps_b, p_lo, p_hi)


np.random.seed(42)
b0_boot, b1_boot, p_boot, q_boot, profit_boot = fit_and_solve_bootstrap()
print("\n" + "=" * 60)
print("Part 6: Single bootstrap resample")
print("=" * 60)
print(f"beta_0 (boot) : {b0_boot:.4f}")
print(f"beta_1 (boot) : {b1_boot:.4f}")
print(f"p* (boot)     : {p_boot:.4f}")
print(f"q* (boot)     : {q_boot:.4f}")
print(f"profit (boot) : ${profit_boot:.4f}")


# ============================================================================
# Part 7: Many bootstraps -> distributions of p*, q*, profit
# ============================================================================
B = 500
np.random.seed(123)
opt_prices, opt_quantities, opt_profits = [], [], []
for _ in range(B):
    _, _, pb, qb, prb = fit_and_solve_bootstrap()
    opt_prices.append(pb)
    opt_quantities.append(qb)
    opt_profits.append(prb)
opt_prices = np.array(opt_prices)
opt_quantities = np.array(opt_quantities)
opt_profits = np.array(opt_profits)

print("\n" + "=" * 60)
print(f"Part 7: {B} bootstrap replications")
print("=" * 60)
print(f"Price    : mean={opt_prices.mean():.4f}  std={opt_prices.std(ddof=1):.4f}")
print(f"Quantity : mean={opt_quantities.mean():.4f}  std={opt_quantities.std(ddof=1):.4f}")
print(f"Profit   : mean=${opt_profits.mean():.4f}  std=${opt_profits.std(ddof=1):.4f}")

# Marginal histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(opt_prices, bins=20, edgecolor="black", alpha=0.7)
axes[0].set(xlabel="Optimal price p*", ylabel="Frequency",
            title="Bootstrap distribution of optimal price")
axes[1].hist(opt_quantities, bins=20, edgecolor="black", alpha=0.7)
axes[1].set(xlabel="Optimal quantity q*", ylabel="Frequency",
            title="Bootstrap distribution of optimal quantity")
fig.tight_layout()
fig.savefig(OUT / "part7_hist_price_quantity.png", dpi=150)
plt.close(fig)

# Joint scatter with marginal histograms
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(4, 4)
ax_sc = fig.add_subplot(gs[1:4, 0:3])
ax_hx = fig.add_subplot(gs[0, 0:3], sharex=ax_sc)
ax_hy = fig.add_subplot(gs[1:4, 3], sharey=ax_sc)
ax_sc.scatter(opt_prices, opt_quantities, alpha=0.6)
ax_sc.set(xlabel="Optimal price p*", ylabel="Optimal quantity q*")
ax_sc.grid(alpha=0.3)
ax_hx.hist(opt_prices, bins=20, edgecolor="black", alpha=0.7)
ax_hx.set_ylabel("Count")
plt.setp(ax_hx.get_xticklabels(), visible=False)
ax_hy.hist(opt_quantities, bins=20, orientation="horizontal",
           edgecolor="black", alpha=0.7)
ax_hy.set_xlabel("Count")
plt.setp(ax_hy.get_yticklabels(), visible=False)
ax_hx.set_title("Optimal price vs quantity with marginal histograms", y=0.95)
fig.tight_layout()
fig.savefig(OUT / "part7_scatter_marginals.png", dpi=150)
plt.close(fig)

# Profit histogram (with the fixed-price benchmark)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(opt_profits, bins=20, edgecolor="black", alpha=0.7)
ax.axvline(profit3, color="r", ls="--", lw=2,
           label=f"Fixed-price profit = ${profit3:.2f}")
ax.set(xlabel="Expected profit at optimum", ylabel="Frequency",
       title="Bootstrap distribution of expected profit")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "part7_hist_profit.png", dpi=150)
plt.close(fig)


# ============================================================================
# Part 8: Is the boss's standard newsvendor model as good as ours?
# ============================================================================
# The boss uses the STANDARD newsvendor model: a fixed price (p=1), lost sales
# (no rush orders), and no separate disposal cost. He chooses only q:
#   max_q (1/n) sum_i [ p*min(q, D_i) - c*q ]
# We solve this for the boss's quantity, then evaluate every decision on the
# SAME real-world metric (the rush + disposal cost structure the firm actually
# faces). That is the only fair way to ask "would profit go up if we switched?"
def solve_standard_nvm(D, price):
    """Boss's model: fixed price, lost sales, choose q only."""
    m = gp.Model()
    m.Params.OutputFlag = 0
    nn = len(D)
    q = m.addVar(lb=0.0, name="q")
    h = m.addMVar(nn, lb=-GRB.INFINITY, name="profit")   # per-day profit
    for i in range(nn):
        m.addConstr(h[i] <= price * D[i] - c * q)   # demand-limited case
        m.addConstr(h[i] <= price * q - c * q)      # supply-limited case
    m.setObjective((1 / nn) * gp.quicksum(h[i] for i in range(nn)), GRB.MAXIMIZE)
    m.optimize()
    return q.X, m.ObjVal


def true_profit(price, q, b0=beta_0, b1=beta_1, eps=residuals):
    """Realized expected profit under the firm's true rush + disposal costs."""
    D = b0 + b1 * price + eps
    short = np.maximum(D - q, 0.0)
    exc = np.maximum(q - D, 0.0)
    return float(np.mean(price * D - c * q - g * short - t * exc))


def lostsales_profit(price, q, b0=beta_0, b1=beta_1, eps=residuals):
    """Profit under the boss's own (lost-sales) accounting, for context."""
    D = b0 + b1 * price + eps
    return float(np.mean(price * np.minimum(q, D) - c * q))


q_boss, profit_boss_own = solve_standard_nvm(demand_at_p1, p)

# Three decisions, scored on the firm's true cost structure:
boss_true = true_profit(p, q_boss)        # boss: standard NVM, p=1
fixedmodel_true = true_profit(p, q3)      # correct cost model, still p=1
ours_true = true_profit(p4, q4)           # our joint price-quantity model

print("\n" + "=" * 60)
print("Part 8: Boss's standard newsvendor model vs ours")
print("=" * 60)
print(f"Boss's quantity (standard NVM, p=1) : {q_boss:.4f}")
print("\nExpected profit under the TRUE (rush + disposal) cost structure:")
print(f"  Boss  (standard NVM, p=1, q={q_boss:6.1f}) : ${boss_true:.4f}")
print(f"  Right cost model, fixed p=1 (q={q3:6.1f}) : ${fixedmodel_true:.4f}")
print(f"  Our joint model  (p={p4:.3f}, q={q4:6.1f}) : ${ours_true:.4f}")
print("\nDecomposition of our gain over the boss:")
print(f"  Value of modeling the true cost structure : ${fixedmodel_true - boss_true:+.4f}")
print(f"  Value of optimizing price                 : ${ours_true - fixedmodel_true:+.4f}")
print(f"  Total gain over the boss                  : ${ours_true - boss_true:+.4f} "
      f"({100*(ours_true - boss_true)/boss_true:+.1f}%)")
print("\nHonest caveat - under the boss's own lost-sales accounting:")
print(f"  Boss  (p=1, q={q_boss:6.1f}) : ${lostsales_profit(p, q_boss):.4f}")
print(f"  Ours  (p={p4:.3f}, q={q4:6.1f}) : ${lostsales_profit(p4, q4):.4f}")
print("  -> If rush/disposal costs were not real, the boss's q would be fine.")

# Comparison figure: profit of each decision on the true metric
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels = ["Boss\n(standard NVM)", "Right cost model\n(fixed p=1)", "Our model\n(joint p, q)"]
vals = [boss_true, fixedmodel_true, ours_true]
colors = ["#b0b0b0", "#6699cc", "#2a7f3f"]
bars = axes[0].bar(labels, vals, color=colors, edgecolor="black")
axes[0].set(ylabel="Expected profit ($, true cost structure)",
            title="Profit under the firm's real costs")
axes[0].set_ylim(min(vals) - 8, max(vals) + 4)
for b, v in zip(bars, vals):
    axes[0].text(b.get_x() + b.get_width()/2, v + 0.3, f"${v:.2f}",
                 ha="center", va="bottom", fontweight="bold")
axes[0].grid(axis="y", alpha=0.3)

# Waterfall: how the gain is built up
steps = ["Boss", "+ true\ncost model", "+ price\noptimization"]
deltas = [boss_true, fixedmodel_true - boss_true, ours_true - fixedmodel_true]
cum = np.cumsum([0] + deltas[:-1])
wbars = axes[1].bar(steps, deltas, bottom=cum,
                    color=["#b0b0b0", "#6699cc", "#2a7f3f"], edgecolor="black")
axes[1].set(ylabel="Expected profit ($)", title="Where the gain comes from")
axes[1].set_ylim(0, max(vals) + 10)
for s, d, base in zip(steps, deltas, cum):
    axes[1].text(s, base + d + 0.5,
                 (f"${d:.2f}" if s == "Boss" else f"+${d:.2f}"),
                 ha="center", va="bottom", fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "part8_model_comparison.png", dpi=150)
plt.close(fig)


# ============================================================================
# Save a tidy results table for the report / README
# ============================================================================
results = pd.DataFrame(
    [
        ["beta_0 (intercept)", beta_0],
        ["beta_1 (slope)", beta_1],
        ["residual std", residuals.std()],
        ["Part 3: q* at p=1", q3],
        ["Part 3: expected profit", profit3],
        ["Part 4: p*", p4],
        ["Part 4: q*", q4],
        ["Part 4: expected profit", profit4],
        ["Part 7: bootstrap price mean", opt_prices.mean()],
        ["Part 7: bootstrap price std", opt_prices.std(ddof=1)],
        ["Part 7: bootstrap quantity mean", opt_quantities.mean()],
        ["Part 7: bootstrap quantity std", opt_quantities.std(ddof=1)],
        ["Part 7: bootstrap profit mean", opt_profits.mean()],
        ["Part 7: bootstrap profit std", opt_profits.std(ddof=1)],
        ["Part 8: boss q (standard NVM)", q_boss],
        ["Part 8: boss profit (true costs)", boss_true],
        ["Part 8: fixed-model profit (true costs)", fixedmodel_true],
        ["Part 8: our profit (true costs)", ours_true],
        ["Part 8: gain over boss ($)", ours_true - boss_true],
        ["Part 8: gain over boss (%)", 100 * (ours_true - boss_true) / boss_true],
    ],
    columns=["metric", "value"],
)
results["value"] = results["value"].round(4)
results.to_csv(OUT / "results_summary.csv", index=False)
print(f"\nSaved figures and results_summary.csv to {OUT}")
