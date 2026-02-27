# DID Analysis Script
# Estimates the average treatment effect of turning off eBay's paid search.
# Uses preprocessed pivot tables from preprocess.py.
# Output: LaTeX table in output/tables/did_table.tex


import pandas as pd
import numpy as np
import os
# Load pivot tables saved by preprocess.py
treated_pivot = pd.read_csv('temp/treated_pivot.csv', index_col='dma')
untreated_pivot = pd.read_csv('temp/untreated_pivot.csv', index_col='dma')

# task 2

# compute means
treated_diffs = treated_pivot["log_revenue_diff"].astype(float).dropna()
untreated_diffs = untreated_pivot["log_revenue_diff"].astype(float).dropna()

r1_bar = treated_diffs.mean()
r0_bar = untreated_diffs.mean()

gamma_hat = r1_bar - r0_bar

# standard error
n_treated = treated_diffs.shape[0]
n_untreated = untreated_diffs.shape[0]

var_treated = treated_diffs.var(ddof=1) if n_treated > 1 else 0.0
var_untreated = untreated_diffs.var(ddof=1) if n_untreated > 1 else 0.0

se = np.sqrt((var_treated / n_treated) + (var_untreated / n_untreated))

# 95% ci
ci_lower = gamma_hat - 1.96 * se
ci_upper = gamma_hat + 1.96 * se

print("DID Results (Log Scale)")
print("=======================")
print(f"Gamma hat: {gamma_hat:.4f}")
print(f"Std Error: {se:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")


os.makedirs("output/tables", exist_ok=True)

latex = r"""\begin{table}[h]
\centering
\caption{Difference-in-Differences Estimate of the Effect of Paid Search on Revenue}
\begin{tabular}{lc}
\hline
& Log Scale \\
\hline
Point Estimate ($\hat{\gamma}$) & $%.4f$ \\
Standard Error & $%.4f$ \\
95\%% CI & $[%.4f, \; %.4f]$ \\
\hline
\end{tabular}
\label{tab:did}
\end{table}""" % (gamma_hat, se, ci_lower, ci_upper)

with open("output/tables/did_table.tex", "w") as f:
    f.write(latex)
