import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = '/Users/juanvilla/ebay_replication'

DATA_PATH = os.path.join(BASE_DIR, 'input', 'PaidSearch.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'output', 'figures')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')


# Load data
df = pd.read_csv('input/PaidSearch.csv')
df['log_revenue'] = np.log(df['revenue'])

#treated pivot

treated_units = df[df['search_stays_on'] == 0]

treated_pivot = treated_units.pivot_table(index='dma', columns='treatment_period', values='log_revenue')
treated_pivot.columns = ['log_revenue_pre', 'log_revenue_post']
treated_pivot['log_revenue_diff'] = treated_pivot['log_revenue_post'] - treated_pivot['log_revenue_pre']

treated_pivot.to_csv(os.path.join(TEMP_DIR, 'treated_pivot.csv'))

treated_pivot.head()

# Compute the average log difference for treated units
avg_log_diff_treated = treated_pivot['log_revenue_diff'].mean()

# Compute the variance of the log differences for treated units and divide by the number of treated units
var_log_diff_treated = treated_pivot['log_revenue_diff'].var() / len(treated_pivot)

#untreated pivot

untreated_units = df[df['search_stays_on'] == 1]

untreated_pivot = untreated_units.pivot_table(index='dma', columns='treatment_period', values='log_revenue')
untreated_pivot.columns = ['log_revenue_pre', 'log_revenue_post']
untreated_pivot['log_revenue_diff'] = untreated_pivot['log_revenue_post'] - untreated_pivot['log_revenue_pre']

untreated_pivot.to_csv(os.path.join(TEMP_DIR, 'untreated_pivot.csv'))

untreated_pivot.head()

# Compute the average log difference for untreated units
avg_log_diff_untreated = untreated_pivot['log_revenue_diff'].mean()

# Compute the variance of the log differences for untreated units and divide by the number of untreated units
var_log_diff_untreated = untreated_pivot['log_revenue_diff'].var() / len(untreated_pivot)

# Compute the difference between the average log differences (gamma_hat)
gamma_hat = avg_log_diff_treated - avg_log_diff_untreated

# Compute the sum of the variances
sum_variances = var_log_diff_treated + var_log_diff_untreated

# Compute the standard error
standard_error = np.sqrt(sum_variances)

# Compute the 95% confidence interval for the treatment effect
ci_lower = gamma_hat - 1.96 * standard_error
ci_upper = gamma_hat + 1.96 * standard_error

# Exponentiate the midpoint and the extremes of the interval
gamma_hat_exp = np.exp(gamma_hat)
ci_lower_exp = np.exp(ci_lower)
ci_upper_exp = np.exp(ci_upper)

# Display stats

treated_dmas = df[df['search_stays_on'] == 0]['dma'].nunique()
untreated_dmas = df[df['search_stays_on'] == 1]['dma'].nunique()

print(f"Treated DMAs: {treated_dmas}")
print(f"Untreated DMAs: {untreated_dmas}")
print(f"Date range: {min(df.date)} to {max(df.date)}")


#fig 2


df['date'] = pd.to_datetime(df['date'], format='%d-%b-%y')

daily_avg = df.groupby(['date', 'search_stays_on'])['revenue'].mean().reset_index()

control = daily_avg[daily_avg['search_stays_on'] == 1]
treatment = daily_avg[daily_avg['search_stays_on'] == 0]

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(control['date'], control['revenue'], label='Control (search stays on)', color='blue')
ax.plot(treatment['date'], treatment['revenue'], label='Treatment (search goes off)', color='red')

ax.axvline(pd.Timestamp('2012-05-22'), color='black', linestyle='--', label='Treatment Date (May 22, 2012)')

ax.set_xlabel('Date')
ax.set_ylabel('Revenue')
ax.set_title('Average Daily Revenue for Treatment and Control DMAs')
ax.legend()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure_5_2.png'), dpi=150)
plt.show()


#fig 3
df['log_revenue'] = np.log(df['revenue'])

daily_avg_log = df.groupby(['date', 'search_stays_on'])['log_revenue'].mean().reset_index()

pivot = daily_avg_log.pivot(index='date', columns='search_stays_on', values='log_revenue')
pivot.columns = ['treatment', 'control']
pivot['diff'] = pivot['control'] - pivot['treatment']

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(pivot.index, pivot['diff'], color='blue')

ax.axvline(pd.Timestamp('2012-05-22'), color='black', linestyle='--', label='Treatment Date (May 22, 2012)')

ax.set_xlabel('Date')
ax.set_ylabel('log(rev_control) - log(rev_treat)')
ax.set_title('Difference in Log Average Revenue: Control vs Treatment DMAs')
ax.legend()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'figure_5_3.png'), dpi=150)
plt.show()
