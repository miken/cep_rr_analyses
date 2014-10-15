# coding: utf-8
from ops import to_perc, get_client_rr
client_rr = get_client_rr()

# Here's how the descriptive statistics of the response rate dataset look
rr_global_stats = client_rr.describe()
rr_global_stats["Final"]

mean = rr_global_stats.ix['mean', 'Final'] 
std = rr_global_stats.ix['std', 'Final']
# 95% range of response rates
low_rr_95 = mean - 3*std
high_rr_95 = mean + 3*std
print 'We can expect 95% of the times for response rates to be between {low}-{high}'.format(
    low=to_perc(low_rr_95),
    high=to_perc(high_rr_95)
)
# 67% range of response rates
low_rr_67 = mean - 2*std
high_rr_67 = mean + 2*std
print 'We can expect 67% of the times for response rates to be between {low}-{high}'.format(
    low=to_perc(low_rr_67),
    high=to_perc(high_rr_67)
)

# Next break down response rates by round
raw_round_stats = client_rr.groupby('Round').describe()
# Reshape this table
round_stats = raw_round_stats.reset_index().pivot(index="level_1",columns="Round",values="Final")
round_stats = round_stats.ix[["mean", "count", "std", "min", "25%", "50%", "75%", "max"]]
round_stats = round_stats[["12S", "12X", "12F", "13S", "13X", "13F", "14S", "14X"]]
# Export this file to CSV
round_stats.to_csv('round_stats.csv')


# Next, construct regression model of response rates vs. funder size & bounce numbers
client_rr = final_client_rr
funder_size = client_rr['Targets']
rr = client_rr['Final']
