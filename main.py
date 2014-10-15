# coding: utf-8
from ops import to_perc, get_client_rr, rr_stats_by_round
client_rr = get_client_rr()

# Here's how the descriptive statistics of the response rate dataset look
rr_global_stats = client_rr.describe()
rr_global_stats["Final"]

mean = rr_global_stats.ix['mean', 'Final'] 
std = rr_global_stats.ix['std', 'Final']
# Compute the low and the high values 
# 67% range of response rates
low_rr_67 = mean - 2*std
high_rr_67 = mean + 2*std
print 'We can expect 95% of the times for response rates to be between {low}-{high}'.format(
    low=to_perc(low_rr_67),
    high=to_perc(high_rr_67)
)

# Retrieve descriptive statistics by round
round_stats = rr_stats_by_round(client_rr)
round_stats


# Next, construct regression model of response rates vs. funder size & bounce numbers
client_rr = final_client_rr
funder_size = client_rr['Targets']
rr = client_rr['Final']
