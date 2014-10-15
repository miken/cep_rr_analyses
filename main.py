# coding: utf-8
from ops import to_perc, get_client_rr, rr_stats_by_round, show_expected_rr_range
client_rr = get_client_rr()

# Here's how the descriptive statistics of the response rate dataset look
rr_global_stats = client_rr.describe()
rr_global_stats["Final"]

show_expected_rr_range(rr_global_stats)

# Retrieve descriptive statistics by round
round_stats = rr_stats_by_round(client_rr)
round_stats[["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6", "Final"]]


# Next, construct regression model of response rates vs. funder size & bounce numbers
client_rr = final_client_rr
funder_size = client_rr['Targets']
rr = client_rr['Final']
