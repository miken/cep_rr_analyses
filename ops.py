# coding: utf-8
import pandas as pd
from sklearn import linear_model

# Helper functions
def to_perc(val):
    return "{0:.0f}%".format(val*100)


def map_round_to_season(round_str):
    '''
    Using a dict, this function will convert a round_str, e.g., '12X'
    into the appropriate season, e.g., 'Summer'
    '''
    season_map = {
        "S": "Spring",
        "X": "Summer",
        "F": "Fall",
    }
    return season_map[round_str[-1]]


# Main functions
def read_data():
    '''
    Read data from data.xlsx and then return a dictionary
    with each key corresponding to the survey round and
    value the response rate data for that round
    '''
    xl = pd.ExcelFile('data.xlsx')
    dsets = {}
    for n in xl.sheet_names:
        dsets[n] = xl.parse(n)
    return dsets


def get_client_rr():
    '''
    Convert the dictionary object generated by read_data()
    function into a stacked csv file of all response rate
    data collected over the 8 rounds
    '''
    # Bring data in
    dsets = read_data()
    raw_collist = [
        "Client",
        "Round",
        "Week 1",
        "Week 2",
        "Week 3",
        "Week 4",
        "Week 5",
        "Week 6",
        "Final",
        "Targets",
        "#Bounces",
        "%Bounces",
    ]
    final_client_rr = pd.DataFrame(columns=raw_collist)
    for rnd, data in dsets.iteritems():
        data["Round"] = rnd
        sliced = data[raw_collist]
        final_client_rr = final_client_rr.append(sliced)
    # Then reindex
    final_client_rr = final_client_rr.reset_index(drop=True)
    # Add "Season" column
    final_client_rr["Season"] = final_client_rr["Round"].apply(map_round_to_season)
    # Export to CSV for use later
    final_client_rr.to_csv('client_rr_data.csv', encoding='utf-8', index=False)
    return final_client_rr


def show_expected_rr_range(rr_stats):
    '''
    Given the descriptive statistics dataset rr_stats, this function will
    output text telling users the 95% range of expected response rates
    '''
    mean = rr_stats.ix['mean', 'Final'] 
    std = rr_stats.ix['std', 'Final']
    # Compute the low and the high values 
    # 67% range of response rates
    low_rr_67 = mean - 2*std
    high_rr_67 = mean + 2*std
    print 'We can expect 95% of the times for response rates to be between {low}-{high}'.format(
        low=to_perc(low_rr_67),
        high=to_perc(high_rr_67)
    )


def rr_stats_by_round(client_rr):
    '''
    Given the client response rate dataset client_rr, this
    function will generate descriptive statistics broken down
    by survey round

    Make sure you have `numexpr` package installed for the query to work
    '''
    # Next break down response rates by round
    raw_round_stats = client_rr.groupby('Round').describe()
    # We're only interested in mean response rates across weeks here,
    # so let's filter for those indices
    # First, name index for query
    raw_round_stats.index.names = [u'Round', u'Stats']
    # Now do the query
    round_stats = raw_round_stats.query('Stats == "mean"')
    # Reindex the table with just round values
    round_stats["Round"] = round_stats.index.levels[0]
    # Reset the index
    round_stats = round_stats.reset_index(drop=True).set_index("Round")
    # Reorder the index
    round_stats = round_stats.ix[["12S", "12X", "12F", "13S", "13X", "13F", "14S", "14X"]]
    # Export this file to CSV
    round_stats.to_csv('round_stats.csv')
    return round_stats


def rr_stats_by_season(client_rr):
    '''
    Given the client response rate dataset client_rr, this
    function will generate descriptive statistics broken down
    by survey round

    Make sure you have `numexpr` package installed for the query to work
    '''
    # Next break down response rates by round
    raw_round_stats = client_rr.groupby('Season').describe()
    # We're only interested in mean response rates across weeks here,
    # so let's filter for those indices
    # First, name index for query
    raw_round_stats.index.names = [u'Round', u'Stats']
    # Now do the query
    round_stats = raw_round_stats.query('Stats == "mean"')
    # Reindex the table with just round values
    round_stats["Round"] = round_stats.index.levels[0]
    # Reset the index
    round_stats = round_stats.reset_index(drop=True).set_index("Round")
    # Reorder the index
    round_stats = round_stats.ix[["12S", "12X", "12F", "13S", "13X", "13F", "14S", "14X"]]
    # Export this file to CSV
    round_stats.to_csv('round_stats.csv')
    return round_stats