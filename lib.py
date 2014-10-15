from __future__ import division
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, levene, f_oneway, f
from math import sqrt
from itertools import combinations
from qsturng import psturng

# ===== STATISTICAL TESTS USED BY CEP =====
def get_cohens(sample_a, sample_b):
    '''
    Calculate absolute value of Cohen's d from two samples
    Sample A and Sample B are array-like data stores
    Ideally they should be numpy arrays or pandas Series
    So we can perform mean and standard deviation calculations with them    
    '''
    mean_a = sample_a.mean()
    mean_b = sample_b.mean()
    std_a = sample_a.std()
    std_b = sample_b.std()
    numer = mean_a - mean_b
    denom = sqrt((std_a**2 + std_b**2) / 2)
    cohens_d = numer / denom
    return abs(cohens_d)


def welch_anova(*args):
    '''
    This helper function calculate Welch's ANOVA where
    the homogeneity assumption of variance is violated
    args here is the list of array-like data stores, ideally numpy arrays
    See this web link for the derived formula:
    http://www.uvm.edu/~dhowell/gradstat/psych340/Lectures/Anova/anova2.html
    '''
    # Number of groups
    k = len(args)
    total_weight = 0
    total_weighted_sum = 0
    weight_list = []
    mean_list = []
    count_list = []
    for sample in args:
        mean = sample.mean()
        mean_list.append(mean)
        var = sample.var()
        count = sample.count()
        count_list.append(count)
        weight = count / var
        weight_list.append(weight)
        total_weight += weight
        weighted_sum = weight * mean
        total_weighted_sum += weighted_sum
    weighted_grand_mean = total_weighted_sum / total_weight
    # Next, let's find Welch's F
    total_weighted_var = 0
    crazy_sum = 0
    for w, m, n in zip(weight_list, mean_list, count_list):
        # This part is used for f_stat calculation
        element = w * ((m - weighted_grand_mean) ** 2)
        total_weighted_var += element
        denom_squared_element = (1 - w / total_weight) ** 2
        crazy_element = denom_squared_element / (n - 1)
        crazy_sum += crazy_element
    f_numer = total_weighted_var / (k - 1)
    f_denom = 1 + 2 * (k - 2) * crazy_sum / (k**2 - 1) 
    f_stat = f_numer / f_denom
    # Next, let's find Welch's degree of freedom
    df = (k**2 - 1) / (3 * crazy_sum)
    # Now determine p-value from df
    pval = 1 - f.cdf(f_stat, k - 1, df)
    return f_stat, pval


def get_msw_et_al(*args):
    '''
    This helper function calculates mean squares within of a list of samples
    args here is the list of array-like data stores, ideally numpy arrays
    This function returns three values:
        msw: Mean Squares Within
        k: number of groups
        df_within: degree of freedom
    '''
    # N is the total number of cases
    # k is the number of groups
    N = 0
    k = len(args)
    # Within Sum of Squares
    wss = 0
    for sample in args:
        count = sample.count()
        N += count
        var = sample.var()
        squares = var * (count - 1)
        wss += squares
    # Finally divide WSS by df_within
    df_within = N - k
    msw = wss / df_within
    return msw, k , df_within


def tukey(sample_a, sample_b, **kwargs):
    '''
    Calculate Tukey's HSD and significance from two samples
    Sample A and Sample B are array-like data stores
    Ideally they should be numpy arrays or pandas Series
    So we can perform mean and standard deviation calculations with them
    We'll also pass the Mean Squares Within here as msw
    This functions will return the mean difference and the p-value
        r: number of samples in total
        df: degrees of freedom - this will be the sum of (count of each sample -1)
    '''
    # Retrieve arguments
    msw = kwargs.get('msw')
    r = kwargs.get('r')
    df = kwargs.get('df')
    mean_a = sample_a.mean()
    count_a = sample_a.count()
    mean_b = sample_b.mean()
    count_b = sample_b.count()
    standard_error = sqrt(msw * (1/2) * (1/count_a + 1/count_b))
    mean_diff = mean_a - mean_b
    q = abs(mean_diff) / standard_error
    p = psturng(q, r, df)
    return mean_diff, p



def gh(sample_a, sample_b, **kwargs):
    '''
    Calculate Games-Howell from two samples
    Sample A and Sample B are array-like data stores
    Ideally they should be numpy arrays or pandas Series
    So we can perform mean and standard deviation calculations with them
    This functions will return the mean difference and the p-value
    '''
    # Retrieve argument(s)
    r = kwargs.get('r')
    # For Games-Howell, we'll have to calculate a custom standard error
    # And custom df to get q statistic
    mean_a = sample_a.mean()
    var_a = sample_a.var()
    count_a = sample_a.count()
    s2n_a = var_a / count_a
    mean_b = sample_b.mean()
    var_b = sample_b.var()
    count_b = sample_b.count()
    s2n_b = var_b / count_b
    standard_error = sqrt((1/2) * (s2n_a + s2n_b))
    mean_diff = mean_a - mean_b
    q = abs(mean_diff) / standard_error
    # Next, calculate custom df
    df_numer = (s2n_a + s2n_b)**2
    df_denom = (s2n_a**2 / (count_a - 1)) + (s2n_b**2 / (count_b - 1))
    df = df_numer / df_denom
    p = psturng(q, r, df)
    return mean_diff, p    


def translate_result(pval, mean_diff, sample_a, sample_b):
    '''
    This function returns a tuple of verdict, sign, and cohens_d
    verdict can be "Not significant", "Small effect size", etc.
    sign can be blank, "*", "**", etc.
    based on the given p-value
    '''
    if pval < SIG_LEVEL:
        cohens_d = get_cohens(sample_a, sample_b)
        if cohens_d < .15:
            lang = 'NONE'
        elif .15 <= cohens_d < .45:
            lang = 'SMALL'
        elif .45 <= cohens_d < .75:
            lang = 'MEDIUM'
        elif .75 <= cohens_d:
            lang = 'LARGE'
        message = EFF_LANG_DICT[lang]
        cohen_sign = EFF_SIGN_DICT[lang]
        # Append a rounded mean_diff to the sign
        # ===== Mike 10/06 =====
        # Temporarily set to 4 decimal points 
        # And disable cohen_sign so Ramya can do checking
        diff = '%0.4f' % mean_diff
        sign = diff
        # sign = '{diff}{cs}'.format(diff=diff, cs=cohen_sign)
        verdict = message.title()
    else:
        verdict = "Not significant"
        sign = None
        cohens_d = None
    return verdict, sign, cohens_d


def cep_ttest(sample_a, sample_b):
    '''
    Sample A and Sample B are array-like data stores
    Ideally they should be numpy arrays or pandas Series
    So we can perform mean and standard deviation calculations with them
    The function will return a dictionary with the following entries:
        "test": "Standard" (equal variance) or "Welch" (not equal variance)
        "pval": P-value of the test performed
        "verdict": "Not significant" or effect size specified
        "cohen": Cohen's d value
        "sign": blank, ".", "*", "**", or "***" depending on p-value and significance
        "g1_n": response count in sample_a
        "g2_n": response count in sample_b
    '''
    # Construct a result_dict
    result_dict = {}
    # First, perform a Levene's test to determine whether the samples have equal variances
    equal_var_test = levene(sample_a, sample_b, center='mean')
    # The significance stat is the second element in the result tuple
    equal_var_test_sig = equal_var_test[1]
    # Then, depending on the result, we'll perform either a standard or a Welch's test
    # If there's no result, then end test here
    if pd.isnull(equal_var_test_sig):
        result_dict['test'] = 'N/A'
    else:
        if equal_var_test_sig >= SIG_LEVEL:
            equal_var_arg = True
            result_dict['test'] = 'Standard'
        elif equal_var_test_sig < SIG_LEVEL:
            equal_var_arg = False
            result_dict['test'] = 'Welch'
        ttest_result = ttest_ind(sample_a, sample_b, axis=0, equal_var=equal_var_arg)
        ttest_result_sig = ttest_result[1]
        result_dict['pval'] = ttest_result_sig
        # If it's not significant, end here
        # Translate result here
        mean_diff = sample_a.mean() - sample_b.mean()
        verdict, sign, cohens_d = translate_result(ttest_result_sig, mean_diff, sample_a, sample_b)
        result_dict['cohen'] = cohens_d
        result_dict['verdict'] = verdict
        result_dict['sign'] = sign
        result_dict['g1_n'] = sample_a.count()
        result_dict['g2_n'] = sample_b.count()
        result_dict['g1_mean'] = sample_a.mean()
        result_dict['g2_mean'] = sample_b.mean()        
    return result_dict


def cep_anova(samples_dict):
    '''
    Perform ANOVAs for the samples listed in sample_list
    '''
    samples_list = samples_dict.values()
    result_dict = {}
    # First, perform a Levene test to determine the homogeneity of variance
    equal_var_test = levene(*samples_list, center='mean')
    # The significance stat is the second element in the result tuple
    equal_var_test_sig = equal_var_test[1]
    # Then, depending on the result, we'll perform either a standard or a Welch's test
    # If there's no result, then end test here
    if pd.isnull(equal_var_test_sig):
        result_dict['test'] = 'N/A'
    else:
        if equal_var_test_sig >= SIG_LEVEL:
            result_dict['test'] = 'Standard'
            # Perform an ANOVA here
            anova_result = f_oneway(*samples_list)
        elif equal_var_test_sig < SIG_LEVEL:
            result_dict['test'] = 'Welch'
            # Perform a Welch test here
            anova_result = welch_anova(*samples_list)
        anova_result_sig = anova_result[1]
        result_dict['anova_p'] = anova_result_sig
        if anova_result_sig < SIG_LEVEL:
            # If significant, we'll continue with posthoc tests
            # First, split samples into pairs so we can perform tests
            # on each pair
            c = combinations(samples_dict.items(), 2)
            pairs_dict = {}
            for i in c:
                # Get the value tuple first
                val_tuple = i[0][0], i[1][0]
                # Then the sample tuple
                sample_tuple = i[0][1], i[1][1]
                # Then assign all to pairs_dict
                pairs_dict[val_tuple] = sample_tuple
            # If we did standard test earlier, follow with Tukey posthoc
            # If we did Welch earlier, follow with Games-Howell
            # First, let's calculate msw, r, and df to feed into the posthoc
            msw, r, df = get_msw_et_al(*samples_list)
            kwargs_dict = {}
            kwargs_dict['r'] = r
            if result_dict['test'] == 'Standard':
                result_dict['posthoc'] = 'Tukey'
                posthoc = tukey
                kwargs_dict['msw'] = msw
                kwargs_dict['df'] = df
            elif result_dict['test'] == 'Welch':
                result_dict['posthoc'] = 'Games-Howell'
                posthoc = gh
            for key, sample_tuple in pairs_dict.items():
                sample_a = sample_tuple[0]
                sample_b = sample_tuple[1]
                mean_diff, pval = posthoc(sample_a, sample_b, **kwargs_dict)
                # Translate result into verdict, sign, and cohens_d
                # And save this tuple in the key entry of the result_dict
                result_dict[key] = translate_result(pval, mean_diff, sample_a, sample_b)
    return result_dict
