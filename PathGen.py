import numpy as np
import math


def Path_Generator(length, train):
    """
    Generate return paths from a given dataset
    for example, generating from training days

    Args:
        length (int): return length (for example, 10 years)
        train (csv dataframe) training dataset
    """
    df = train.sample(n=length, replace=True)
    df.drop(columns = 'month', inplace=True)
    df.reset_index(inplace =True, drop=True)
    return df
    

def MC_generate(subdf, CAP, SAVE, g, ratio):
    """Function for one MC path

    Args:
        subdf: return dataframe, with all 5 assets
        simulated from a MC sampling with replacement
        CAP (float): initial capital
        SAVE (float): initial saving
        g (float): saving growth
        ratio (float,list): ratio of asset allocations

    Returns:
        dataframe: history information of a path
    """
    
    # First method
    isJan = np.tile(np.arange(1, 13), 10) == 1
    subdf['month'] = isJan # Manually restart the month id
    # subdf['month'] = (subdf['month'] == 1)  
    subdf.iloc[0,5] = False # Can't can only increase after one year of working
    subdf['addCap_flag'] = subdf['month'].cumsum() # column idx 6

    # Calculate add capital
    subdf['addCap'] = subdf['addCap_flag'].apply(lambda x: SAVE*(1+g)**(x-1)) # column idx 7
    subdf['addCap'] = subdf['addCap'] * subdf['month']

    # return & Capital
    subdf['ret'] = (subdf.iloc[:,:5] * ratio).sum(axis = 1) # column idx 8
    subdf['cap_total'] = Capital_growth(subdf,CAP) # column index 9
    subdf['cap_input'] = CAP + subdf['addCap'].cumsum() # column index 10
    return subdf.loc[:,['addCap','ret','cap_total','cap_input']]


# Functions --------------------------------------------------------------
def Capital_growth(subdf,CAP):
    """
    Calculate capital growth over time
    """
    cap_total = [CAP]
    for t in range(len(subdf)):
        cap_tmp = cap_total[-1]
        if subdf.iloc[t,4]:
            # if month == True (Jan)
            cap_tmp += subdf.iloc[t,7] # new capital col idx 7
        cap_tmp *= (subdf.iloc[t,8] + 1) # Portfolio return; return col idx 8
        cap_total.append(cap_tmp)
    return cap_total[1:]


def ret_annual_sharpe(rst_arr):
    # On return level
    sharpe = rst_arr.mean()/rst_arr.std()
    return sharpe * math.sqrt(12)


def max_drawdown(arr):
    """Calculating maxdrawdown for a given array:
    can be capital or return array; 
    if it's return:
        should be cumulative return 
        cumu_rets = (1 + rst_arr).cumprod()

    Args:
        arr (float,array): return (cumulative) or capital

    Returns:
        float: max drawdown (up to now)
    """
    drawdowns = 1 - arr / np.maximum.accumulate(arr)
    max_drawdown = np.max(drawdowns)
    return max_drawdown