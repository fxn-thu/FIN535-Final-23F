import numpy as np
import pandas as pd
import math
import random
import pickle


def Path_Generator(length, data):
    """
    Generate return paths from a given dataset
    for example, generating from training days

    Args:
        length (int): return length (for example, 10 years)
        train (csv dataframe) training dataset
    """
    i = random.randint(0, data.shape[0]-length)
    df = data.iloc[i:i+length,:]
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
    isJan = np.tile(np.arange(1, 13), int(subdf.shape[0]/12)) == 1
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
    sharpe = rst_arr.mean()*12/(rst_arr.std()* math.sqrt(12))
    return sharpe 


def ret_annual_sortino(rst_arr):
    # On return level
    downside_dev = np.sqrt(np.mean(np.minimum(0, rst_arr - rst_arr.mean())**2))
    sortino = rst_arr.mean()*12/(downside_dev* math.sqrt(12))
    return sortino


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
    drawdowns = np.maximum.accumulate(arr)/arr-1
    max_drawdown = np.max(drawdowns)
    return max_drawdown


def max_drawdown2(rst_arr):
    """Calculating maxdrawdown for cumulative returns arr

    Args:
        rst_arr (float,array): cumulative return

    Returns:
        float: max drawdown (up to now)
    """
    arr = (1 + rst_arr).cumprod()
    drawdowns = np.maximum.accumulate(arr)/arr-1
    max_drawdown = np.max(drawdowns)
    return max_drawdown

def Path_Generator_random(length, train):
    """
    Generate return paths from a given dataset
    for example, generating from training days

    Args:
        length (int): return length (for example, 10 years)
        train (csv dataframe) training dataset
    """
    df = train.sample(n=length, replace=True)
    return df

# Functions for HMM
def simulate_returns(model, start_regime, n_days):
    current_regime = start_regime
    simulated_returns = []
    
    for _ in range(n_days):
        # Sample the next regime based on transition probabilities
        next_regime = np.random.choice(np.arange(3), p=model.transmat_[current_regime])
        
        # Sample a return from the selected regime
        simulated_return = model.sample(1)[0][0]
        
        simulated_returns.append(simulated_return)
        current_regime = next_regime
    
    return simulated_returns

def Path_Generator_HMM(length):
    """
    Path Generator with HMM model
    With Three regimes, with starting regimes as 2

    Args:
        length (int): length of predicting dates
    """
    with open('hmm_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    simulated_returns = simulate_returns(loaded_model, start_regime=2, n_days=length)
    simulated_returns = np.array(simulated_returns)
    df = pd.DataFrame(simulated_returns, columns=['SPY', 'TLT', 'BAB', 'VOL', 'rf'])
    return df