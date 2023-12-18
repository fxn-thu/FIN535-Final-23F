import numpy as np
import pandas as pd

from PathGen import *

def Experiments(MC, path_length, ret_df, CAP, SAVE, g, ratio, years=[60,84,120], tax=0):
    """
    Doing evaluations on all MC paths

    Args:
        MC (int): number of MC paths
        path_length: length of each MC path
        ret_df (dataframe): return datafram, can be train or test etc.
        CAP (float): initial capital
        SAVE (float): initial saving
        g (float): saving growth
        ratio (float,list): ratio of asset allocations
        years (list, optional): _description_. Defaults to [60,84,120].
        tax (int, optional): _description_. Defaults to 0.

    Returns:
        Simu_table: _description_ [TO DO]
    """
    # MC simulation --------------------------------------------------------------------------------------------
    # Years need to be 3 examples
    Simu_table = pd.DataFrame(index = range(MC),\
    columns = ['5Y_cap','7Y_cap','10Y_cap','5Y_sharpe','7Y_sharpe','10Y_sharpe',\
        '5Y_mdd','7Y_mdd','10Y_mdd',\
            '5Y_ret','7Y_ret','10Y_ret'])

    for mc in range(MC):
        # Call path generator function; ret_df can be train or test df
        mcdf = Path_Generator(path_length, ret_df) 
        temdf = MC_generate(mcdf,CAP,SAVE,g,ratio)
        temdf.reset_index(inplace=True,drop=True)
        # Loc out end of different years
        rst = temdf.loc[np.r_[years[0]-1,years[1]-1,years[2]-1],'cap_total']
        rst2 = temdf.loc[np.r_[years[0]-1,years[1]-1,years[2]-1],'cap_input'] 
        cumu_rets = (1 + temdf['ret']).cumprod() - 1
        # Total Capital after tax for different dates
        Simu_table.iloc[mc,:3] = rst - tax*(rst - rst2) # Tax Adjustment (only pay for capital gain)
        Simu_table.iloc[mc,3] = ret_annual_sharpe(temdf.iloc[:years[0],1])
        Simu_table.iloc[mc,4] = ret_annual_sharpe(temdf.iloc[:years[1],1])
        Simu_table.iloc[mc,5] = ret_annual_sharpe(temdf.iloc[:years[2],1])
        Simu_table.iloc[mc,6] = max_drawdown(temdf.iloc[:years[0],2]) #MaxdDD on capital level
        Simu_table.iloc[mc,7] = max_drawdown(temdf.iloc[:years[1],2])
        Simu_table.iloc[mc,8] = max_drawdown(temdf.iloc[:years[2],2])
        Simu_table.iloc[mc,9] = cumu_rets[years[0]-1] # cumulative return to each time
        Simu_table.iloc[mc,10] = cumu_rets[years[1]-1]
        Simu_table.iloc[mc,11] = cumu_rets[years[2]-1]
        
    return Simu_table



def GetSummary(Simu_table):
    Summary_table = pd.DataFrame(index = ['5Y','7Y','10Y'], \
    columns = ['Expected Cap','Std Cap','Exp Cap < 500k','Prob >= 500k',\
        'Expected Annual_sharpe','Expected MDD',\
            'ret VaR','ret CVaR','GaR','CGaR'])

    Summary_table.iloc[:,0] = Simu_table.iloc[:,:3].mean()
    Summary_table.iloc[:,1] = Simu_table.iloc[:,:3].std()
    Summary_table.iloc[:,2] = Simu_table.iloc[:,:3][Simu_table.iloc[:,:3] < 500000].mean()
    Summary_table.iloc[:,3] = (Simu_table.iloc[:,:3] >= 500000).mean() # Probability of meeting the Goal
    Summary_table.iloc[:,4] = Simu_table.iloc[:,3:6].mean()
    Summary_table.iloc[:,5] = Simu_table.iloc[:,6:9].mean()
    var = np.quantile(Simu_table.iloc[:,-3:],0.01,axis=0)
    cvar = np.nanmean(Simu_table.iloc[:,-3:][Simu_table.iloc[:,-3:]<=var],axis = 0) 
    cap_var = np.quantile(Simu_table.iloc[:,:3],0.01,axis=0)
    cap_cvar = np.nanmean(Simu_table.iloc[:,:3][Simu_table.iloc[:,:3]<=cap_var],axis = 0) 
    Summary_table.iloc[:,6] = var
    Summary_table.iloc[:,7] = cvar
    Summary_table.iloc[:,8] = cap_var - 500000  
    Summary_table.iloc[:,9] = cap_cvar - 500000
    return Summary_table


