import cvxpy as cp
import numpy as np

def get_mu_Sigma_rf1(data, year):
    # data cols = [SPY, TLT, BAB, rf]
    mu = data.iloc[:year*12,:].mean().to_numpy().reshape(4,1)
    Sigma = data.iloc[:year*12,:].cov().to_numpy().reshape(4,4)
    rf = data.iloc[:year*12,-1].mean()
    return mu, Sigma, rf

def max_sharpe_opt1(data, year, allow_short = False):
    # compute mu, Sigma, rf for the given time period
    mu, Sigma, rf = get_mu_Sigma_rf1(data, year)
    
    # Create optimization variables.
    y = cp.Variable((4,1))
    # Create two constraints.
    if allow_short:
        constraints = [(mu[[0,1,3]]-np.ones((3,1))*rf).T@y[[0,1,3]] == 1]
    else:
        constraints = [y[[0,1,3]]>= 0, (mu[[0,1,3]]-np.ones((3,1))*rf).T@y[[0,1,3]] == 1]
    # Form objective.
    obj = cp.Minimize(cp.quad_form(y, Sigma))
    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve() # Returns the optimal value.
#     print("status:", prob.status)
    print("optimal objective value", prob.value)
    print("optimal var y", y.value)
    
    # compute allocation weights x
    x = y.value/sum(y.value[[0,1,3]])
    x[[2,0]] = y.value[[2,0]]/sum(y.value[[0,1,3]])
    print("optimal allocation x", x)
    
    return x