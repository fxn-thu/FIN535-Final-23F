import pandas as pd
from datetime import datetime
from hmmlearn import hmm
import pickle

data = pd.read_csv('data.csv')
data['month'] = data['month'].apply(lambda x: datetime.strptime(x, '%Y-%m')) #assume each date per month is the last day
data.set_index(['month'],inplace=True)

# Define the number of regimes (A, B, C)
n_regimes = 3

# Train the Hidden Markov Model
model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=1000)
model.fit(data)

# with open('hmm_model.pkl', 'wb') as file:
#     pickle.dump(model, file)
