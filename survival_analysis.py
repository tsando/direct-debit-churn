import pandas as pd
import numpy as np
import datetime
from lifelines.utils import datetimes_to_durations
from lifelines import KaplanMeierFitter


def get_death_time_v2(x):
    end_time = None
    dead = False
    for i, p in enumerate(x):
        if dead == False:
            end_time = x.index[i]
        # Require at least 3 months of no activity
        if x[i] == 0 and x[i-1]== 0 and x[i-2] == 0:
            dead = True
            # print('dead:', x.index[i], p)
        else:
            dead = False
    if end_time and dead:
        return pd.to_datetime(end_time.split('_')[0])
    return np.nan


def get_death_time_v1(x):
    # Considering dead if no payments in two consecutive months
    end_time = None
    dead = False
    for i, p in enumerate(x):
        if dead == False:
            end_time = x.index[i]
        # Require at least 2 months of no activity
        if x[i] == 0 and x[i-1]== 0:
            dead = True
            # print('dead:', x.index[i], p)
        else:
            dead = False
    if end_time and dead:
        return pd.to_datetime(end_time.split('_')[0])
    return np.nan


def get_data():
    # Data preproc
    df = pd.read_csv('data/monthly_data.csv')
    # Get rid of redundant first column
    df = df.iloc[:, 1:]
    # Convert string to datetime
    df['incorporation_date'] = pd.to_datetime(df['incorporation_date'])
    return df


if __name__ == '__main__':

    df = get_data()

    # # Reorder columns
    cols_chars = ['company_id', 'vertical', 'incorporation_date']
    cols_payments = [x for x in df if 'payments' in x]
    df = df[cols_chars + cols_payments]

    # Calculate the death date over all rows
    df['death_date'] = df[cols_payments].apply(lambda x: get_death_time_v2(x), axis=1)

    # Create duration and churn status
    start_times = df['incorporation_date']
    end_times = df['death_date']
    obs_time = datetime.datetime(2015, 1, 1)
    T, E = datetimes_to_durations(start_times, end_times, freq='M', fill_date=obs_time)
    df['T'] = T  # duration (in months)
    df['E'] = E  # churn status

    kmf = KaplanMeierFitter()

    # vertical_type = 'gym/fitness'
    vertical_types = np.unique(df['vertical'])
    for i, _type in enumerate(vertical_types):
        ix = (df['vertical'] == _type)
        kmf.fit( T[ix], E[ix], label=_type)
        if i == 0:
            ax = kmf.plot()
        else:
            kmf.plot(ax=ax)
