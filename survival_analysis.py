import pandas as pd
import numpy as np
import datetime
from lifelines.utils import datetimes_to_durations
from lifelines import KaplanMeierFitter


def get_death_time_v1(x):
    end_time = None
    dead = False
    for i, p in enumerate(x):
        if not dead:
            end_time = x.index[i]
        # Require at least 3 months of no activity
        if x[i] == 0 and x[i - 1] == 0 and x[i - 2] == 0:
            dead = True
            # print('dead:', x.index[i], p)
        else:
            dead = False
    if end_time and dead:
        return pd.to_datetime(end_time.split('_')[0])
    return np.nan


def get_death_time_v2(x):
    # In how many months do we have payment activity?
    active_months = len(x[(x > 0)])

    # remove the number of trailing '0' payment months
    length = len(x)
    add = True
    counter = 0
    for i in range(0, length):
        i += 1
        if x[-i] == 0:
            if add:
                counter += 1
        else:
            add = False
    if counter > 0:
        y = x[:-counter]
    else:
        y = x
    removed_trailing_months = len(y)

    # Calculate the average payment frequency observed, ignoring any recent '0' payment months
    frequency = int(removed_trailing_months / active_months)

    # As dead time threshold I set 2*frequency.
    # So if a customer did not have any payments for twice his usual payment frequency
    # I consider that customer churned.
    dead_time = 2 * frequency

    # Extract deadtime if it exists:
    end_time = None
    dead = False
    z = x[int(removed_trailing_months):int(removed_trailing_months + dead_time)]

    if len(z) > 0:
        if len(z[z == 0.0]) == dead_time:
            end_time = x.index[removed_trailing_months]
            dead = True
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


def run_two_churn_defs(df):
    for i, f in enumerate([get_death_time_v1, get_death_time_v2]):
        # Calculate the death date over all rows
        df['death_date'] = df[cols_payments].apply(lambda x: f(x), axis=1)

        # Create duration and churn status
        start_times = df['incorporation_date']
        end_times = df['death_date']
        obs_time = datetime.datetime(2015, 1, 1)
        T, E = datetimes_to_durations(start_times, end_times, freq='M', fill_date=obs_time)
        df['T'] = T  # duration (in months)
        df['E'] = E  # churn status

        kmf = KaplanMeierFitter()
        kmf.fit(T, event_observed=E)  # or, more succiently, kmf.fit(T, E)
        if i == 0:
            ax = kmf.plot()
        else:
            kmf.plot(ax=ax)
    # plt.title('KM Survival Function using v1 churn def')
    pass

if __name__ == '__main__':

    df = get_data()

    # Reorder and select columns
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
        kmf.fit(T[ix], E[ix], label=_type)
        if i == 0:
            ax = kmf.plot()
        else:
            kmf.plot(ax=ax)
