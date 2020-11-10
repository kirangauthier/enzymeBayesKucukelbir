import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import pymc3 as pm
import theano.tensor as tt


def loadMin(df, minLength, frameToSecond, pixelToMeter, file_number):
    iMax = int(df['Trajectory'].iloc[-1])
    x_lis, y_lis, t_lis = [], [], []
    traj_track = []
    traj_id = []

    for i in range(1, iMax + 1):
        idx = (df['Trajectory'] == float(i))
        if np.sum(idx) >= minLength:
            x_lis.append(df[idx]['x'].to_numpy() * pixelToMeter)
            y_lis.append(df[idx]['y'].to_numpy() * pixelToMeter)
            t_lis.append(df[idx]['Frame'].to_numpy() * frameToSecond)
            traj_track.append(np.sum(idx))
            traj_id.append([file_number, i])

    return [x_lis, y_lis, t_lis, traj_track, traj_id]


def combineData(lis):
    # obtain the first traj (from the first item of the list)
    total_var = lis[0][0].reshape(len(lis[0][0]), 1)

    for i in range(len(lis)):

        # for the first item of the list,
        # the start point for the following loop will 1
        if i == 0:
            start = 1
        else:
            start = 0

        # loop through the remaining traj and stack the data
        for j in range(start, len(lis[i])):
            temp_var = lis[i][j].reshape(len(lis[i][j]), 1)
            total_var = np.vstack((total_var, temp_var))

    return total_var.reshape(len(total_var), )


def xTodx(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = x[i][j][1:] - x[i][j][:-1]
    return x


def lookup(track_tot, num_data, isDx):
    # initilize empty int arries to hold lookup values
    lookup_ = np.zeros((num_data,), dtype=int)

    # start and end point for tracking traj
    start, end = 0, 0

    # lookup value of 0 corresponds to the first traj
    lookup_value = 0

    if isDx:
        corrector = 1
    else:
        corrector = 0

    # track_len is the length of ith traj in track_tot
    for i in range(len(track_tot)):
        for track_len in track_tot[i]:
            # update start and end point for next traj
            start = end
            end = end + track_len - corrector

            lookup_[start:end] = lookup_value
            lookup_value += 1

    return lookup_


# Main function for loading data
# this function will go through a folder filled with diffusion csv files
# then selects all trajectories with minimum length >= min_length
# also creates a lookup table for hierarchical model
# function will return all dx, dy and dt data satisfy min_length in single arrays
# will also return the length for each track as well as the lookup table
def loadRawMinData(dir_, min_length=50, isDx=True, frameToSecond=1, pixelToMeter=1):
    entries = Path(dir_)

    # obtain total number of trajectory files
    file_count = 0
    for entry in entries.iterdir():
        if entry.name[0] == '.':
            continue
        file_count += 1

    # initialize empty list to hold data from each csv file
    x_tot_lis, y_tot_lis, t_tot_lis = [], [], []

    # initialize empty list to hold length of each trajectory
    track_tot = []

    # initialize empty list to hold trajectory ID
    track_id = []

    # init files name index
    file_index = 0

    # loop through 1mM urea folder to obatin data
    for entry in entries.iterdir():
        if entry.name[0] == '.':
            continue

        filename = dir_ + entry.name
        # used to locate the file numbering
        if file_index == 0:
            for i, s in enumerate(entry.name):
                if s == '-':
                    file_index = i + 4
                    break
        if entry.name[file_index + 3] == '-':
            file_number = entry.name[file_index:file_index + 5]
        else:
            file_number = entry.name[file_index:file_index + 3]
        df = pd.read_csv(filename)
        temp = loadMin(df, min_length, frameToSecond, pixelToMeter, file_number)

        if len(temp[0]) is not 0:
            x_tot_lis.append(temp[0])
            y_tot_lis.append(temp[1])
            t_tot_lis.append(temp[2])
            track_tot.append(temp[3])
            track_id.append(temp[4])

    # check which data type is needed

    if isDx:
        t_tot_lis = xTodx(t_tot_lis)
        x_tot_lis = xTodx(x_tot_lis)
        y_tot_lis = xTodx(y_tot_lis)
    else:
        pass

    # stack all data point into a 1d array
    x_tot_arr = combineData(x_tot_lis)
    y_tot_arr = combineData(y_tot_lis)
    t_tot_arr = combineData(t_tot_lis)

    # create lookup table for hierarchical model
    lookup_table = lookup(track_tot, int(x_tot_arr.shape[0]), isDx)

    track_tot_ = []
    track_id_ = []
    for i in range(len(track_tot)):
        for j in range(len(track_tot[i])):
            track_tot_.append(track_tot[i][j])
            track_id_.append(track_id[i][j])

    del x_tot_lis, y_tot_lis, t_tot_lis, track_tot, track_id

    print('Total %d files read; Total %d trajectories (length >= %d) loaded; Total %d data points' \
          % (int(file_count), len(track_tot_), min_length, int(x_tot_arr.shape[0])))

    return x_tot_arr, y_tot_arr, t_tot_arr, track_tot_, lookup_table, track_id_


# use track_info and full data set to obtain a specific trajectory
def loadSelectTraj(dx, dy, dt, track_info, index, isDx, is2D=False):
    # check data type
    if isDx:
        corrector = 1
    else:
        corrector = 0

    # get the start position of desired trajectory
    start = 0
    i = -1
    for i, length in enumerate(track_info[:index]):
        start += length - corrector

    #     # treat dt separately
    #     if isDx:
    #         temp_dt = dt[start:start + (track_info[i + 1] - corrector)]
    #     else:
    #         temp_dt = dt[start - (i + 1) * 1:start - (i + 1) + (track_info[i + 1] - 1)]

    # select target trajectory based on start location
    return dx[start:start + (track_info[i + 1] - corrector)], dy[start:start + (track_info[i + 1] - corrector)], \
           dt[start:start + (track_info[i + 1] - corrector)]


def sort_by_entry(list_lis, dtype, order):
    """
    Input is a list of arrays, with different data types.
    function return sorted array with respect to one specific order 
    """
    m = np.array(list_lis[0]).reshape((len(list_lis[0]), 1))
    for i in range(len(list_lis))[1:]:
        temp_arr = np.array(list_lis[i]).reshape((len(list_lis[i]), 1))
        m = np.hstack((m, temp_arr))

    # sort based on specific entry
    temp = []
    for i in range(m.shape[0]):
        temp.append(tuple(m[i, :]))
    sorted_m = np.sort(np.array(temp, dtype=dtype), order=order).tolist()

    # modify the input list
    for i in range(len(sorted_m)):
        for j in range(len(list_lis)):
            list_lis[j][i] = sorted_m[i][j]

    del sorted_m, temp, m


def removeOutLiar(sx, sy, st):
    """detect and remove outliers in the data"""

    zscore_x, zscore_y = np.abs(stats.zscore(sx)), np.abs(stats.zscore(sy))
    outlier_x = np.where(zscore_x >= 2.5)[0]
    outlier_y = np.where(zscore_y >= 2.5)[0]
    all_outlier = list((set(outlier_x).union(set(outlier_y))))
    sx, sy = np.delete(sx, all_outlier), np.delete(sy, all_outlier)
    st = np.delete(st, all_outlier)
    return sx, sy, st


def inspect_prior(D_prior, k_prior, dx, x, ind, bins):
    """Inspect different priors using the simpler gaussian likelihood"""

    # get track
    sdx, sdy, sdt = dx
    sx, sy, st = x

    # simulate prior
    prior = pm.Model()
    with prior:
        D_prior_ = pm.Lognormal('Dp', D_prior[0], D_prior[1])
        k_prior_ = pm.Lognormal('kp', k_prior[0], k_prior[1])

    prior_samples = pm.sample_prior_predictive(samples=4000, model=prior)

    Dp = prior_samples['Dp']
    kp = prior_samples['kp']

    # paramter inference
    model = pm.Model()
    with model:
        D = pm.Lognormal('D', D_prior[0], D_prior[1])
        k = pm.Lognormal('k', k_prior[0], k_prior[1])

        mean_x = (-(sx - sx.mean())[:-1]) * (1 - tt.exp(-k * sdt))
        mean_y = (-(sy - sy.mean())[:-1]) * (1 - tt.exp(-k * sdt))
        std = tt.sqrt(D * (1 - tt.exp(-2 * k * sdt)) / k)

        like_x = pm.Normal('like_x', mu=mean_x, sd=std, observed=sdx)
        like_y = pm.Normal('like_y', mu=mean_y, sd=std, observed=sdy)

    with model:
        trace = pm.sample(2000, tune=2000, chains=2, cores=2)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(sx, sy, label='Track: %d, length: %d' % (ind, len(sx)))
    axes[0].legend()

    axes[1].hist(np.log(trace['D']), bins=bins, density=True, alpha=0.5, label='posterior')
    axes[1].hist(np.log(Dp), bins=bins, density=True, alpha=0.5,
                 label='prior: D(%d, %d)' % (D_prior[0], D_prior[1]))
    axes[1].legend()

    axes[2].hist(np.log(trace['k']), bins=bins, density=True, alpha=0.5, label='posterior')
    axes[2].hist(np.log(kp), bins=bins, density=True, alpha=0.5,
                 label='prior: k(%d, %d)' % (k_prior[0], k_prior[1]))
    axes[2].legend()

    plt.show()

    return trace


def MAP_hpw(sx, sy):
    corr_y = (sy - sy.mean())
    corr_x = (sx - sx.mean())

    lam = np.log(
        np.sum(corr_y[1:] ** 2 + corr_x[1:] ** 2) / np.sum(corr_y[1:] * corr_y[:-1] + corr_x[1:] * corr_x[:-1]))

    I = 1 - np.exp(-2 * lam)
    D = lam * (np.sum((corr_y[1:] - corr_y[:-1] * np.exp(-lam)) ** 2 +
                      (corr_x[1:] - corr_x[:-1] * np.exp(-lam)) ** 2) / I / 2 / (len(sy) - 1))

    return D, lam


def MAP_bm(dx, dy, dt, a, b):
    alpha = len(dx) + a
    beta = np.sum((dx ** 2 + dy ** 2) / 4 / dt) + b
    return beta / (alpha - 1)