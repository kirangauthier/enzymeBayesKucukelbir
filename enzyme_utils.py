import numpy as np
import pandas as pd
import numpy.linalg as la
import pymc3 as pm
import theano.tensor as tt
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
# import Generative_Model as GM
import logging 
import os 
from scipy import stats 


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
                    file_index = i+4
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


def MCevidence(theta, logp, k=1):
    '''
    MCevidence takes as input:
        N x m array of N m-dimensional parameter estimates, theta
        N x 1 array of unnormalized log-probability values, prob

    It returns an estimate for the log of the normalizing constant and the uncertainty
    '''
    N, m = theta.shape

    # Pre-Whiting
    if m > 1:
        mean = np.reshape(np.mean(theta, axis=0), (m, 1))  # m x 1
        covariance = np.cov(np.transpose(theta))  # m x m
        [w, v] = la.eig(covariance)
        white = np.transpose(np.diag(1 / np.sqrt(w)) @ np.transpose(v) @ (np.transpose(theta) - mean))
    else:
        w = 1
        white = theta

    # nearest-neighbor distances
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(white)
    distances, indices = nbrs.kneighbors(white)

    # Evidence
    logTemp = logp + 0.5 * np.log(w).sum()
    logE = logTemp.max() + np.log(N / (N * k + 1)) + \
           np.log(np.sum(np.pi ** (m / 2) * distances[:, k] ** m * np.exp(logTemp - logTemp.max())))
    return logE, 1 / np.sqrt(N * k + 1)


# for any given model, infer parameters with varying data size
# return a list that contains trace for data size
def varyDataSize(model_name, min_len, dir_, sampler_param, isEv=False):
    # Turn off sampler information
    logger = logging.getLogger("pymc3")
    logger.setLevel(logging.ERROR)

    trace_lis = []

    for i, lens in enumerate(min_len):
        dx, dy, dt, track_info, lookup_, track_id = loadRawMinData(dir_, min_length=lens, isDx=True)

        # define mcmc object that contains model name, data and sampling parameters
        diffusion_model = GM.diffusion_model(model_name, dx, dy, dt, track_info,
                                             lookup_, sampler_param[0], sampler_param[1], sampler_param[2])

        model, param_ = diffusion_model.selectModel()

        with model:
            trace = pm.sample(diffusion_model.steps, chains=diffusion_model.chains, cores=diffusion_model.cores)

        #         np.savetxt('results/d_minlen_' + str(lens) + model_name + 'txt')
        # store trace into a list
        trace_lis.append(trace)
        print('Min length = %d sampled' % lens)

    return trace_lis


# fit a Gaussian curve to the sampled diffusion coefficient
def fitGaussian(trace, pixelToMeter, frameToSecond):
    d_distribution = np.empty((trace['D'].shape[1],))
    for i in range(trace['D'].shape[1]):
        d_distribution[i] = np.log10(trace['D'][:, i].mean() * pixelToMeter ** 2 / frameToSecond)
    mean, std = norm.fit(d_distribution)
    xmin, xmax = d_distribution.max(), d_distribution.min() - 0.5
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, std)

    return d_distribution, x, y


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
    return dx[start:start + (track_info[i + 1] - corrector)], dy[start:start + (track_info[i + 1] - corrector)],\
           dt[start:start + (track_info[i + 1] - corrector)]


# compute MC evidence for each trajectory
# also returns the mean for inferred value in each trajectory
def EvEachTraj(dir_, minlen, model_name, sample_param, verbose, isDx, pixelToMeter, frameToSecond, middle_start=None):

    dx, dy, dt, track_info, lookup_, track_id = loadRawMinData(dir_, min_length=minlen, isDx=isDx,
                                                               pixelToMeter=1, frameToSecond=1)
    Ev, Diff = np.empty((len(track_info),)), np.empty((len(track_info),))

    if isDx:
        corrector = 1
    else:
        corrector = 0

    start, end = 0, 0

    for i, length in enumerate(track_info):
        end = start + length - corrector
        temp_dx, temp_dy, temp_dt = dx[start:end], dy[start:end], dt[start:end]
        start = end

        if middle_start is not None:
            if i < middle_start[0]:
                continue

        # define mcmc object that contains model name, data and sampling parameters
        diffusion_model = GM.diffusion_model(model_name, temp_dx, temp_dy, temp_dt, track_info,
                                             lookup_, sample_param[0], sample_param[1], sample_param[2])

        # return model based on model_name, and a set of parameters name to be inferred
        model, param_ = diffusion_model.selectModel()

        with model:
            trace = pm.sample(diffusion_model.steps, chains=diffusion_model.chains, cores=diffusion_model.cores,
                              progressbar=verbose)

        # compute MC evidence using sampled points and model likelihood
        skip = diffusion_model.chains
        theta1 = np.exp(trace['logD'])[0:-1:skip].reshape(len(trace['logD'][0:-1:skip]), 1)
        theta2 = trace['Ux'][0:-1:skip].reshape(len(trace['Ux'][0:-1:skip]), 1)
        theta3 = trace['Uy'][0:-1:skip].reshape(len(trace['Uy'][0:-1:skip]), 1)
        theta1 = np.hstack((theta1, theta2))
        theta1 = np.hstack((theta1, theta3))

        logp = trace.model_logp[0:-1:skip]
        logEA, dEA = MCevidence(theta1, logp)

        Ev[i], Diff[i] = logEA, np.exp(trace['logD'].mean())*pixelToMeter**2/frameToSecond

        if middle_start is not None:
            np.savetxt('new_data/' + model_name + str(middle_start[1]) + 'th middle start MC evidence.txt', Ev[middle_start[0]:i + 1])
            np.savetxt('new_data/' + model_name + str(middle_start[1]) + 'th middle start Mean diffusion.txt', Diff[middle_start[0]:i + 1])
        else:
            np.savetxt('new_data/' + model_name + 'MC evidence.txt', Ev[0:i+1])
            np.savetxt('new_data/' + model_name + 'Mean diffusion.txt', Diff[0:i+1])

        if i % 50 == 0 and i is not 0:
            print('%d trajectories computed' % i)

    return Ev, Diff


# compute number of times that dx data changes sign
def crossZero(data):
    n_cross = 0
    for i in range(len(data)-1): 
        if data[i] > 0: 
            if data[i+1] < 0: 
                n_cross += 1
            else: 
                continue 
        else: 
            if data[i+1] > 0: 
                n_cross += 1
            else: 
                continue
    return n_cross


def crossMean(data): 
    n_cross = 0
    try: 
        mean = data.mean()
    except: 
        mean = np.array(data).mean()
        
    for i in range(len(data)-1): 
        if data[i] > mean: 
            if data[i+1] < mean: 
                n_cross += 1
            else: 
                continue 
        else: 
            if data[i+1] > mean: 
                n_cross += 1
            else: 
                continue
    return n_cross    


# model comparison by posterior predective check
def posterPredictive(trace, model, data, metric):
    ppc = pm.sample_posterior_predictive(trace, model=model)
    if metric == 'mean':
        hist_x = [likex.mean() for likex in ppc[model.observed_RVs[0].name]]
        hist_y = [likey.mean() for likey in ppc[model.observed_RVs[1].name]]
        data_statx = data[0].mean()
        data_staty = data[1].mean()
    elif metric == 'std':
        hist_x = [likex.std() for likex in ppc[model.observed_RVs[0].name]]
        hist_y = [likey.std() for likey in ppc[model.observed_RVs[1].name]]
        data_statx = data[0].std()
        data_staty = data[1].std()
    elif metric == 'max':
        hist_x = [likex.max() for likex in ppc[model.observed_RVs[0].name]]
        hist_y = [likey.max() for likey in ppc[model.observed_RVs[1].name]]
        data_statx = data[0].max()
        data_staty = data[1].max()
    elif metric == 'changesign':
        hist_x = [crossZero(likex) for likex in ppc[model.observed_RVs[0].name]]
        hist_y = [crossZero(likey) for likey in ppc[model.observed_RVs[1].name]]
        data_statx = crossZero(data[0])
        data_staty = crossZero(data[1])
    elif metric == 'crossmean':
        hist_x = [crossMean(likex) for likex in ppc[model.observed_RVs[0].name]]
        hist_y = [crossMean(likey) for likey in ppc[model.observed_RVs[1].name]]
        data_statx = crossMean(data[0])
        data_staty = crossMean(data[1])
    else:
        print('metric not implemented')
        return None

    count = 0
    for j, i in enumerate(hist_x):
        if i >= data_statx:
            count += 1
    precentToRightX = count * 100 / (j+1)

    count = 0
    for j, i in enumerate(hist_y):
        if i >= data_staty:
            count += 1
    precentToRightY = count * 100 / (j + 1)

    return hist_x, hist_y, precentToRightX, precentToRightY


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
    

def manyplots(row, column, r2, min_indx, filename, min_length=None): 
    """
    plots multiple trajecotries according to inout index
    """
    count = 0
    outer_break = False
    fig, axes = plt.subplots(row, column, figsize=(15, 15*(row/column)))
    
#     # calcuate max interval for each trajectory
#     max_interval = computeInterval(min_indx)
    
    for i in range(row): 
        if outer_break: 
            break
        for j in range(column): 
            if count == len(r2): 
                outer_break = True
                break
            sx, sy, st = loadSelectTraj(x, y, t, track_info, min_indx[count], False)
            if r2[count] >= 0.85: 
                col = 'r'
            else: 
                col = 'b'
#             if max_interval[count] > 3: 
#                 style = '.--'
#             else: 
#                 style = '.-'
            if min_length is None: 
                axes[i, j].plot(sx, sy,'.-', c=col)
            else: 
                axes[i, j].plot(sx, sy,'.-', c=col, label='logD = ' + str(min_length[count])[:4])
                axes[i, j].legend()
            axes[i, j].set_title(str(min_indx[count]))
            count += 1
    plt.savefig(filename)
    
    
def inCircle(center, radius, pos): 
    """
    Return True if the input point lies within the specified circle 
    """    
    
    x_max = np.sqrt(radius**2 - (pos[1]-center[1])**2) + center[0]
    x_min = -np.sqrt(radius**2 - (pos[1]-center[1])**2) + center[0]
    if pos[0] < x_min or pos[0] > x_max: 
        return False
    
    y_max = np.sqrt(radius**2 - (pos[0]-center[0])**2) + center[1]
    y_min = -np.sqrt(radius**2 - (pos[0]-center[0])**2) + center[1]
    if pos[1] < y_min or pos[1] > y_max: 
        return False
    
    return True


def findTetheredRadius(sx, sy, step_size=0.5): 
    """
    To find the radius of motion for those 'confined'/'tethered' trajectories
    """
    
    # detect and remove outliers in the data
    zscore_x, zscore_y = np.abs(stats.zscore(sx)), np.abs(stats.zscore(sy)) 
    outlier_x = np.where(zscore_x >= 2.5)[0]
    outlier_y = np.where(zscore_y >= 2.5)[0]
    all_outlier = list((set(outlier_x).union(set(outlier_y))))
    sx, sy = np.delete(sx, all_outlier), np.delete(sy, all_outlier)

    # find center of modified dataset
    center = [sx.mean(), sy.mean()]

    # find the circle that will enclose all data points
    r_range = np.arange(0.5, 20, step_size)
    for radius in r_range: 
        full_radius = True
        for pos in range(len(sx)): 
            if inCircle(center, radius, [sx[pos], sy[pos]]) != True: 
                full_radius = False
                break
        if full_radius: 
            break
                
    return center, radius


def upperCircle(x, radius, center): 
    """
    poor man version of circle drawing 
    """
    temp = radius**2 - (x-center[0])**2
    neg = np.where(temp < 0)[0]
    temp[neg] = np.nan
    return np.sqrt(temp) + center[1]


def lowerCircle(x, radius, center): 
    """
    poor man version of circle drawing 
    """
    temp = radius**2 - (x-center[0])**2
    neg = np.where(temp < 0)[0]
    temp[neg] = np.nan
    return -np.sqrt(temp) + center[1]


def compareEv(Ev, dtype, order): 
    """
    Ev has shape n_model x n_traj; compute ratio for each model for each traj
    Return precentage array ordered based on one model
    """
    n_model, n_traj = Ev.shape[0], Ev.shape[1]
    tot_precent = np.empty((n_model, n_traj))
    for i in range(n_traj): 
        max_evidence = np.max(Ev[:, i])
        tot_evidence = np.sum(np.exp(Ev[:, i] - max_evidence))
        for j in range(n_model): 
            tot_precent[j, i] = (np.exp(Ev[j, i] - max_evidence)) / tot_evidence
           
    precent_list = []        
    for i in range(n_model): 
        precent_list.append(tot_precent[i, :])
    del tot_precent
    
    return sort_by_entry(precent_list, dtype, order)


def computeZscore(len_specific_indx, mode='max'): 
    """
    Compute and return Z score for the crossover times for each input track
    """
    
    n_test = 1000
    D_test = 1
    interval_holder = np.empty((len(len_specific_indx), ))
    
    for i, track_idx in enumerate(len_specific_indx): 
        # calculate actual sample crossover
        temp_cross_x, temp_cross_y = np.empty((n_test, )), np.empty((n_test, ))
        sx, sy, st = loadSelectTraj(dx, dy, dt, track_info, track_idx, True)
        sample_crossmean_x = crossMean(sx)
        sample_crossmean_y = crossMean(sy) 

        # generate expected crossover with sample mean
        for j in range(n_test):
            # x direction
            temp_dx = np.random.normal(sx.mean(), np.sqrt(2*D_test*st))
            temp_cross_x[j] = utils.crossMean(temp_dx) 
            
            # y direction
            temp_dy = np.random.normal(sy.mean(), np.sqrt(2*D_test*st))
            temp_cross_y[j] = utils.crossMean(temp_dy) 
            
        # compare expected crossover with sample crossover
        temp_mean_x, temp_std_x = temp_cross_x.mean(), temp_cross_x.std()
        dif_x = np.abs(sample_crossmean_x - temp_mean_x)
        interval_x = dif_x/temp_std_x
        
        temp_mean_y, temp_std_y = temp_cross_y.mean(), temp_cross_y.std()
        dif_y = np.abs(sample_crossmean_y - temp_mean_y)
        interval_y = dif_y/temp_std_y
        
        if mode == 'max': 
            interval_holder[i] = max(interval_x, interval_y)
        elif mode == 'avg': 
            interval_holder[i] = (interval_x + interval_y) / 2
        else: 
            print('mode not implmentted')
            return None
    
    return interval_holder


def getLIndex(nd): 
    """
    Return L shape index that help construct the diff_cov
    """
    
    n_each = 2*nd - 1
    current_n = nd
    start = 0
    indx_lis = []
    for i in range(nd): 
        indx_lis.append(np.arange(start, current_n).tolist())
        current_n = current_n + i
        indx_lis[i].append(current_n)
        start = current_n + 1
        for j in range((n_each - len(indx_lis[i]))): 
            current_n = current_n + nd
            indx_lis[i].append(current_n)
        current_n = start + nd - i -1
        n_each = n_each - 2
        if i < nd-1: 
            indx_lis[i] = np.array(indx_lis[i])
    temp = indx_lis[-1].pop(-1)
    indx_lis[-1] = np.array(indx_lis[-1])
    return indx_lis

def diffusing_covariance_mt(dt): 
    """
    Build coefficient matrix for diffusion cov, with varying dt 
    """
    
    n_times = len(dt) + 1
    t = np.insert(np.cumsum(dt), 0, 0)
    mat1 = np.zeros((n_times**2, ))
    L_indx = getLIndex(n_times)
    for i in range(n_times): 
        mat1[L_indx[i]] = t[i]
    mat1 = mat1.reshape((n_times, n_times))
    
    return mat1

def diffusing_covariance(n_times, sigma0, noise, logD):
    """
    Build coefficient matrix for diffusion cov, with constant dt 
    """
    
    i, j = np.meshgrid(np.arange(n_times + 1), np.arange(n_times + 1))
    Sigma1 = np.minimum(i, j) * (2 * np.exp(logD)) + sigma0 ** 2 + noise ** 2 * np.eye(n_times + 1)

    # Zero mean vector
    mu1 = np.zeros(n_times + 1)

    return mu1, Sigma1


def removeOutLiar(sx, sy, st):
    """detect and remove outliers in the data"""
    
    zscore_x, zscore_y = np.abs(stats.zscore(sx)), np.abs(stats.zscore(sy)) 
    outlier_x = np.where(zscore_x >= 2.5)[0]
    outlier_y = np.where(zscore_y >= 2.5)[0]
    all_outlier = list((set(outlier_x).union(set(outlier_y))))
    sx, sy = np.delete(sx, all_outlier), np.delete(sy, all_outlier)
    st = np.delete(st, all_outlier)
    return sx, sy, st


def twoSummaryStats(index, track_info, sigma0=100, mu0=100): 
    """ Compute summary statistics for stick 
    and Diffusion models for all input tracks """
    
    n_tracks = len(index)
    sigma0 = 100
    mume_lis, sigme_lis = np.empty((n_tracks, )), np.empty((n_tracks, ))
    mud_lis, sigd_lis = np.empty((n_tracks, )), np.empty((n_tracks, ))
    mume_lis_dif, sigme_lis_dif = np.empty((n_tracks, )), np.empty((n_tracks, ))

    for i, index in enumerate(index): 

        # load single track
        sx, sy, st = loadSelectTraj(x, y, t, track_info, index, False)
        sx, sy, st = removeOutLiar(sx, sy, st)
        sdt = st[1:] - st[:-1]
        n_times = len(sdt)

        # compute stick summary statistics
        mat2 = np.eye(n_times+1) 

        normal_approx = pm.Model()
        with normal_approx: 

            logMe = pm.Normal('logMe', mu=0, sd=1)

            Sigma0 = sigma0**2 + tt.exp(logMe)**2 * mat2

            like_x = pm.MvNormal('like_x', mu=mu0, cov=Sigma0, observed=sx)
            like_y = pm.MvNormal('like_y', mu=mu0, cov=Sigma0, observed=sy)

            map_estimate = pm.find_MAP(model=normal_approx)
            hess_estimate = pm.find_hessian(map_estimate)       

        mume = map_estimate['logMe']
        cov = np.sqrt(la.inv(hess_estimate))
        sigme = cov[0,0]
        mume_lis[i], sigme_lis[i] = mume, sigme

        # compute diffusion summary statistics
        mat1 = utils.diffusing_covariance_mt(sdt)

        normal_approx = pm.Model()    
        with normal_approx: 

            logD = pm.Normal('logD', mu=0, sd=1)
            logMe = pm.Normal('logMe', mu=0, sd=1)

            Sigma1 = mat1 * (2 * tt.exp(logD)) + sigma0**2 + tt.exp(logMe)**2 * np.eye(n_times+1)

            like_x = pm.MvNormal('like_x', mu=mu0, cov=Sigma1, observed=sx)
            like_y = pm.MvNormal('like_y', mu=mu0, cov=Sigma1, observed=sy)  

            map_estimate = pm.find_MAP(model=normal_approx)
            hess_estimate = pm.find_hessian(map_estimate)       

        mud, mume = map_estimate['logD'], map_estimate['logMe']
        cov = np.sqrt(la.inv(hess_estimate))
        sigd, sigme = cov[0,0], cov[1,1]

        mud_lis[i], sigd_lis[i] = mud, sigd
        mume_lis_dif[i], sigme_lis_dif[i] = mume, sigme

        if i % 10 == 0 and i is not 0: 
            print('%d tracks done, %d tracks remain' %(i, n_tracks-i))
            
    return mume_lis, sigme_lis, mud_lis, sigd_lis, mume_lis_dif, sigme_lis_dif

############ 
## Start Kiran edits 
############ 

import matplotlib.pyplot as plt 
import seaborn as sns 
from decimal import Decimal 

seed = 4444 

## must do this because of the way random seeds are generated in numpy (sequentially random) 
def base_brownian_D(init_pos, t, D): 
    x0, y0 = init_pos 
    x = [x0] 
    y = [y0]
    tau = np.diff(t) 
    for idx in range(len(tau)):  
        x.append((x[-1] + np.random.normal(scale=np.sqrt(2*D*tau), size=1))[0]) 
        y.append((y[-1] + np.random.normal(scale=np.sqrt(2*D*tau), size=1))[0]) 
    return np.vstack((x,y)) 

## must do this because of the way random seeds are generated in numpy (sequentially random) 
def base_HPW_D(init_pos, t, D, well_pos, lambda_): 
    x0, y0 = init_pos 
    x_c, y_c = well_pos 
    x = [x0] 
    y = [y0]
    tau = np.diff(t) 
    for idx in range(len(tau)):  
        mu_x = x_c + (x[-1] - x_c)*np.exp(-lambda_*tau) 
        sd_x = np.sqrt(D/lambda_ * (1. - np.exp(-2.*lambda_*tau))) 
        x.append((np.random.normal(loc=mu_x, scale=sd_x, size=1))[0]) 
        
        mu_y = y_c + (y[-1] - y_c)*np.exp(-lambda_*tau) 
        sd_y = np.sqrt(D/lambda_ * (1. - np.exp(-2.*lambda_*tau))) 
        y.append((np.random.normal(loc=mu_y, scale=sd_y, size=1))[0]) 
    return np.vstack((x,y)) 

def augment_with_noise(traj, Me=None): 
    if not Me == None: 
        for idx in range(len(traj[0])): 
            traj[0, idx] += np.random.normal(scale=Me, size=1) 
            traj[1, idx] += np.random.normal(scale=Me, size=1) 
    else: 
        pass 
    
    return traj 

def plot_2D_trajectory(traj): 
    plt.plot(traj[0], traj[1], 'k.-') 
    # Mark the start and end points.
    plt.plot(traj[0,0], traj[1,0], 'go')
    plt.plot(traj[0,-1], traj[1,-1], 'ro') 
    plt.axis('equal') 
    # plt.legend(fontsize=12) 
    plt.show() 
    pass 

def plot_test_Me(init_pos, t_, D, Me_array): 
    np.random.seed(seed) 

    plt.figure(figsize=(8,6))
    baseBD = base_brownian_D(init_pos, t_, D) 
    res = base_brownian_D(init_pos, t_, D)  
    res = augment_with_noise(baseBD, Me=None) 
    plt.plot(res[0], res[1], 'k.-', label=r'$M_e = 0.$') 
    # Mark the start and end points.
    plt.plot(res[0,0], res[1,0], 'go')
    plt.plot(res[0,-1], res[1,-1], 'ro') 

    colours = ['g', 'r', 'b', 'm']
    for idx, Me_val in enumerate(Me_array): 
        res = augment_with_noise(baseBD, Me=Me_val) 
        if len(Me_array) == 4:     
            plt.plot(res[0], res[1], '.-', c=colours[idx], label=r'$M_e =$'+str(round(Me_val, 2)), alpha=0.3) 
        else: 
            plt.plot(res[0], res[1], '.-', label=r'$M_e =$'+str(round(Me_val, 2)), alpha=0.3) 
    plt.axis('equal') 
    plt.legend(fontsize=12) 
    plt.show() 
    
    pass 

def plot_test_lambda_single(init_pos, t_, D, well_pos, lambda_val, baseBD): 
    np.random.seed(seed) 
    plt.figure(figsize=(8,6))
    baseHPW = base_HPW_D(init_pos, t_, D, well_pos, lambda_val) 
    plt.plot(baseHPW[0], baseHPW[1], 'b.-', label=r'HPW $\lambda$ ='+str(round(lambda_val, 2)), alpha=0.5)  
    plt.plot(baseBD[0], baseBD[1], 'r.-', label='BD', alpha=0.5) 
    plt.axvline(x=well_pos[0], c='k', ls='--', alpha=0.3, label='HPW center') 
    plt.axhline(y=well_pos[1], c='k', ls='--', alpha=0.3) 
    plt.axis('equal') 
    plt.legend(fontsize=12)
    pass 

def plot_test_lambda(init_pos, t_, D, well_pos, lambda_array): 

    plt.figure(figsize=(8,6)) 
    for idx, lambda_val in enumerate(lambda_array): 
        np.random.seed(seed)
        lambda_ = lambda_array[idx] 
        baseHPW1 = base_HPW_D(init_pos, t_, D, well_pos, lambda_) 
        plt.plot(baseHPW1[0], baseHPW1[1], '.-', label=r'HPW $\lambda$ = '+str('%.1E' % Decimal(lambda_val)), alpha=0.5)  

    plt.axvline(x=well_pos[0], c='k', ls='--', alpha=0.3, label='HPW center') 
    plt.axhline(y=well_pos[1], c='k', ls='--', alpha=0.3) 
    plt.axis('equal') 
    plt.legend(fontsize=12)
    plt.show() 
    pass 

def generate_params(Jiayu_params=False): 

    if not Jiayu_params: 
        param = {} 
        param['mu_D'] = -3. 
        param['sigma_D'] = 2. 
        param['mu_lambda'] = -2. 
        param['sigma_lambda'] = 2. 
        param['mu_Me'] = -2. 
        param['sigma_Me'] = 2. 
    else: 
        param = {} 
        param['mu_D'] = -3. 
        param['sigma_D'] = 2. 
        param['mu_lambda'] = -2. # Jiayu likes stronger priors on lambda  
        param['sigma_lambda'] = 1. ## 
        param['mu_Me'] = -2. 
        param['sigma_Me'] = 2.  

    return param 

def plot_priors(param, parent=None, save=False): 
    priors = [('D', pm.Lognormal.dist(mu=param['mu_D'], sd=param['sigma_D'])), 
         ('lambda', pm.Lognormal.dist(mu=param['mu_lambda'], sd=param['sigma_lambda'])), 
         ('Me', pm.Lognormal.dist(mu=param['mu_Me'], sd=param['sigma_Me']))] 

    ranges = [np.logspace(param['mu_D'] - 3.*param['sigma_D'], param['mu_D'] + 3.*param['sigma_D'], 201), 
             np.logspace(param['mu_lambda'] - 1.*param['sigma_lambda'], param['mu_lambda'] + 5.*param['sigma_lambda'], 201), 
             np.logspace(param['mu_Me'] - 1.*param['sigma_Me'], param['mu_Me'] + 5.*param['sigma_Me'], 201)]

    stats = [[param['mu_D'], param['sigma_D']], [param['mu_lambda'], param['sigma_lambda']], 
             [param['mu_Me'], param['sigma_Me']]] 

    for i, prior in enumerate(priors): 
        plt.figure() 
        sns.distplot(prior[1].random(size=50000), bins=ranges[i], kde=False, ) ## count data 
    #     plt.plot(ranges[i], np.exp(prior[1].logp(ranges[i]).eval()), label=str(prior[0]), lw=3) ## pdf 
        plt.semilogx() 
        plt.title(prior[0], fontsize=18)
        plt.axvline(np.exp(stats[i][0]), c='k', alpha=0.4, label='geometric mean') 
        plt.axvline(np.exp(stats[i][0] + stats[i][1]), c='r', alpha=0.4, label='geometric std dev') 
        plt.axvline(np.exp(stats[i][0] - stats[i][1]), c='r', alpha=0.4)
        plt.legend(fontsize=12) 
        if save: plt.savefig(parent + '/_priors'+str(i)+'.png')
    plt.tight_layout()  
    plt.show() 
    pass 

def plot_prior_posterior(param, trace, true_vals=None): 
    # plt.figure()
    # sns.distplot(trace['D'], bins=np.logspace(-4, 2, 101), 
    #              kde=False, label='posterior') 
    # plt.hist(trace['D'], density=False, bins=np.logspace(-4, 2, 101), alpha=0.4) 
    # plt.semilogx() 
    # plt.show() 

    plt.figure(1, figsize=(8,8)) 
    
    bins_D = np.logspace(param['mu_D'] - 3.*param['sigma_D'], param['mu_D'] + 3.*param['sigma_D'], 51)  


    plt.subplot(211)
    # plt.subplot(311) 

    sns.distplot(pm.Lognormal.dist(param['mu_D'], param['sigma_D']).random(size=len(trace['D'])), 
                 bins=bins_D, kde=False, label='prior') 
    sns.distplot(trace['D'], bins=np.logspace(np.log10(min(trace['D'])), np.log10(max(trace['D'])), 21), 
                 kde=False, label='posterior') 
    # sns.distplot(pm.Lognormal.dist(param['mu_D'], param['sigma_D']).random(size=len(trace)), 
    #              bins=np.logspace(np.log(min(trace['D'])), np.log(max(trace['D'])), 51), kde=False, label='prior') 
    # sns.distplot(trace['D'], bins=np.logspace(np.log(min(trace['D'])), np.log(max(trace['D'])), 51), kde=False, label='posterior') 
    plt.semilogx() 
    plt.legend() 
    plt.xlim(1e-5, 1e3)
    plt.title('D', fontsize=18) 
    if not true_vals == None: plt.axvline(true_vals[0], c='r', alpha=0.8) 

    bins_lam = np.logspace(param['mu_lambda'] - 1.*param['sigma_lambda'], param['mu_lambda'] + 5.*param['sigma_lambda'], 51)
    plt.subplot(212)
    # plt.subplot(312) 
    sns.distplot(pm.Lognormal.dist(param['mu_lambda'], param['sigma_lambda']).random(size=len(trace['lam'])), 
                 bins=bins_lam, kde=False, label='prior') 
    sns.distplot(trace['lam'], bins=np.logspace(np.log10(min(trace['lam'])), np.log10(max(trace['lam'])), 21), 
                 kde=False, label='posterior') 
    # sns.distplot(pm.Lognormal.dist(param['mu_lambda'], param['sigma_lambda']).random(size=len(trace)), 
    #              bins=np.logspace(np.log(min(trace['D'])), np.log(max(trace['D'])), 51), kde=False, label='prior') 
    # sns.distplot(trace['lam'], bins=np.logspace(np.log(min(trace['lam'])), np.log(max(trace['lam'])), 51), 
    #              kde=False, label='posterior') 
    plt.semilogx() 
    plt.legend() 
    plt.xlim(1e-15, 1e5)
    plt.title('lambda', fontsize=18) 
    if not true_vals == None: plt.axvline(true_vals[1], c='r', alpha=0.8) 

    plt.tight_layout();  
    plt.show() 

    # try: 
    #     bins_Me = np.logspace(param['mu_Me'] - 1.*param['sigma_Me'], param['mu_Me'] + 5.*param['sigma_Me'], 51)
    #     plt.subplot(313) 
    #     sns.distplot(pm.Lognormal.dist(param['mu_Me'], param['sigma_Me']).random(size=len(trace['me'])), 
    #                  bins=bins_Me, kde=False, label='prior') 
    #     sns.distplot(trace['me'], bins=np.logspace(np.log10(min(trace['me'])), np.log10(max(trace['me'])), 21), 
    #                  kde=False, label='posterior') 
    #     # sns.distplot(pm.Lognormal.dist(param['mu_lambda'], param['sigma_lambda']).random(size=len(trace)), 
    #     #              bins=np.logspace(np.log(min(trace['D'])), np.log(max(trace['D'])), 51), kde=False, label='prior') 
    #     # sns.distplot(trace['lam'], bins=np.logspace(np.log(min(trace['lam'])), np.log(max(trace['lam'])), 51), 
    #     #              kde=False, label='posterior') 
    #     plt.semilogx() 
    #     plt.legend() 
    #     plt.xlim(1e-2, 1e0)
    #     plt.title('Me', fontsize=18) 
    # except: 
    #     pass   


def plot_prior_posterior_comparison(param, trace1, trace2, true_vals=None): 
    # plt.figure()
    # sns.distplot(trace['D'], bins=np.logspace(-4, 2, 101), 
    #              kde=False, label='posterior') 
    # plt.hist(trace['D'], density=False, bins=np.logspace(-4, 2, 101), alpha=0.4) 
    # plt.semilogx() 
    # plt.show() 

    plt.figure(1, figsize=(8,8)) 
    
    bins_D = np.logspace(param['mu_D'] - 3.*param['sigma_D'], param['mu_D'] + 3.*param['sigma_D'], 51)  


    # plt.subplot(211)
    plt.subplot(311) 

    sns.distplot(pm.Lognormal.dist(param['mu_D'], param['sigma_D']).random(size=len(trace1['D'])), 
                 bins=bins_D, kde=False, label='prior') 
    sns.distplot(trace1['D'], bins=np.logspace(np.log10(min(trace1['D'])), np.log10(max(trace1['D'])), 21), 
                 kde=False, label='posterior 1') 
    sns.distplot(trace2['D'], bins=np.logspace(np.log10(min(trace2['D'])), np.log10(max(trace2['D'])), 21), 
                 kde=False, label='posterior 1') 
    # sns.distplot(pm.Lognormal.dist(param['mu_D'], param['sigma_D']).random(size=len(trace)), 
    #              bins=np.logspace(np.log(min(trace['D'])), np.log(max(trace['D'])), 51), kde=False, label='prior') 
    # sns.distplot(trace['D'], bins=np.logspace(np.log(min(trace['D'])), np.log(max(trace['D'])), 51), kde=False, label='posterior') 
    plt.semilogx() 
    plt.legend() 
    plt.xlim(1e-5, 1e3)
    plt.title('D', fontsize=18) 
    if not true_vals == None: plt.axvline(true_vals[0], c='r', alpha=0.8) 

    try: 
        bins_lam = np.logspace(param['mu_lambda'] - 1.*param['sigma_lambda'], param['mu_lambda'] + 5.*param['sigma_lambda'], 51)
        # plt.subplot(212)
        plt.subplot(312) 
        sns.distplot(pm.Lognormal.dist(param['mu_lambda'], param['sigma_lambda']).random(size=len(trace1['lam'])), 
                     bins=bins_lam, kde=False, label='prior') 
        sns.distplot(trace1['lam'], bins=np.logspace(np.log10(min(trace1['lam'])), np.log10(max(trace1['lam'])), 21), 
                     kde=False, label='posterior 1') 
        sns.distplot(trace2['lam'], bins=np.logspace(np.log10(min(trace2['lam'])), np.log10(max(trace2['lam'])), 21), 
                     kde=False, label='posterior 2') 
        # sns.distplot(pm.Lognormal.dist(param['mu_lambda'], param['sigma_lambda']).random(size=len(trace)), 
        #              bins=np.logspace(np.log(min(trace['D'])), np.log(max(trace['D'])), 51), kde=False, label='prior') 
        # sns.distplot(trace['lam'], bins=np.logspace(np.log(min(trace['lam'])), np.log(max(trace['lam'])), 51), 
        #              kde=False, label='posterior') 
        plt.semilogx() 
        plt.legend() 
        plt.xlim(1e-15, 1e5)
        plt.title('lambda', fontsize=18) 
        if not true_vals == None: plt.axvline(true_vals[1], c='r', alpha=0.8) 
    except: 
        pass 

    bins_Me = np.logspace(param['mu_Me'] - 1.*param['sigma_Me'], param['mu_Me'] + 5.*param['sigma_Me'], 51)
    plt.subplot(313) 
    sns.distplot(pm.Lognormal.dist(param['mu_Me'], param['sigma_Me']).random(size=len(trace2['me'])), 
                 bins=bins_Me, kde=False, label='prior') 
    try: 
        sns.distplot(trace1['me'], bins=np.logspace(np.log10(min(trace1['me'])), np.log10(max(trace1['me'])), 21), 
                 kde=False, label='posterior 1') 
    except: 
        pass 
    sns.distplot(trace2['me'], bins=np.logspace(np.log10(min(trace2['me'])), np.log10(max(trace2['me'])), 21), 
                 kde=False, label='posterior 2') 
    # sns.distplot(pm.Lognormal.dist(param['mu_lambda'], param['sigma_lambda']).random(size=len(trace)), 
    #              bins=np.logspace(np.log(min(trace['D'])), np.log(max(trace['D'])), 51), kde=False, label='prior') 
    # sns.distplot(trace['lam'], bins=np.logspace(np.log(min(trace['lam'])), np.log(max(trace['lam'])), 51), 
    #              kde=False, label='posterior') 
    plt.semilogx() 
    plt.legend() 
    plt.xlim(1e-4, 1e2)
    plt.title('Me', fontsize=18) 

    plt.tight_layout() 
    plt.show() 
    pass 

    return  

def plot_prior_posterior_validation(param, trace, true_vals=None, parent=None, save=False): 
    # plt.figure()
    # sns.distplot(trace['D'], bins=np.logspace(-4, 2, 101), 
    #              kde=False, label='posterior') 
    # plt.hist(trace['D'], density=False, bins=np.logspace(-4, 2, 101), alpha=0.4) 
    # plt.semilogx() 
    # plt.show() 

    plt.figure(1, figsize=(8,8)) 
    
    bins_D = np.logspace(param['mu_D'] - 3.*param['sigma_D'], param['mu_D'] + 3.*param['sigma_D'], 51)  


    # plt.subplot(211)
    plt.subplot(311) 

    sns.distplot(pm.Lognormal.dist(param['mu_D'], param['sigma_D']).random(size=len(trace['D'])), 
                 bins=bins_D, kde=False, label='prior') 
    sns.distplot(trace['D'], bins=np.logspace(np.log10(min(trace['D'])), np.log10(max(trace['D'])), 21), 
                 kde=False, label='posterior') 
    # sns.distplot(pm.Lognormal.dist(param['mu_D'], param['sigma_D']).random(size=len(trace)), 
    #              bins=np.logspace(np.log(min(trace['D'])), np.log(max(trace['D'])), 51), kde=False, label='prior') 
    # sns.distplot(trace['D'], bins=np.logspace(np.log(min(trace['D'])), np.log(max(trace['D'])), 51), kde=False, label='posterior') 
    plt.semilogx() 
    plt.legend() 
    plt.xlim(1e-5, 1e3)
    plt.title('D', fontsize=18) 
    if not true_vals == None: plt.axvline(true_vals[0], c='r', alpha=0.8) 

    try: 
        bins_lam = np.logspace(param['mu_lambda'] - 1.*param['sigma_lambda'], param['mu_lambda'] + 5.*param['sigma_lambda'], 51)
        # plt.subplot(212)
        plt.subplot(312) 
        sns.distplot(pm.Lognormal.dist(param['mu_lambda'], param['sigma_lambda']).random(size=len(trace['lam'])), 
                     bins=bins_lam, kde=False, label='prior') 
        sns.distplot(trace['lam'], bins=np.logspace(np.log10(min(trace['lam'])), np.log10(max(trace['lam'])), 21), 
                     kde=False, label='posterior') 
        # sns.distplot(pm.Lognormal.dist(param['mu_lambda'], param['sigma_lambda']).random(size=len(trace)), 
        #              bins=np.logspace(np.log(min(trace['D'])), np.log(max(trace['D'])), 51), kde=False, label='prior') 
        # sns.distplot(trace['lam'], bins=np.logspace(np.log(min(trace['lam'])), np.log(max(trace['lam'])), 51), 
        #              kde=False, label='posterior') 
        plt.semilogx() 
        plt.legend() 
        plt.xlim(1e-15, 1e5)
        plt.title('lambda', fontsize=18) 
        if not true_vals == None: plt.axvline(true_vals[1], c='r', alpha=0.8) 
    except: 
        pass

    try: 
        bins_Me = np.logspace(param['mu_Me'] - 1.*param['sigma_Me'], param['mu_Me'] + 5.*param['sigma_Me'], 51)
        plt.subplot(313) 
        sns.distplot(pm.Lognormal.dist(param['mu_Me'], param['sigma_Me']).random(size=len(trace['me'])), 
                     bins=bins_Me, kde=False, label='prior') 
        sns.distplot(trace['me'], bins=np.logspace(np.log10(min(trace['me'])), np.log10(max(trace['me'])), 21), 
                     kde=False, label='posterior') 
        # sns.distplot(pm.Lognormal.dist(param['mu_lambda'], param['sigma_lambda']).random(size=len(trace)), 
        #              bins=np.logspace(np.log(min(trace['D'])), np.log(max(trace['D'])), 51), kde=False, label='prior') 
        # sns.distplot(trace['lam'], bins=np.logspace(np.log(min(trace['lam'])), np.log(max(trace['lam'])), 51), 
        #              kde=False, label='posterior') 
        plt.semilogx() 
        plt.legend() 
        plt.xlim(1e-4, 1e2) 
        plt.title('Me', fontsize=18) 
    except: 
        pass 

    plt.tight_layout() 
    if save: plt.savefig(parent+'/_pripost.png', bbox_inches='tight') 
    plt.show() 
    pass 
    
    return  

def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) 

# funtion that construct the cov matrix for the bm-noise likelihood

def diffusing_covariance(n_times, sigma0, noise, diffusivity):
    # Build covariance matrix
    i,j = np.meshgrid(np.arange(n_times+1), np.arange(n_times+1))
    Sigma1 = np.minimum(i,j) * (2 * diffusivity) + sigma0**2 + noise**2 * np.eye(n_times+1)
    
    # Zero mean vector
    mu1 = np.zeros(n_times+1)
    
    return mu1, Sigma1 

def plot_diffusiveTracks(dir_, min_length, ind_array, save=False): 
    x, y, t, track_info, lookup, track_id = loadRawMinData(dir_, min_length=min_length, isDx=False) 
    dx, dy, dt, track_info2, lookup, track_id = loadRawMinData(dir_, min_length=min_length, isDx=True) 
    
    for idx, ind in enumerate(ind_array): 
        sdx, sdy, sdt = loadSelectTraj(dx, dy, dt, track_info2, ind, True) 
        sx, sy, st = loadSelectTraj(x, y, t, track_info, ind, False) 

        plt.figure()  
        plt.title(ind, fontsize=18) 
        plt.plot(sx, sy, '.-') 
        plt.xlabel('x [pix]', fontsize=14) 
        plt.ylabel('y [pix]', fontsize=14) 
        plt.show() 

    return 



def run_simpleBD_with_noise(dir_, param, min_length, ind, save=False): 
    parent = 'results_ind_' + str(ind) 
    createFolder(parent) 
    
    plot_priors(param, parent, save=save)  

    x, y, t, track_info, lookup, track_id = loadRawMinData(dir_, min_length=min_length, isDx=False) 
    dx, dy, dt, track_info2, lookup, track_id = loadRawMinData(dir_, min_length=min_length, isDx=True) 
    sdx, sdy, sdt = loadSelectTraj(dx, dy, dt, track_info2, ind, True) 
    sx, sy, st = loadSelectTraj(x, y, t, track_info, ind, False)
    
    plt.figure()  
    plt.plot(sx, sy, '.-'); 
    plt.xlabel('x [pix]', fontsize=14) 
    plt.ylabel('y [pix]', fontsize=14) 
    if save: plt.savefig(parent + '/_traj_' + str(ind) + '.png') 
    plt.show() 

    corr_y = (sy-sy.mean())
    corr_x = (sx-sx.mean())
    n_times = len(sx) - 1 

    sigma0 = 10. 

    model = pm.Model()

    with model: 
        D = pm.Lognormal('D', param['mu_D'], param['sigma_D']) 
        me = pm.Lognormal('me', param['mu_Me'], param['sigma_Me']) 
            
    #     D = pm.Lognormal('D', mu=-3, sd=2) 
    #     noise = pm.Lognormal('noise', mu=0, sd=0.5)
        
        mu0, sig0 = diffusing_covariance(n_times, sigma0, me, D) 
        
        like = pm.MvNormal('likelihood', mu=mu0, cov=sig0, observed=corr_x) + pm.MvNormal('like', mu=mu0, cov=sig0, observed=corr_y) 

    with model:
        trace = pm.sample(5000, tune=5000, chains=3, cores=1, target_accept=0.99) 

    pm.traceplot(trace)  
    if save: plt.savefig(parent + '/_trace_' + str(ind) + '.png', bbox_inches='tight') 

    plot_prior_posterior_validation(param, trace, None, parent, save=save)  

    np.savetxt(parent + '/_postsamp'+ str(ind) +'.tsv', np.vstack((trace['D'], trace['me'])).T, delimiter="\t") 

    return 

def removeOutLiar(sx, sy, st):
    """detect and remove outliers in the data"""

    zscore_x, zscore_y = np.abs(stats.zscore(sx)), np.abs(stats.zscore(sy))
    outlier_x = np.where(zscore_x >= 2.5)[0]
    outlier_y = np.where(zscore_y >= 2.5)[0]
    all_outlier = list((set(outlier_x).union(set(outlier_y))))
    sx, sy = np.delete(sx, all_outlier), np.delete(sy, all_outlier)
    st = np.delete(st, all_outlier)
    return sx, sy, st 

def run_HPWinference(dir_, param, file_stats, save=False): 
    root = 'enzymeBayes_results' 
    createFolder(root) 

    # plot_priors(param, parent, save=save) 

    x, y, t, track_info, lookup, track_id = file_stats 

    # correcting for outlier and center the tracks
    new_x, new_y, new_t = [], [], []
    new_dx, new_dy, new_dt = [], [], []
    new_lookup = []

    for i in range(len(track_info)): 
        parent = root + '/results_track_info_idx_' + str(i) 
        createFolder(parent)   

        sx, sy, st = loadSelectTraj(x, y, t, track_info, i, False)
        sx, sy, st = removeOutLiar(sx, sy, st)
        new_x.append((sx-sx.mean())[:-1])
        new_y.append((sy-sy.mean())[:-1])
        new_t.append((st-st.mean())[:-1])
        
        new_dx.append(sx[1:]-sx[:-1])
        new_dy.append(sy[1:]-sy[:-1])
        new_dt.append(st[1:]-st[:-1])
        
        new_lookup.append(i*np.ones((len(sx[1:]-sx[:-1]), ))) 


    # assemble the tracks into a long array for pymc3
    new_x = np.concatenate(new_x)
    new_y = np.concatenate(new_y)
    new_t = np.concatenate(new_t)

    new_dx = np.concatenate(new_dx)
    new_dy = np.concatenate(new_dy)
    new_dt = np.concatenate(new_dt)

    new_lookup = (np.concatenate(new_lookup)).astype('int') 

    model = pm.Model()

    with model: 
        
        D = pm.Lognormal('D', param['mu_D'], param['sigma_D'], shape=len(track_info))
        k = pm.Lognormal('lam', param['mu_lambda'], shape=len(track_info))
        D_, k_ = D[new_lookup], k[new_lookup]

        
        mean_x = (-(new_x)) * (1-tt.exp(-k_*new_dt))
        mean_y = (-(new_y)) * (1-tt.exp(-k_*new_dt))
        std = tt.sqrt(D_*(1-tt.exp(-2*k_*new_dt))/k_)

        like_x = pm.Normal('like_x', mu=mean_x, sd=std, observed=new_dx)
        like_y = pm.Normal('like_y', mu=mean_y, sd=std, observed=new_dy)
        
    # x, y, t, track_info, lookup, track_id = [file_stats] 

    # sx, sy, st = loadSelectTraj(x, y, t, track_info, ind, False) 
    # sx, sy, st = removeOutLiar(sx, sy, st) 
    
    # sdx = np.diff(sx); sdy = np.diff(sy); sdt = np.diff(st); 

    # corr_y = (sy-sy.mean())
    # corr_x = (sx-sx.mean())

    # with pm.Model() as model: 
    #     D = pm.Lognormal('D', param['mu_D'], param['sigma_D'])
    #     lamda = pm.Lognormal('lam', param['mu_lambda'], param['sigma_lambda']) 
        
    #     mean_x = (0. - corr_x[:-1])*(1-tt.exp(-lamda*sdt))
    #     mean_y = (0. - corr_y[:-1])*(1-tt.exp(-lamda*sdt))
    #     std = tt.sqrt(D*(1-tt.exp(-2*lamda*sdt))/lamda)
        
    #     like_x = pm.Normal('like_x', mu=mean_x, sd=std, observed=sdx)
    #     like_y = pm.Normal('like_y', mu=mean_y, sd=std, observed=sdy) 


    with model:
        trace = pm.sample(5000, tune=5000, chains=3, cores=1, target_accept=0.99) 

    pm.traceplot(trace)  
    if save: plt.savefig(parent + '/_trace_' + str(ind) + '.png', bbox_inches='tight') 

    # plot_prior_posterior_validation(param, trace, None, parent, save=save)  

    np.savetxt(parent + '/_postsamp'+ str(ind) +'.tsv', np.vstack((trace['D'], trace['lam'])), delimiter="\t") 

    return 



def generate_data(dict_): 
    init_pos, t_, D, well_pos, lambda_, Me = dict_ 
    
    ## BD model 
    if lambda_ == None and well_pos == None: 
        traj = base_brownian_D(init_pos, t_, D)  
    
    ## HPW model 
    else: 
        traj = base_HPW_D(init_pos, t_, D, well_pos, lambda_) 
    
    ## add measurement noise 
    if Me == None: 
        pass 
    else: 
        traj = augment_with_noise(traj, Me=Me) 
        
    return traj 