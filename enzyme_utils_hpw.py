import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import pymc3 as pm
import theano.tensor as tt 
import seaborn as sns 


def loadMin(df, minLength, frameToSecond, pixelToMeter, file_number):
    iMax = int(df["Trajectory"].iloc[-1])
    x_lis, y_lis, t_lis = [], [], []
    traj_track = []
    traj_id = []

    for i in range(1, iMax + 1):
        idx = df["Trajectory"] == float(i)
        if np.sum(idx) >= minLength:
            x_lis.append(df[idx]["x"].to_numpy() * pixelToMeter)
            y_lis.append(df[idx]["y"].to_numpy() * pixelToMeter)
            t_lis.append(df[idx]["Frame"].to_numpy() * frameToSecond)
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

    return total_var.reshape(len(total_var),)


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


def loadRawMinData(
    dir_, min_length=50, isDx=True, frameToSecond=1, pixelToMeter=1
):
    entries = Path(dir_)

    # obtain total number of trajectory files
    file_count = 0
    for entry in entries.iterdir():
        if entry.name[0] == ".":
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
        if entry.name[0] == ".":
            continue

        filename = dir_ + entry.name
        # used to locate the file numbering
        if file_index == 0:
            for i, s in enumerate(entry.name):
                if s == "-":
                    file_index = i + 4
                    break
        if entry.name[file_index + 3] == "-":
            file_number = entry.name[file_index: file_index + 5]
        else:
            file_number = entry.name[file_index: file_index + 3]
        df = pd.read_csv(filename)
        temp = loadMin(
            df, min_length, frameToSecond, pixelToMeter, file_number
        )

        if len(temp[0]) != 0:
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

    print(
        "Total %d files read; Total %d trajectories (length >= %d) loaded; Total %d data points"
        % (
            int(file_count),
            len(track_tot_),
            min_length,
            int(x_tot_arr.shape[0]),
        )
    )

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

    return (
        dx[start: start + (track_info[i + 1] - corrector)],
        dy[start: start + (track_info[i + 1] - corrector)],
        dt[start: start + (track_info[i + 1] - corrector)],
    )


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
        D_prior_ = pm.Lognormal("Dp", D_prior[0], D_prior[1])
        k_prior_ = pm.Lognormal("kp", k_prior[0], k_prior[1])

    prior_samples = pm.sample_prior_predictive(samples=4000, model=prior)

    Dp = prior_samples["Dp"]
    kp = prior_samples["kp"]

    # paramter inference
    model = pm.Model()
    with model:
        D = pm.Lognormal("D", D_prior[0], D_prior[1])
        k = pm.Lognormal("k", k_prior[0], k_prior[1])

        mean_x = (-(sx - sx.mean())[:-1]) * (1 - tt.exp(-k * sdt))
        mean_y = (-(sy - sy.mean())[:-1]) * (1 - tt.exp(-k * sdt))
        std = tt.sqrt(D * (1 - tt.exp(-2 * k * sdt)) / k)

        like_x = pm.Normal("like_x", mu=mean_x, sd=std, observed=sdx)
        like_y = pm.Normal("like_y", mu=mean_y, sd=std, observed=sdy)

    with model:
        trace = pm.sample(2000, tune=2000, chains=2, cores=2)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(sx, sy, label="Track: %d, length: %d" % (ind, len(sx)))
    axes[0].legend()

    axes[1].hist(
        np.log(trace["D"]),
        bins=bins,
        density=True,
        alpha=0.5,
        label="posterior",
    )
    axes[1].hist(
        np.log(Dp),
        bins=bins,
        density=True,
        alpha=0.5,
        label="prior: D(%d, %d)" % (D_prior[0], D_prior[1]),
    )
    axes[1].legend()

    axes[2].hist(
        np.log(trace["k"]),
        bins=bins,
        density=True,
        alpha=0.5,
        label="posterior",
    )
    axes[2].hist(
        np.log(kp),
        bins=bins,
        density=True,
        alpha=0.5,
        label="prior: k(%d, %d)" % (k_prior[0], k_prior[1]),
    )
    axes[2].legend()

    plt.show()

    return trace


def MAP_hpw(sx, sy, dt):
    lam = (
        np.log(
            np.sum(sy[1:] ** 2 + sx[1:] ** 2)
            / np.sum(sy[1:] * sy[:-1] + sx[1:] * sx[:-1])
        )
        / dt
    )

    I_ = 1 - np.exp(-2 * lam * dt)
    D = lam * (
        np.sum(
            (sy[1:] - sy[:-1] * np.exp(-lam * dt)) ** 2
            + (sx[1:] - sx[:-1] * np.exp(-lam * dt)) ** 2
        )
        / I_
        / 2
        / (len(sy) - 1)
    )

    return D


def MAP_bm(dx, dy, dt, a, b):
    alpha = len(dx) + a
    beta = np.sum((dx ** 2 + dy ** 2) / 4 / dt) + b
    return beta / (alpha - 1)


def manyplots_real(
    row, column, min_indx, track_info, stats, filename=None, min_length=None
):
    """
    plots multiple trajecotries according to inout index
    """
    count = 0
    outer_break = False
    fig, axes = plt.subplots(row, column, figsize=(15, 15 * (row / column)))

    x, y, t = stats

    #     # calcuate max interval for each trajectory
    #     max_interval = computeInterval(min_indx)
    for i in range(row):
        if outer_break:
            break
        for j in range(column):
            if count == len(min_indx):
                outer_break = True
                break
            sx, sy, st = loadSelectTraj(
                x, y, t, track_info, min_indx[count], False
            )
            #             sx, sy, st = removeOutLiar(sx, sy, st)

            if min_length is None:
                if row == 1:
                    axes[j].plot(sx, sy, ".-")
                    axes[j].set_title(str(min_indx[count]))
                else:
                    axes[i, j].plot(sx, sy, ".-")
                    axes[i, j].set_title(str(min_indx[count]))
            else:
                axes[i, j].plot(
                    sx, sy, ".-", label="logD = " + str(min_length[count])[:4]
                )
                axes[i, j].legend()
                axes[i, j].set_title(str(min_indx[count]))
            #             axes[i, j].set_xlim(sx.mean()-ran, sx.mean()+ran)
            #             axes[i, j].set_ylim(sy.mean()-ran, sy.mean()+ran)
            count += 1


#     plt.savefig(filename)


def removeAllOutliers(x, y, t, track_info):

    new_x, new_y, new_t = [], [], []
    new_dx, new_dy, new_dt = [], [], []
    new_track, new_lookup = [], []

    for i in range(len(track_info)):
        sx, sy, st = loadSelectTraj(x, y, t, track_info, i, False)
        sx, sy, st = removeOutLiar(sx, sy, st)
        new_x.append((sx - sx.mean()))
        new_y.append((sy - sy.mean()))
        new_t.append((st - st.mean()))

        new_dx.append(sx[1:] - sx[:-1])
        new_dy.append(sy[1:] - sy[:-1])
        new_dt.append(st[1:] - st[:-1])

        new_lookup.append(i * np.ones((len(sx[1:] - sx[:-1]),)))
        new_track.append(len(sx))

    new_x = np.concatenate(new_x)
    new_y = np.concatenate(new_y)
    new_t = np.concatenate(new_t)

    new_dx = np.concatenate(new_dx)
    new_dy = np.concatenate(new_dy)
    new_dt = np.concatenate(new_dt)

    new_lookup = (np.concatenate(new_lookup)).astype("int")

    return new_x, new_y, new_t, new_dx, new_dy, new_dt, new_lookup, new_track


def draw_semilog_posterior(D, alpha, beta, xpos, ypos, ax, max_):
    posterior = stats.invgamma.pdf(D, alpha, scale=beta)
    posterior /= posterior.max()
    hd_region = np.where(np.log10(posterior) >= -2)
    ax.plot(xpos, max_, ".", c="r", markersize=4)
    return ax.plot(
        posterior[hd_region] + xpos, np.log(D[hd_region]), c="b", alpha=0.2
    )


def draw_vertical_posterior_bm(stat, a, b, track_info):

    dx, dy, dt = stat

    map_estimator = np.zeros((len(track_info),))
    for i in range(len(track_info)):
        sdx, sdy, sdt = loadSelectTraj(dx, dy, dt, track_info, i, True)
        map_estimator[i] = MAP_bm(sdx, sdy, sdt, a, b)

    ind = [i for i in range(len(track_info))]
    dtype = [("log_d", float), ("index", int)]
    order = "log_d"
    sort_by_entry([map_estimator, ind], dtype, order)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    D = np.logspace(-2, 2, 500)
    for i in range(len(track_info)):
        sdx, sdy, sdt = loadSelectTraj(dx, dy, dt, track_info, ind[i], True)
        alpha = len(sdx) + a
        beta = np.sum((sdx ** 2 + sdy ** 2) / 4 / sdt) + b
        draw_semilog_posterior(
            D, alpha, beta, i, i, ax=ax[0], max_=np.log(map_estimator[i])
        )
    prior = stats.invgamma.pdf(D, a, scale=b)
    ax[0].plot(
        (prior * 20 + 0), np.log(D), c="k", lw=2, ls="--", label="prior"
    )
    ax[1].hist(np.log(map_estimator), bins=30, label="MAP estimates")
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel("Order track number")
    ax[0].set_ylabel("Ln(D (pixel^2/frame))")
    ax[1].set_xlabel("Ln(D (pixel^2/frame))")
    ax[1].set_ylabel("counts")


def base_HPW_D(init_pos, t, D, well_pos, lambda_):
    x0, y0 = init_pos
    x_c, y_c = well_pos
    x = [x0]
    y = [y0]
    tau = np.diff(t)
    for idx in range(len(t) - 1):
        mu_x = x_c + (x[-1] - x_c) * np.exp(-lambda_ * tau)
        sd_x = np.sqrt(D / lambda_ * (1.0 - np.exp(-2.0 * lambda_ * tau)))
        x.append((np.random.normal(loc=mu_x, scale=sd_x, size=1))[0])

        mu_y = y_c + (y[-1] - y_c) * np.exp(-lambda_ * tau)
        sd_y = np.sqrt(D / lambda_ * (1.0 - np.exp(-2.0 * lambda_ * tau)))
        y.append((np.random.normal(loc=mu_y, scale=sd_y, size=1))[0])
    return np.vstack((x, y))


def sticking_covariance(n_times, sigma0, noise):
    # Build covariance matrix
    Sigma0 = sigma0 ** 2 + noise ** 2 * np.eye(n_times + 1)

    # Zero mean vector
    mu0 = np.zeros(n_times + 1)

    return mu0, Sigma0


def simulate_stuck(me, n_times):

    sigma0 = 0  # standard deviation of the initial enzyme position
    mu1, Sigma1 = sticking_covariance(n_times, sigma0, me)
    x = np.random.multivariate_normal(mu1, Sigma1)
    mu2, Sigma2 = sticking_covariance(n_times, sigma0, me)
    y = np.random.multivariate_normal(mu2, Sigma2)

    return (x, y)


def calAutoCorr(sdx, n_lag):
    autocorrelation = np.zeros((n_lag,))
    for i in range(n_lag):
        shift = i + 1
        correlation = np.corrcoef(sdx[shift:], sdx[:-shift])[0, 1]
        autocorrelation[i] = correlation
    return np.insert(autocorrelation, 0, 1)


def autoCorrFirstX(x, dt):
    single_jump_ind = np.where(dt == 1.0)[0]
    corr = np.corrcoef(x[single_jump_ind], x[single_jump_ind + 1])

    return corr[0, 1]


def auto_x_dx(D, lam_lis, t_, repeat, n_lag):
    dx_lis, x_lis, er_lis = [], [], []
    for i in range(len(lam_lis)):
        temp1, temp2, temp3 = [], [], []
        for j in range(repeat):
            lambda_ = lam_lis[i]
            baseHPW = base_HPW_D([0, 0], t_, D, [0, 0], lam_lis[i])
            autocorr_x = calAutoCorr(np.diff(baseHPW[0]), n_lag)
            autocorr_y = calAutoCorr(np.diff(baseHPW[1]), n_lag)
            autocorr2_x = calAutoCorr(baseHPW[0], n_lag)
            autocorr2_y = calAutoCorr(baseHPW[1], n_lag)
            meandx = (autocorr_x[-1] + autocorr_y[-1]) / 2
            meanx = (autocorr2_x[-1] + autocorr2_y[-1]) / 2
            D_temp = MAP_hpw(baseHPW[0], baseHPW[1], np.diff(t_)[0])

            temp1.append(meandx)
            temp2.append(meanx)
            temp3.append(D_temp / D)

        dx_lis.append(temp1)
        x_lis.append(temp2)
        er_lis.append(temp3)

    return dx_lis, x_lis, er_lis


def PPCs(stat, model):
    sx, sy, st, sdx, sdy, sdt = stat

    if model == "bm":
        bm = pm.Model()

        with bm:
            D = pm.Lognormal("D", 0, 1)
            like_x = pm.Normal(
                "like_x", mu=0, sd=tt.sqrt(2 * D * sdt), observed=sdx
            )
            like_y = pm.Normal(
                "like_y", mu=0, sd=tt.sqrt(2 * D * sdt), observed=sdy
            )

            trace_bm = pm.sample(2000, chains=2, cores=2, progressbar=False)

        ppc_bm = pm.sample_posterior_predictive(trace_bm, model=bm, progressbar=False)
        simulated_dx_bm = ppc_bm[bm.observed_RVs[0].name]
        simulated_dy_bm = ppc_bm[bm.observed_RVs[1].name]
        simulated_x_bm = np.insert(
            np.cumsum(simulated_dx_bm, axis=1), 0, 0, axis=1
        )
        simulated_y_bm = np.insert(
            np.cumsum(simulated_dy_bm, axis=1), 0, 0, axis=1
        )

        # ppc in autocorrX, lag=1
        pxstd = []
        for i in range(4000):
            pxstd.append(calAutoCorr(simulated_x_bm[i, :], 1)[-1])

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for i, j in zip(simulated_x_bm[::2, :], simulated_y_bm[::2, :]):
            axes[0].plot(i, j, alpha=0.2)
        axes[0].plot(sx, sy, c="k", label="True data")
        axes[1].hist(simulated_x_bm.std(axis=1), bins=30)
        axes[2].hist(pxstd, bins=30)
        axes[2].axvline(
            x=autoCorrFirstX(sx, sdt), ls="--", c="r", label="data autocorrX"
        )
        axes[1].axvline(x=sx.std(), ls="--", c="r", label="data std in x")
        axes[0].legend()
        axes[1].legend()
        axes[2].legend()
        axes[0].set_title("PP Samples from BM model and True track")
        axes[1].set_title("PP Samples std in x")
        axes[2].set_title("PP Samples autocorrX")

    if model == "me":

        model_stick = pm.Model()
        with model_stick:

            me = pm.Lognormal("me", 0, 1)
            _, sig = sticking_covariance(len(sx) - 1, 0, me)

            likex = pm.MvNormal("likex", mu=0, cov=sig, observed=sx)
            likey = pm.MvNormal("likey", mu=0, cov=sig, observed=sy)

        with model_stick:
            trace_stick = pm.sample(2000, chains=2, cores=2, progressbar=False)

        # manually generate ppc samples
        simulated_x_stick, simulated_y_stick = (
            np.zeros((4000, len(sx))),
            np.zeros((4000, len(sx))),
        )
        for i in range(4000):
            mu1, Sigma1 = sticking_covariance(
                len(sx) - 1, 0, trace_stick["me"][i]
            )
            x = np.random.multivariate_normal(mu1, Sigma1)
            mu2, Sigma2 = sticking_covariance(
                len(sx) - 1, 0, trace_stick["me"][i]
            )
            y = np.random.multivariate_normal(mu2, Sigma2)

            simulated_x_stick[i, :], simulated_y_stick[i, :] = x, y

        # ppc in autocorrX, lag=1
        pxstd = []
        for i in range(4000):
            pxstd.append(calAutoCorr(simulated_x_stick[i, :], 1)[-1])

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for i, j in zip(simulated_x_stick[::2, :], simulated_y_stick[::2, :]):
            axes[0].plot(i, j, alpha=0.2)
        axes[0].plot(sx, sy, c="k", label="True data")
        axes[1].hist(pxstd, bins=30)
        axes[2].hist(simulated_x_stick.std(axis=1), bins=30)
        axes[1].axvline(
            x=autoCorrFirstX(sx, sdt), ls="--", c="r", label="data autocorrX"
        )
        axes[2].axvline(x=sx.std(), ls="--", c="r", label="data std in x")
        axes[0].legend()
        axes[1].legend()
        axes[2].legend()
        axes[0].set_title("PP Samples from Stuck model and True track")
        axes[1].set_title("PP Samples autocorrX")
        axes[2].set_title("PP Samples std in x")
        plt.show()

    if model == 'hpw':
        model_hpw = pm.Model()
        with model_hpw:
            D = pm.Lognormal("D", 0, 1)
            k = pm.Lognormal("k", 0, 1)

            mean_x = (-sx[:-1]) * (1 - tt.exp(-k * sdt))
            mean_y = (-sy[:-1]) * (1 - tt.exp(-k * sdt))
            std = tt.sqrt(D * (1 - tt.exp(-2 * k * sdt)) / k)

            like_x = pm.Normal("like_x", mu=mean_x, sd=std, observed=sdx)
            like_y = pm.Normal("like_y", mu=mean_y, sd=std, observed=sdy)

        with model_hpw:
            trace_hpw = pm.sample(2000, tune=2000, chains=2, cores=2, progressbar=False)

        simulated_x_hpw, simulated_y_hpw = np.zeros((4000, len(sx))), np.zeros((4000, len(sx)))
        for i in range(trace_hpw['D'].shape[0]):
            base = base_HPW_D([0, 0], [i for i in range(len(sx))], trace_hpw['D'][i], [0, 0], trace_hpw['k'][i])
            simulated_x_hpw[i, :], simulated_y_hpw[i, :] = base[0], base[1]

        pxstd = []
        for i in range(4000):
            pxstd.append(calAutoCorr(simulated_x_hpw[i, :], 1)[-1])

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for i, j in zip(simulated_x_hpw[::2, :], simulated_y_hpw[::2, :]):
            axes[0].plot(i, j, alpha=0.2)
        axes[0].plot(sx, sy, c="k", label="True data")
        axes[1].hist(pxstd, bins=30)
        axes[2].hist(simulated_x_hpw.std(axis=1), bins=30)
        axes[1].axvline(
            x=autoCorrFirstX(sx, sdt), ls="--", c="r", label="data autocorrX"
        )
        axes[2].axvline(x=sx.std(), ls="--", c="r", label="data std in x")
        axes[0].legend()
        axes[1].legend()
        axes[2].legend()
        axes[0].set_title("PP Samples from Stuck model and True track")
        axes[1].set_title("PP Samples autocorrX")
        axes[2].set_title("PP Samples std in x")
        plt.show()

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


def plot_DMeAnalysis(stats):  
    param = generate_params(Jiayu_params=False) 

    [dx, dy, dt, track_info] = stats 
    MAP_est = [] 
    for idx, ind in enumerate(np.array([268, 180, 411, 286, 237, 632, 242, 426, 402, 374])): 
        sdx, sdy, sdt = loadSelectTraj(dx, dy, dt, track_info, ind, True) 
    #     sx, sy, st = utils.loadSelectTraj(x, y, t, track_info, ind, False)
        
        MAP_est.append(MAP_bm(sdx, sdy, sdt, 2, 1)) 

    diffusive_ind = np.array([268, 180, 411, 286, 237, 632, 242, 426, 402, 374]) 

    plt.figure(1, figsize=(8,6)) 

    bins_D = np.logspace(param['mu_D'] - 3.*param['sigma_D'], param['mu_D'] + 3.*param['sigma_D'], 51)  


    plt.subplot(211)
    sns.distplot(pm.Lognormal.dist(param['mu_D'], param['sigma_D']).random(size=15000), 
                 bins=bins_D, kde=False, label='prior') 
    for idx, ind in enumerate(diffusive_ind): 
        parent = 'results_ind_' + str(ind) 
        D_trace = np.loadtxt(parent + '/_postsamp'+ str(ind) +'.tsv')[:,0] 
        sns.distplot(D_trace, bins=np.logspace(np.log10(min(D_trace)), np.log10(max(D_trace)), 21), 
                     kde=False)  
        plt.axvline(MAP_est[idx], c='k', alpha=0.35) 
    plt.axvline(MAP_est[idx], c='k', alpha=0.35, label='MAP estimate') 
    plt.semilogx() 
    plt.legend() 
    plt.xlim(1e-5, 1e3)
    plt.title(r'$D \ \ [pix^2/frame]$', fontsize=18) 

    bins_Me = np.logspace(param['mu_Me'] - 1.*param['sigma_Me'], param['mu_Me'] + 5.*param['sigma_Me'], 51)
    plt.subplot(212) 
    sns.distplot(pm.Lognormal.dist(param['mu_Me'], param['sigma_Me']).random(size=15000), 
                 bins=bins_Me, kde=False, label='prior') 
    for idx, ind in enumerate(diffusive_ind): 
        parent = 'results_ind_' + str(ind) 
        Me_trace = np.loadtxt(parent + '/_postsamp'+ str(ind) +'.tsv')[:,1] 
        sns.distplot(Me_trace, bins=np.logspace(np.log10(min(Me_trace)), np.log10(max(Me_trace)), 21), 
                     kde=False)  
    plt.semilogx() 
    plt.legend() 
    plt.xlim(1e-4, 1e2) 
    plt.title(r'$M_e \ \ [pix]$', fontsize=18) 

    plt.tight_layout() 
    plt.show() 

    return 