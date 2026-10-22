import numpy
import matplotlib.pyplot
import scipy
from skopt import gp_minimize


# This function is to fit double half-Gaussians with a straight line in between for SX3 position calibration
def top_hat(x, left, right, sigma, left_amp, right_amp):
    y = []
    for xp in x:
        amp = left_amp + (right_amp - left_amp) * (xp - left) / (right - left)
        if (xp < left):
            y.append(amp * numpy.exp(-(xp - left) ** 2 / (2 * sigma ** 2)))
        elif (xp < right):
            y.append(amp)
        else:
            y.append(amp * numpy.exp(-(xp - right) ** 2 / (2 * sigma ** 2)))

    return y

chi2_history = []

# Main fitter that brings everything together
class global_fitter:
    def __init__(self):
        self.params = numpy.zeros(35)
        self.r_E_barrel = 100.2;
        self.sx3width = 40.0;
        self.radius = [
            numpy.sqrt(self.r_E_barrel ** 2 + (3. * self.sx3width / 8.) ** 2),
            numpy.sqrt(self.r_E_barrel ** 2 + (self.sx3width / 8.) ** 2),
            numpy.sqrt(self.r_E_barrel ** 2 + (self.sx3width / 8.) ** 2),
            numpy.sqrt(self.r_E_barrel ** 2 + (3. * self.sx3width / 8.) ** 2),
        ]

        self.pad_offset = numpy.zeros(4)
        self.bb10_offset = numpy.zeros(8)
        self.chi2_LR = 0
        self.chi2_PID = 0
        self.chi2_Ex = 0
        self.chi2_edges = 0
        self.datasets = []
        self.chi2_LR_scale = 1
        self.chi2_PID_scale = 1
        self.chi2_Ex_scale = 1
        self.chi2_edges_scale = 1
        self.eval_ct = 0

        self.data_PID_E = []
        self.data_PID_dE = []
        self.data_Ex = []
        self.data_back_E = []
        self.data_front_E = []
        self.data_raw_pos = [[], [], [], []]

    def load_PID(self, filename):
        data = numpy.genfromtxt(filename)
        self.PID_dE = scipy.interpolate.interp1d(data[:, 0], data[:, 1], fill_value="extrapolate", bounds_error=False)
        self.PID_E = scipy.interpolate.interp1d(data[:, 1], data[:, 0], fill_value=(0, data[:, 0][-1]),
                                                bounds_error=False)

    def load_params(self, filename):
        f = open(filename)
        for line in f.readlines():
            if len(line) == 0: continue
            line = line.split()
            if line[0] == "SX3b":
                pad = int(line[1])
                strip = int(line[2])
                gain = float(line[3])
                self.params[pad * 4 + strip] = gain
            elif line[0] == "SX3f":
                strip = int(line[1])
                gainL = float(line[2])
                gainR = float(line[3])
                self.params[16 + strip] = gainL
                self.params[20 + strip] = gainR
            elif line[0] == "BB10":
                strip = int(line[1])
                gain = float(line[2])
                self.params[24 + strip] = gain
            elif line[0] == "SX3p":
                strip = int(line[1])
                left = float(line[2])
                right = float(line[3])
                #self.params[32 + strip] = left
                #self.params[36 + strip] = right

    def set_kinematics(self, m_beam, m_targ, m_eject, m_rec, Q_gs):
        self.m_beam_MeV = m_beam * 931.478;
        self.m_targ_MeV = m_targ * 931.478;
        self.m_eject_MeV = m_eject * 931.478;
        self.m_rec_MeV = m_rec * 931.478;
        self.Q_gs = Q_gs

    def get_Ex(self, energy, theta, beam_en):
        p_beam = numpy.sqrt(beam_en ** 2 + 2.0 * self.m_beam_MeV * beam_en)
        p_eject = numpy.sqrt(energy * energy + 2.0 * self.m_eject_MeV * energy);
        e_beam = beam_en + self.m_beam_MeV;
        e_eject = self.m_eject_MeV + energy;

        e_rec = e_beam + self.m_targ_MeV - e_eject;

        p_beam = numpy.sqrt(beam_en * beam_en + 2.0 * self.m_beam_MeV * beam_en);

        relQval = self.m_beam_MeV + self.m_targ_MeV - self.m_eject_MeV \
                  - numpy.sqrt(
            self.m_beam_MeV * self.m_beam_MeV + self.m_targ_MeV * self.m_targ_MeV + self.m_eject_MeV * self.m_eject_MeV + 2.0 * self.m_targ_MeV * e_beam \
            - 2.0 * e_eject * (e_beam + self.m_targ_MeV) + 2. * p_beam * p_eject * numpy.cos(theta));

        return (self.Q_gs - relQval);

    def eval(self, data):
        # data = [det, pad, strip, bb10strip, pad E, left E, right E, bb10 E, beam energy]
        # self.params = [pad gains, strip L gains, strip R gains, bb10 gains, left edges, right edges]
        # fetch correct parameters
        if (int(data[0]) != self.detID):
            return
        data = data[1:]
        pad = int(data[0])
        strip = int(data[1])
        bb10strip = int(data[2])

        pad_gain = self.params[pad * 4 + strip]
        stripL_gain = self.params[16 + strip]
        stripR_gain = self.params[20 + strip]
        bb10_gain = self.params[24 + bb10strip]
        left_edge = -0.5
        right_edge = 0.5
        z_offset = self.params[32]
        beam_en_offset = self.params[33]
        beam_corr_f = self.params[34]

        en_dE = data[6] * bb10_gain + self.bb10_offset[bb10strip]
        en_E = data[3] * pad_gain + self.pad_offset[pad]
        front_sum = stripL_gain * data[4] + stripR_gain * data[5]
        raw_position = (stripR_gain * data[5] - stripL_gain * data[4]) / front_sum
        self.data_raw_pos[strip].append(raw_position)
        position = (raw_position - left_edge) * (1. / (right_edge - left_edge)) * 75.0 + z_offset;
        theta = numpy.arctan(self.radius[strip] / position)
        if (theta < 0):
            theta += numpy.pi
        self.chi2_LR += (front_sum - en_E) ** 2
        self.data_back_E.append(en_E)
        self.data_front_E.append(front_sum)
        alpha_E = en_dE + en_E;

        en_dE = en_dE * numpy.sin(theta)
        en_E = en_E + en_dE * (1 - numpy.sin(theta))
        self.chi2_PID += (self.PID_dE(en_E / 1000.0) - en_dE / 1000.0) ** 2
        self.chi2_PID += (self.PID_E(en_dE / 1000.0) - en_E / 1000.0) ** 2
        self.data_PID_dE.append(en_dE / 1000.0)
        self.data_PID_E.append(en_E / 1000.0)

        if (self.ref_Ex > 0):
            # beam_en = numpy.max([25.0, self.nominal_beam_en + (data[7]-330)*beam_corr_f + beam_en_offset])
            beam_en = numpy.max([25.0, self.nominal_beam_en * (1 + (data[7] - 330) * beam_corr_f)])
            # print(self.nominal_beam_en, data[7], beam_corr_f, beam_en, beam_en_offset)
            self.chi2_Ex += (self.get_Ex(alpha_E / 1000., theta, beam_en) - self.ref_Ex) ** 2
            self.data_Ex.append(self.get_Ex(alpha_E / 1000., theta, beam_en))

    def eval_dataset(self, dataset, ref_Ex):
        self.ref_Ex = ref_Ex;
        data = numpy.genfromtxt(dataset)
        for d in data:
            self.eval(d)

    def clear(self):
        self.chi2_LR = 0
        self.chi2_PID = 0
        self.chi2_Ex = 0
        self.chi2_edges = 0
        self.data_PID_E = []
        self.data_PID_dE = []
        self.data_Ex = []
        self.data_back_E = []
        self.data_front_E = []
        self.data_raw_pos = [[], [], [], []]

    def add_dataset(self, name, ref_Ex):
        self.datasets.append([name, ref_Ex])

    def plot(self, figname):
        fig, ax = matplotlib.pyplot.subplots(2, 2, figsize=(10, 8))
        ax[0][0].plot(self.data_PID_E, self.data_PID_dE, 'k.', ms=1)
        x = numpy.linspace(0, 32, 100)
        ax[0][0].plot(x, self.PID_dE(x), 'r-')
        counts, bins, patches = ax[1][0].hist(numpy.array(self.data_front_E) / numpy.array(self.data_back_E), bins=100)
        ax[1][0].plot([1, 1], [0, max(counts)], 'r-')
        counts, bins, patches = ax[0][1].hist(self.data_Ex, bins=100, range=(0, 8))
        for i in range(len(self.datasets)):
            if (self.datasets[i][1] > 0):
                ax[0][1].plot([self.datasets[i][1], self.datasets[i][1]], [0, max(counts)], 'r-')

        cols = ['r', 'xkcd:cobalt', 'xkcd:green', 'orange']
        for strip in range(0, 4):
            counts, bins, patches = ax[1][1].hist(self.data_raw_pos[strip], bins=100, color=cols[strip], fill=False,
                                                  histtype='step', range=(-1, 1))
            #ax[1][1].plot([self.params[32 + strip], self.params[32 + strip]], [0, max(counts)], '-', color=cols[strip],
            #ax[1][1].plot([self.params[36 + strip], self.params[36 + strip]], [0, max(counts)], '-', color=cols[strip],


        fig.savefig(figname, dpi=300)
        matplotlib.pyplot.close(fig)

    def eval_all(self, params):
        ## set params = to params
        left_edge = -0.5
        right_edge = 0.5
        self.clear()
        # for bayesian
        self.params = params

        for d in self.datasets:
           self.eval_dataset(d[0], d[1])

        self.chi2_edges = 0
        if (self.eval_ct % 100 == 0):
            fig, ax = matplotlib.pyplot.subplots(2, 2)
        for strip in range(0, 4):
            hist, bin_edges = numpy.histogram(self.data_raw_pos[strip], bins=100, range=(-1, 1))
            x = [(bin_edges[i] + bin_edges[i + 1]) / 2. for i in range(len(hist))]
            # popt = scipy.optimize.curve_fit(top_hat, x, hist, p0=(-0.6, 0.6, 0.05, 10, 10))
            # left_edge = (popt[0][0]-abs(popt[0][2])*2.3548/2.)
            # right_edge = (popt[0][1]+abs(popt[0][2])*2.3548/2.)
            for i in range(len(hist)):
                if (hist[i] > self.threshold):
                    left_edge = x[i]
                    break
            for i in reversed(range(len(hist))):
                if (hist[i] > self.threshold):
                    right_edge = x[i]
                    break

            if (self.eval_ct % 100 == 0):
                ax[int(strip / 2)][strip % 2].plot(x, hist, 'ko', markersize=3)
                # ax[int(strip/2)][strip%2].plot(x,top_hat(x, *popt[0]), 'r')
                ax[int(strip / 2)][strip % 2].plot([left_edge, left_edge], [0, max(hist)], '--', color='grey')
                ax[int(strip / 2)][strip % 2].plot([right_edge, right_edge], [0, max(hist)], '--', color='grey')

            #self.chi2_edges += (left_edge - self.params[32 + strip]) ** 2
            ## self.chi2_edges += (right_edge - self.params[36 + strip]) ** 2

        if (self.eval_ct % 100 == 0):
            fig.savefig("global_fitter_edges.png", dpi=300)
            matplotlib.pyplot.close(fig)

        chi2 = self.chi2_LR * self.chi2_LR_scale
        chi2 += self.chi2_PID * self.chi2_PID_scale
        chi2 += self.chi2_Ex * self.chi2_Ex_scale
        chi2 += self.chi2_edges * self.chi2_edges_scale

        if (self.eval_ct % 100 == 0):
            print("%5.4f   %5.4f   %5.4f   %5.4f" % (self.chi2_LR * self.chi2_LR_scale,
                                                     self.chi2_PID * self.chi2_PID_scale,
                                                     self.chi2_Ex * self.chi2_Ex_scale,
                                                     self.chi2_edges * self.chi2_edges_scale))

        if (self.eval_ct % 100 == 0):
            self.plot("global_fitter_prog.png")

        self.eval_ct += 1
        # for bayesian
        total_loss = 0
        #for d in self.datasets:
            #total_loss += self.eval_dataset(d[0], d[1])

        chi2_history.append(chi2)

        return chi2


fitter = global_fitter()
fitter.detID = 16  # Change this for every clock (24-hour) position
fitter.threshold = 5
fitter.load_PID("SiPID_65um_4He.dat")
fitter.set_kinematics(3.0160, 38.9637, 4.0026, 37.9691, 7.4999)
fitter.nominal_beam_en = 31.5;
fitter.load_params("SX3.16.params")  # Change clock position here as well
# fitter.params[42] = 0.0115;
fitter.params[34] = 0.0004025;
print(fitter.params)
# Below are the different data sets for the different states
fitter.eval_dataset("labeled_data_2401.dat", 2.401)
fitter.eval_dataset("labeled_data_458.dat", 0.458)
fitter.eval_dataset("labeled_data_4661.dat", 4.661)
fitter.eval_dataset("labeled_data_7140.dat", 7.140)
fitter.eval_dataset("labeled_data_1698.dat", 1.698)
print(fitter.chi2_LR, fitter.chi2_PID, fitter.chi2_Ex)
fitter.eval_dataset("alpha_ds_data.dat", -1)
print(fitter.chi2_LR, fitter.chi2_PID, fitter.chi2_Ex)
# Knobs to fine-tune for optimal optimization
fitter.chi2_LR_scale = 1e-4
fitter.chi2_PID_scale = 1
fitter.chi2_Ex_scale = 500
fitter.chi2_edges_scale = 100000

fitter.plot("global_fitter_init.png")

fitter.add_dataset("labeled_data_2401.dat", 2.401)
fitter.add_dataset("labeled_data_458.dat", 0.458)
fitter.add_dataset("labeled_data_4661.dat", 4.661)
fitter.add_dataset("labeled_data_7140.dat", 7.140)
fitter.add_dataset("labeled_data_1698.dat", 1.698)
fitter.add_dataset("alpha_ds_data.dat", -1)
p0 = fitter.params

# popt = scipy.optimize.minimize(fitter.eval_all, p0, method='Powell')

bounds = []
for val in p0:
    if val == 0:
        bounds.append((-10.0, 10.0))  # Fallback range for zero parameters
    else:
        # Example: Allow the optimizer to look between 50% and 150% of the initial guess
        low_bound = min(val * 0.9, val * 1.1)
        high_bound = max(val * 0.9, val * 1.1)
        bounds.append((low_bound, high_bound))


# Feedback from Dr. Gray
# Focus on finding ideal difference between n_calls and n_initial_points
# n_initial_points is the amount of initial guesses, so n_calls-n_initial points
# should ideally be large enough to where the amount of "actual" guesses is considerable
# for example -> n_calls 10 with n_calls 8 would only leave 2 "actual" calls
popt = gp_minimize(
    func=fitter.eval_all,       
    dimensions=bounds,          
    n_calls=500000,
    n_initial_points=400000,
    acq_func='PI',
    random_state=42
)

fig, ax = matplotlib.pyplot.subplots(figsize=(8, 4))

# Plot the progression of chi2 scores
ax.plot(chi2_history, marker='o', color='teal', linestyle='-', label=r'Evaluated $\chi^2$')

# shows minimum found
ax.axhline(popt.fun, color='red', linestyle='--', label=f'Best Fit ({popt.fun:.2f})')

# labels
ax.set_title(r'Convergence Trace of $\chi^2$ via gp_minimize')
ax.set_xlabel('Iteration Index')
ax.set_ylabel(r'Returned $\chi^2$ Value')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()

# save file
matplotlib.pyplot.savefig('gp_optimization_history.png', dpi=300, bbox_inches='tight')
matplotlib.pyplot.close()

# popt.x = res.x
print("Best parameters found:", popt)

# print(popt)
# fitter.eval_all(popt.x)
print(fitter.chi2_LR, fitter.chi2_PID, fitter.chi2_Ex)

fitter.plot("global_fitter_final.png")

for i in range(len(popt.x)):
    print("%8.7f" % popt.x[i])

