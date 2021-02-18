""" this script is extracted from https://github.com/matejbalog/RKHS-private-database/blob/master/src/plot.py """
import json
import matplotlib.pylab as plt
import numpy as np

def json_load(path, report_content=None):
    with open(path, 'r') as f:
        data_json = json.load(f)
    if report_content is not None:
        print('LOADED', '%s from %s' % (report_content, path))
    return data_json

def tableau20(k):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    return tableau20[k]

j2 = json_load('D5_alg2_random_M10000.json')
epsilons, Ms_alg2, D = j2['epsilons'], j2['Ms_report'], j2['D']


fig, ax = plt.subplots(1, 1, figsize=(5, 4))
# ax.title('$D = %d$' % (D))
# ax.text(0.5, 0.95, '$D = %d$' % (D), transform=ax.transAxes)
# x-axis
M_max = 0
M_max = max(M_max, Ms_alg2[-1])
ax.set_xlim((2, M_max))
ax.set_xscale('log')
ax.set_xlabel('$M$ (number of synthetic data points)')

# y-axis
ax.set_ylim((5 * 1e-3, 4.0))
ax.set_yscale('log')
ax.set_ylabel('RKHS distance $\|\| \hat{\mu}_X - \cdot \|\|_{\mathcal{H}}$')

epsilons = epsilons[1:]
Ms_alg2 = Ms_alg2[1:]

colors = [tableau20(2*ei) for ei, _ in enumerate(epsilons)]
input_size = 10

for ei, epsilon in enumerate(epsilons):
    dists = j2['dists_alg2'][ei]
    label = 'Balog18, $\\varepsilon = %s$' % (epsilon)
    ax.plot(Ms_alg2, dists[1:], color=colors[ei], ls='dashed', label=label)

    filename = 'epsilon='+np.str(epsilon)+'input_size='+np.str(input_size)+'.npy'
    dist_ours = np.load(filename)
    label = 'Ours, $\\varepsilon = %s$' % (epsilon)
    ax.plot(Ms_alg2, dist_ours[1:], color=colors[ei], ls='solid', label=label)


# Plot legend
ax.legend(loc='upper left')

plt.show()

# a good proxy for the
# RKHS distance between the true KME µX and the RKHS element represented by the released dataset.