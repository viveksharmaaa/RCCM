import numpy as np
from utils import temp_seed

# for ours training
X_MIN = np.array([-35., -2., -np.pi/3, -2., -1., -np.pi/3]).reshape(-1,1)
X_MAX = np.array([0., 2., np.pi/3, 2., 1., np.pi/3]).reshape(-1,1)

g = 9.81
m = 0.486

az_l = m*g/2 - 1.
az_u = m*g/2 + 1.

UREF_MIN = np.array([az_l, az_l]).reshape(-1,1)
UREF_MAX = np.array([az_u, az_u]).reshape(-1,1)

lim = 1
XE_MIN = np.array([-lim, -lim, -lim, -lim, -lim, -lim]).reshape(-1,1)
XE_MAX = np.array([lim, lim, lim, lim, lim, lim]).reshape(-1,1)

# for sampling ref
X_INIT_MIN = np.array([0., 0., -0.1, 0.5, 0, 0])
X_INIT_MAX = np.array([ 0., 0., 0.1, 1, 0, 0])

XE_INIT_MIN = np.array([-0.5,]*6)
XE_INIT_MAX = np.array([ 0.5,]*6)

time_bound = 6.
time_step = 0.03
t = np.arange(0, time_bound, time_step)

def system_reset(seed):
    SEED_MAX = 10000000
    with temp_seed(int(seed * SEED_MAX)):
        xref_0 = X_INIT_MIN + np.random.rand(len(X_INIT_MIN)) * (X_INIT_MAX - X_INIT_MIN)
        xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
        x_0 = xref_0 + xe_0

        g = 9.81
        freqs = list(range(1,11))
        # freqs = []
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (0.1 * weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))).tolist()  #was 2
        uref = []
        for _t in t:
            u = np.array([0., 0,]) # ref
            for freq, weight in zip(freqs, weights):
                u += np.array([weight[0] * np.sin(freq * _t/time_bound * 2*np.pi), 0.1*weight[1] * np.sin(freq * _t/time_bound * 2*np.pi)]) #np.array([((UREF_MAX-UREF_MIN) * np.random.rand(2,1) + UREF_MIN).reshape(-1)[0],((UREF_MAX-UREF_MIN) * np.random.rand(2,1) + UREF_MIN).reshape(-1)[1]])
            # u += 0.01*np.random.randn(2)
            uref.append(u)

    return x_0, xref_0, uref
