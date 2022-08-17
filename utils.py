import contextlib
import numpy as np

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def EulerIntegrate(controller,system, f, B, Bw, xstar, ustar, xinit, t_max = 10, dt = 0.05, with_tracking = False, sigma = 0., noise_bound = None):
    t = np.arange(0, t_max, dt)

    trace = []
    u = []
    max = 0
    xcurr = xinit
    trace.append(xcurr)
    disturbance = np.random.rand(system.num_dim_distb, 1) #added

    for i in range(len(t)):
        if with_tracking:
            xe = xcurr - xstar[i]
        ui = controller(xcurr, xe, ustar[i]) if with_tracking else ustar[i]
        if with_tracking:
            # print(xcurr.reshape(-1), xstar[i].reshape(-1), ui.reshape(-1))
            pass

        if not noise_bound:
            noise_bound = 3 * sigma
        noise = np.random.randn(*xcurr.shape) * sigma
        noise[noise>noise_bound] = noise_bound
        noise[noise<-noise_bound] = -noise_bound


        norm = np.linalg.norm(disturbance)
        #print(norm)
        if norm > max:
            max = norm
        #dx = f(xcurr) + B(xcurr).dot(ui) + noise
        dx = f(xcurr) + B(xcurr).dot(ui) + Bw(xcurr).dot(disturbance) if with_tracking else f(xcurr) + B(xcurr).dot(ui)
        xnext =  xcurr + dx*dt
        # xnext[xnext>100] = 100
        # xnext[xnext<-100] = -100

        trace.append(xnext)
        u.append(ui)
        xcurr = xnext

    if with_tracking:
        print("Tube Size:", 0.5822179913520813 * max) # 3.028190851211548 0.8038054704666138  1.1543467044830322
    return trace, u
