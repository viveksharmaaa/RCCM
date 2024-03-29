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

def EulerIntegrate(controller,system, f, B, Bw, xstar, ustar, xinit, t_max = 10, dt = 0.05, with_tracking = False, sigma = 0., noise_bound = None, CCM = False, RCCM = True):
    t = np.arange(0, t_max, dt)

    trace = []
    u = []
    max = 0
    max_noise_norm = 0
    xcurr = xinit
    trace.append(xcurr)
    # distb = np.random.randn(system.num_dim_distb, 1)
    # disturbance = sigma * (distb/np.linalg.norm(distb))

    for i in range(len(t)):
        w = (sigma * (0.8 + 0.2 * np.sin(0.2 * np.pi * t[i])) * np.array([np.sin(np.pi / 4), -np.cos(np.pi / 4), 0])).reshape(system.num_dim_distb, 1)
        if with_tracking:
            xe = xcurr - xstar[i]
        ui = controller(xcurr, xe, ustar[i]) if with_tracking else ustar[i]
        if with_tracking:
            # print(xcurr.reshape(-1), xstar[i].reshape(-1), ui.reshape(-1))
            pass

        if CCM == True :
            # if not noise_bound: #Frobenius Norm
            #     #noise_bound = 3 * sigma # multiplier three is norm of Bw aka torch.norm(torch.eye(9)) dist is vector 9 x1, sigma is bound of disturbance
            #     noise_bound = np.linalg.norm(Bw(xcurr),ord = -1) * sigma
            # noise = 0.000001 + np.random.randn(*xcurr.shape) * sigma
            # noise = (noise/np.linalg.norm(noise))*noise_bound
            # noise[noise > noise_bound] = noise_bound
            # noise[noise < -noise_bound] = -noise_bound
            #
            # norm_noise = np.linalg.norm(noise)
            # # print(norm)
            # if norm_noise > max_noise_norm:
            #     max_noise_norm = norm_noise

            #dx = f(xcurr) + B(xcurr).dot(ui) + noise
            dx = f(xcurr) + B(xcurr).dot(ui) + Bw(xcurr).dot(w) if with_tracking else f(xcurr) + B(xcurr).dot(ui)


        elif RCCM == True: # Note RCCM utilize knowledge of Bw. Make sure the distb only influences three states
            # distb = np.random.randn(system.num_dim_distb, 1)
            # disturbance = sigma * (distb/np.linalg.norm(distb))
            # norm = np.linalg.norm(disturbance)
            # #print(norm)
            # if norm > max:
            #     max = norm
            #dx = f(xcurr) + B(xcurr).dot(ui) + Bw(xcurr).dot(disturbance) if with_tracking else f(xcurr) + B(xcurr).dot(ui)
            dx = f(xcurr) + B(xcurr).dot(ui) + Bw(xcurr).dot(w) if with_tracking else f(xcurr) + B(xcurr).dot(ui)

        xnext =  xcurr + dx*dt
        # xnext[xnext>100] = 100
        # xnext[xnext<-100] = -100

        trace.append(xnext)
        u.append(ui)
        xcurr = xnext

    # if with_tracking and RCCM:
    #     print("RCCM max L-2 distb norm:",max)
    #     print("RCCM Tube Size:", 1.1462310552597046 * max)
    # elif with_tracking and CCM:
    #     print("CCM Max noise norm:", max_noise_norm)
    return trace, u
