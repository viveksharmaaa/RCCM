from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from np2pth import get_system_wrapper, get_controller_wrapper

import importlib
import pickle
from utils import EulerIntegrate
import time

import os
import sys
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')
import argparse

# import argparse as args
# args.task = 'QUADROTOR_9D'
# args.pretrained_RCCM = 'log_QUADROTOR_9D_refined_0818/controller_best_ref.pth.tar'
# args.pretrained_CCM = 'log_QUADROTOR_9D_refined_0818/controller_best_ref.pth.tar'
# args.sigma = 0
# args.nTraj = 2
# args.seed = 0
# args.ref_traj_many = False
# args.CCM = True
# args.RCCM = False
# args.init_same = False

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 13
HUGE_SIZE = 25

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=HUGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=HUGE_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)

left = 0.14  # the left side of the subplots of the figure
right = 0.98   # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.925     # the top of the subplots of the figure

parser = argparse.ArgumentParser(description="")
parser.add_argument('--task', type=str,
                        default='CAR')
parser.add_argument('--pretrained_CCM', type=str, default = None)
parser.add_argument('--pretrained_RCCM', type=str, default = None)
parser.add_argument('--plot_type', type=str, default='2D')
parser.add_argument('--plot_dims', nargs='+', type=int, default=[0,1])
parser.add_argument('--nTraj', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--sigma', type=float, default=0.)
parser.add_argument('--xd0scaler',type=float, default=0.5)
parser.add_argument('--x0scaler',type=float, default=0.5)
parser.add_argument('--plot_comp', nargs='+', type=int, default=[0,1])
parser.add_argument('--ref_traj_many', type=bool, default=False)
# parser.add_argument('--CCM', type=bool, default=False)
# parser.add_argument('--RCCM', type=bool, default=True)
parser.add_argument('--init_same', type=bool, default=False)

args = parser.parse_args()

np.random.seed(args.seed)


if args.pretrained_RCCM:
    to = args.pretrained_RCCM.split('/')[0] + '/' + args.task
    with open(to + '.pkl', 'rb') as f:
        al = pickle.load(f)
    alpha = al[2][-1]
    mu = al[3][-1]


system = importlib.import_module('system_'+args.task)
f, B, Bw, _, num_dim_x, num_dim_control = get_system_wrapper(system)


if args.pretrained_CCM:
    controller_CCM = get_controller_wrapper(args.pretrained_CCM)
    CCM = torch.load('log_' + args.task + '_0801/model_best.pth.tar', map_location=torch.device('cpu'))
    #CCM = torch.load(args.pretrained_CCM, map_location=torch.device('cpu'))
    CCM_tube = np.sqrt(CCM['args'].w_lb / CCM['args'].w_ub) * 1 / CCM['args']._lambda

if args.pretrained_RCCM:
    controller_RCCM = get_controller_wrapper(args.pretrained_RCCM)

if __name__ == '__main__':
    config = importlib.import_module('config_'+args.task)
    t = config.t
    time_bound = config.time_bound
    time_step = config.time_step
    XE_INIT_MIN = config.XE_INIT_MIN
    XE_INIT_MAX = config.XE_INIT_MAX

    if not args.ref_traj_many:
        x_0, xstar_0, ustar = config.system_reset(np.random.rand())  # x_0, xstar_0
        # #added
        # #x_0 = args.x0scaler * (config.X_MIN.reshape(-1) + np.random.rand(len(config.X_MIN.reshape(-1))) * (config.X_MAX.reshape(-1) - config.X_MIN.reshape(-1)))
        # #xstar_0 = args.xd0scaler * (config.X_MIN.reshape(-1) + np.random.rand(len(config.X_MIN.reshape(-1))) * (config.X_MAX.reshape(-1) - config.X_MIN.reshape(-1)))
        # print(x_0)
        # print(xstar_0)
        # #added
        ustar = [u.reshape(-1, 1) for u in ustar]
        xstar_0 = xstar_0.reshape(-1, 1)
        xstar, _ = EulerIntegrate(None, system, f, B, Bw, None, ustar, xstar_0, time_bound, time_step,with_tracking=False, CCM=True, RCCM=False)


    fig = plt.figure(figsize=(8.0, 5.0))
    if args.plot_type=='3D':
        ax = fig.gca(projection='3d')
    else:
        ax = fig.gca()

    if args.plot_type == 'time':
        cmap = plt.get_cmap('plasma')
        colors = [cmap(i) for i in np.linspace(0, 1, len(args.plot_dims))]

    x_closed_RCCM = []
    controls_RCCM = []
    x_closed_CCM = []
    controls_CCM = []
    errors_RCCM = []
    errors_CCM = []

    controls_ref = []
    xinits = []
    x_star = [] #added
    xstar0 = [] #added
    for _ in range(args.nTraj):
        #added different xstar in the loop
        if args.ref_traj_many:
            x_0, xstar_0, ustar = config.system_reset(np.random.rand())
            ustar = [u.reshape(-1, 1) for u in ustar]
            xstar_0 = xstar_0.reshape(-1, 1)
            xstar, _ = EulerIntegrate(None, system, f, B, Bw, None, ustar, xstar_0, time_bound, time_step,with_tracking=False, CCM=True, RCCM=False)

        #added different xstar in the loop

        xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
        #xinit = args.x0scaler * (config.X_MIN + np.random.rand(len(config.X_MIN)).reshape(-1,1) * (config.X_MAX - config.X_MIN)) #xstar_0 + xe_0.reshape(-1,1)
        if args.init_same:
            xinit = 1.1 * xstar_0
        else:
            xinit = xstar_0 + xe_0.reshape(-1, 1)
        #print(xinit)
        xinits.append(xinit)
        if args.pretrained_RCCM:
            x_RCCM, u_RCCM = EulerIntegrate(controller_RCCM, system, f, B, Bw, xstar,ustar,xinit,time_bound,time_step,with_tracking=True,sigma=args.sigma,CCM = False, RCCM = True)
            x_closed_RCCM.append(x_RCCM)
            controls_RCCM.append(u_RCCM)
        if args.pretrained_CCM:
            x_CCM, u_CCM = EulerIntegrate(controller_CCM, system, f, B, Bw, xstar, ustar, xinit, time_bound, time_step,with_tracking=True, sigma=args.sigma,CCM = True, RCCM = False)
            x_closed_CCM.append(x_CCM)
            controls_CCM.append(u_CCM)

        x_star.append(xstar)
        xstar0.append(xstar_0)
        controls_ref.append(ustar)

    for n_traj in range(args.nTraj):
        if args.pretrained_RCCM:
            initial_dist_RCCM = np.sqrt(((x_closed_RCCM[n_traj][0] - x_star[n_traj][0])**2).sum())
            errors_RCCM.append([np.sqrt(((x - xs) ** 2).sum()) / initial_dist_RCCM for x, xs in zip(x_closed_RCCM[n_traj][:-1], x_star[n_traj][:-1])])  # xstar
        if args.pretrained_CCM:
            initial_dist_CCM = np.sqrt(((x_closed_CCM[n_traj][0] - x_star[n_traj][0]) ** 2).sum())
            errors_CCM.append([np.sqrt(((x - xs) ** 2).sum()) / initial_dist_CCM for x, xs in zip(x_closed_CCM[n_traj][:-1], x_star[n_traj][:-1])])
        #errors.append([np.sqrt(((x - xs) ** 2).sum()) for x, xs in zip(x_closed[n_traj][:-1], x_star[n_traj][:-1])])  # xstar [:-1]


        if args.plot_type=='2D':
            if args.pretrained_RCCM:
                plt.plot([x[args.plot_dims[0],0] for x in x_closed_RCCM[n_traj]], [x[args.plot_dims[1],0] for x in x_closed_RCCM[n_traj]], 'g', label='closed-loop traj RCCM' if n_traj==0 else None)
            if args.pretrained_CCM:
                plt.plot([x[args.plot_dims[0],0] for x in x_closed_CCM[n_traj]], [x[args.plot_dims[1],0] for x in x_closed_CCM[n_traj]], 'k-.', label='closed-loop traj CCM' if n_traj==0 else None)
        elif args.plot_type=='3D':
            if not args.pretrained_RCCM:
                plt.plot([x[args.plot_dims[0],0] for x in x_closed_RCCM[n_traj]], [x[args.plot_dims[1],0] for x in x_closed_RCCM[n_traj]], [x[args.plot_dims[2],0] for x in x_closed_RCCM[n_traj]], 'g', label='closed-loop traj_RCCM' if n_traj==0 else None)
            if args.pretrained_CCM:
                plt.plot([x[args.plot_dims[0], 0] for x in x_closed_CCM[n_traj]],[x[args.plot_dims[1], 0] for x in x_closed_CCM[n_traj]],[x[args.plot_dims[2], 0] for x in x_closed_CCM[n_traj]], 'k',label='closed-loop traj_CCM' if n_traj == 0 else None)
        elif args.plot_type=='time':
            for i, plot_dim in enumerate(args.plot_dims):
                if not args.pretrained_RCCM:
                    plt.plot(t, [x[plot_dim,0] for x in x_closed_RCCM[n_traj]][:-1], color=colors[i],label='closed-loop traj_RCCM' if i==0 and n_traj ==0 else None)
                if args.pretrained_CCM:
                    plt.plot(t, [x[plot_dim, 0] for x in x_closed_CCM[n_traj]][:-1], color=colors[i],label='closed-loop traj_CCM' if i==0 and n_traj ==0 else None)
        elif args.plot_type=='error':
            if args.pretrained_RCCM:
                plt.plot(t, [np.log(np.sqrt(((x[0:]-xs[0:])**2).sum())) for x, xs in zip(x_closed_RCCM[n_traj][:-1],  x_star[n_traj][:-1])], 'g', label='RCCM' if n_traj==0 else None)
                plt.plot(t, np.repeat(alpha, len(t)), 'm-.',
                         label='RCCM Tube Size' if n_traj == 0 else None)  # ,marker='${}$'.format(alpha) if n_traj==0 else None , markersize=100
                #plt.plot(t, np.repeat(np.log(mu), len(t)), 'r-.',label='mu' if n_traj == 0 else None)
            if args.pretrained_CCM:
                plt.plot(t, [np.log(np.sqrt(((x[0:]-xs[0:])**2).sum())) for x, xs in zip(x_closed_CCM[n_traj][:-1],  x_star[n_traj][:-1])], 'k', label='CCM' if n_traj==0 else None)
                plt.plot(t, np.repeat(np.log(CCM_tube), len(t)), 'b--', label='CCM Tube Size' if n_traj == 0 else None)


        elif args.plot_type=='controller':
            if args.pretrained_RCCM:
                plt.plot(t, [np.sqrt(((u - us) ** 2).sum()) for u, us in zip(controls_RCCM[n_traj], controls_ref[n_traj])], 'g', label='RCCM' if n_traj==0 else None)
            if args.pretrained_CCM:
                plt.plot(t, [np.sqrt(((u - us) ** 2).sum()) for u, us in zip(controls_CCM[n_traj], controls_ref[n_traj])],'k', label='CCM' if n_traj==0 else None)
        elif args.plot_type =='composition':
            if args.pretrained_RCCM:
                plt.plot(t, [np.log(np.sqrt(((np.append(x[0:], u) - np.append(xs[0:],us)) ** 2).sum())) for u, us , x , xs in zip(controls_RCCM[n_traj], controls_ref[n_traj],x_closed_RCCM[n_traj][:-1],x_star[n_traj][:-1])], 'g', label='closed-loop traj_RCCM' if n_traj==0 else None)
                plt.plot(t, np.repeat(alpha, len(t)), 'm-.',label='RCCM Tube Size' if n_traj == 0 else None)  # ,marker='${}$'.format(alpha) if n_traj==0 else None , markersize=100
                #plt.plot(t, np.repeat(mu, len(t)), 'r-.',label='mu' if n_traj == 0 else None)
            if args.pretrained_CCM:
                plt.plot(t, [np.log(np.sqrt(((np.append(x[0:], u) - np.append(xs, us)) ** 2).sum())) for u, us, x, xs in zip(controls_CCM[n_traj], controls_ref[n_traj], x_closed_CCM[n_traj][:-1],x_star[n_traj][:-1])], 'k', label='closed-loop traj_CCM' if n_traj == 0 else None)
                plt.plot(t, np.repeat(CCM_tube, len(t)), 'b--', label='CCM Tube Size' if n_traj == 0 else None)
        elif args.plot_type == 'STD_DEV':
            fig, ax = plt.subplots() #TO BE CHANGED
            x = np.linspace(0, 2 * np.pi, 50)
            y = np.sin(x) + np.random.randn(len(x)) * 0.03
            yerr0 = y - (0.1 + np.random.randn(len(x)) * 0.03)
            yerr1 = y + (0.1 + np.random.randn(len(x)) * 0.03)
            ax.plot(x, y, color='C0')
            plt.fill_between(x, yerr0, yerr1, color='C0', alpha=0.5)


        if args.plot_type == '2D':
            plt.plot([x[args.plot_dims[0], 0] for x in x_star[n_traj]],
                     [x[args.plot_dims[1], 0] for x in x_star[n_traj]], 'm--', label='Reference' if n_traj==0 else None)
            plt.plot(xstar0[n_traj][args.plot_dims[0]], xstar0[n_traj][args.plot_dims[1]], 'ro',
                     markersize=3.)  # xstar_0
            plt.plot(xinits[n_traj][args.plot_dims[0]], xinits[n_traj][args.plot_dims[1]], 'bs', markersize=3.)
            plt.xlabel("x")
            plt.ylabel("y")
        elif args.plot_type == '3D':
            plt.plot([x[args.plot_dims[0], 0] for x in x_star[n_traj]],
                     [x[args.plot_dims[1], 0] for x in x_star[n_traj]],
                     [x[args.plot_dims[2], 0] for x  in x_star[n_traj]], 'm--', label='Reference' if n_traj==0 else None)
            plt.plot(xstar0[n_traj][args.plot_dims[0]], xstar0[n_traj][args.plot_dims[1]],
                     xstar0[n_traj][args.plot_dims[2]], 'ro', markersize=3.)
            plt.plot(xinits[n_traj][args.plot_dims[0]], xinits[n_traj][args.plot_dims[1]],
                     xinits[n_traj][args.plot_dims[2]], 'bs', markersize=3.)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        elif args.plot_type == 'time':
            for plot_dim in args.plot_dims:
                plt.plot(t, [x[plot_dim, 0] for x in x_star[n_traj]][:-1], 'm')
            plt.xlabel("t")
            plt.ylabel("x")
        elif args.plot_type == 'error':
            plt.xlabel("t")
            plt.ylabel("error")
        elif args.plot_type == 'controller':
            plt.xlabel("t")
            plt.ylabel("control")
        elif args.plot_type == 'composition':
            plt.xlabel("t")
            plt.ylabel("error")

    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    plt.legend(frameon=True)
    plt.show()
    #plt.savefig(args.task + '_' + args.plot_type)
