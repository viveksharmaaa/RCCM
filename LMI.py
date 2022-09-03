import cvxpy as cp
import numpy as np
import torch
import argparse as args
import importlib
import sys
sys.path.append('log_QUADROTOR_9D_0711/systems')
sys.path.append('log_QUADROTOR_9D_0711/configs')
sys.path.append('log_QUADROTOR_9D_0711/models')

args.task ='QUADROTOR_9D'
args.w_lb = 0.1
args.use_cuda = False
args.num_train = 131072
args.num_test = 32768
args.w_lb = 0.1


system = importlib.import_module('system_'+args.task)
model = importlib.import_module('model_'+args.task)
config = importlib.import_module('config_'+args.task)

X_MIN = config.X_MIN
X_MAX = config.X_MAX
U_MIN = config.UREF_MIN
U_MAX = config.UREF_MAX
XE_MIN = config.XE_MIN
XE_MAX = config.XE_MAX

num_dim_x = system.num_dim_x
num_dim_control = system.num_dim_control
get_model = model.get_model
effective_dim_start = model.effective_dim_start
effective_dim_end = model.effective_dim_end

def sample_xef():
    return (X_MAX-X_MIN) * np.random.rand(num_dim_x, 1) + X_MIN

def sample_x(xref):
     xe = (XE_MAX-XE_MIN) * np.random.rand(num_dim_x, 1) + XE_MIN
     x = xref + xe
     x[x>X_MAX] = X_MAX[x>X_MAX]
     x[x<X_MIN] = X_MIN[x<X_MIN]
     return x,xe

def sample_uref():
    return (U_MAX-U_MIN) * np.random.rand(num_dim_control, 1) + U_MIN

def sample_full():
    xref = sample_xef()
    uref = sample_uref()
    x,xe = sample_x(xref)
    return (x, xe, uref)

X_tr = [sample_full() for _ in range(args.num_train)]
X_te = [sample_full() for _ in range(args.num_test)]

def data(X):
    x = []
    xe = []
    uref = []
    for id in range(len(X_tr)):
            x.append(torch.from_numpy(X_tr[id][0]).float())
            xe.append(torch.from_numpy(X_tr[id][1]).float())
            uref.append(torch.from_numpy(X_tr[id][2]).float())
    x, xe, uref = (torch.stack(d).detach() for d in (x, xe, uref))
    return x, xe, uref

x,xe,uref = data(X_tr)

model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func = get_model(num_dim_x, num_dim_control, w_lb=args.w_lb, use_cuda=args.use_cuda)

controller = torch.load('log_' + args.task + '_0801/model_best.pth.tar', map_location=torch.device('cpu'))
CCM = torch.load('log_' + args.task + '_0801/model_best.pth.tar', map_location=torch.device('cpu'))


def Wandu(x,xe,uref): #x.shape = torch.Size([1024, 9])
    bs = x.shape[0]
    x = x.squeeze(-1)
    model_W_new = np.matmul(np.tanh(np.matmul(x[:, effective_dim_start:effective_dim_end].detach().numpy(), CCM['model_W']['0.weight'].numpy().transpose()) + CCM['model_W']['0.bias'].numpy()), CCM['model_W']['2.weight'].numpy().transpose())
    model_W_new = model_W_new.reshape(bs,num_dim_x, num_dim_x)
    model_Wbot_new = np.matmul(np.tanh(np.matmul(x[:, effective_dim_start:effective_dim_end-num_dim_control].detach().numpy(), CCM['model_Wbot']['0.weight'].numpy().transpose()) + CCM['model_Wbot']['0.bias'].numpy()),CCM['model_Wbot']['2.weight'].numpy().transpose())
    model_Wbot_new = model_Wbot_new.reshape(bs, num_dim_x - num_dim_control, num_dim_x - num_dim_control)
    model_W_new[:,0:num_dim_x - num_dim_control, 0:num_dim_x - num_dim_control] = model_Wbot_new
    model_W_new[:,num_dim_x - num_dim_control::, 0:num_dim_x - num_dim_control] = 0
    model_W_new = np.matmul(model_W_new.transpose(0,2,1), model_W_new) + args.w_lb * np.eye(num_dim_x).reshape(1,num_dim_x, num_dim_x)

    x = x.unsqueeze(-1)
    dim = effective_dim_end - effective_dim_start
    model_u_w1_new = np.matmul(np.tanh(np.matmul(torch.cat([x[:, effective_dim_start:effective_dim_end, :], (x - xe)[:, effective_dim_start:effective_dim_end, :]],dim=1).squeeze(-1).detach().numpy(),controller['model_u_w1']['0.weight'].numpy().transpose()) + controller['model_u_w1']['0.bias'].numpy()), controller['model_u_w1']['2.weight'].numpy().transpose()) \
                     + controller['model_u_w1']['2.bias'].numpy()
    w1 = model_u_w1_new.reshape(bs, -1, num_dim_x)
    model_u_w2_new = np.matmul(np.tanh(np.matmul(torch.cat([x[:, effective_dim_start:effective_dim_end, :], (x - xe)[:, effective_dim_start:effective_dim_end, :]],dim=1).squeeze(-1).detach().numpy(), controller['model_u_w2']['0.weight'].numpy().transpose())
                                       + controller['model_u_w2']['0.bias'].numpy()), controller['model_u_w2']['2.weight'].numpy().transpose()) + controller['model_u_w2']['2.bias'].numpy()
    w2 = model_u_w2_new.reshape(bs, num_dim_control, -1)
    # x.shape = torch.Size([1024, 9, 1])
    u = np.matmul(w2, np.tanh(np.matmul(w1,xe.numpy()))) + uref.numpy()

    return model_W_new, u

# Problem data.
m = 30
n = 20
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Construct the problem.
x = cp.Variable(n)
alpha = cp.Variable(1)
objective = cp.Minimize(alpha)

#objective = cp.Minimize(cp.sum_squares(A @ x - b))
constraints = [0 <= x, x <= 1]
constraints = [0 <= alpha]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
print(x.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(constraints[0].dual_value)

#
# with torch.no_grad():
#     model_W[0].weight.copy_(CCM['model_W']['0.weight'])
#     model_W[0].bias.copy_(CCM['model_W']['0.bias'])
#     model_W[2].weight.copy_(CCM['model_W']['2.weight'])
#     model_Wbot[0].weight.copy_(CCM['model_Wbot']['0.weight'])
#     model_Wbot[0].bias.copy_(CCM['model_Wbot']['0.bias'])
#     model_Wbot[2].weight.copy_(CCM['model_Wbot']['2.weight'])
#
#     model_u_w1[0].weight.copy_(controller['model_u_w1']['0.weight'])
#     model_u_w1[0].bias.copy_(controller['model_u_w1']['0.bias'])
#     model_u_w1[2].weight.copy_(controller['model_u_w1']['2.weight'])
#     model_u_w1[2].bias.copy_(controller['model_u_w1']['2.bias'])
#     model_u_w2[0].weight.copy_(controller['model_u_w2']['0.weight'])
#     model_u_w2[0].bias.copy_(controller['model_u_w2']['0.bias'])
#     model_u_w2[2].weight.copy_(controller['model_u_w2']['2.weight'])
#     model_u_w2[2].bias.copy_(controller['model_u_w2']['2.bias'])