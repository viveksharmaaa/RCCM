##https://stackoverflow.com/questions/51936145/block-lmi-in-cvxpy
import cvxpy as cp
import numpy as np
import torch
import argparse as args
from torch.autograd import grad
import importlib
import sys
# sys.path.append('log_QUADROTOR_9D_0711/systems')
# sys.path.append('log_QUADROTOR_9D_0711/configs')
# sys.path.append('log_QUADROTOR_9D_0711/models')
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')



args.task ='QUADROTOR_9D'
args.w_lb = 0.1
args.use_cuda = False
args.num_train = 131072
args.num_test = 32768
args.w_lb = 0.1

mu = torch.nn.Linear(1,1,bias=False)
alpha = torch.nn.Linear(1,1,bias=False)
#
alpha.weight = torch.nn.Parameter(torch.tensor([10.0]))
mu.weight = torch.nn.Parameter(torch.tensor([10.0]))


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
f_func = system.f_func
B_func = system.B_func
Bw_func = system.Bw_func
C,D,C_ref,D_ref= system.DgDxu()
if hasattr(system, 'Bbot_func'):
    Bbot_func = system.Bbot_func


num_dim_distb = system.num_dim_distb
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
    x = x.requires_grad_()
    return x, xe, uref

x,xe,uref = data(X_tr)

model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func = get_model(num_dim_x, num_dim_control, w_lb=args.w_lb, use_cuda=args.use_cuda)

controller = torch.load('log_' + args.task + '_0901/model_best.pth.tar', map_location=torch.device('cpu'))
CCM = torch.load('log_' + args.task + '_0901/model_best.pth.tar', map_location=torch.device('cpu'))

def Jacobian_Matrix(M, x):
    # NOTE that this function assume that data are independent of each other
    # along the batch dimension.
    # M: B x m x m
    # x: B x n x 1
    # ret: B x m x m x n
    bs = x.shape[0]
    m = M.size(-1)
    n = x.size(1)
    J = torch.zeros(bs, m, m, n).type(x.type())
    for i in range(m):
        for j in range(m):
            J[:, i, j, :] = grad(M[:, i, j].sum(), x, create_graph=True)[0].squeeze(-1)
    return J

def Jacobian(f, x):
    # NOTE that this function assume that data are independent of each other
    f = f + 0. * x.sum() # to avoid the case that f is independent of x
    # f: B x m x 1
    # x: B x n x 1
    # ret: B x m x n
    bs = x.shape[0]
    m = f.size(1)
    n = x.size(1)
    J = torch.zeros(bs, m, n).type(x.type())
    for i in range(m):
        J[:, i, :] = grad(f[:, i, 0].sum(), x, create_graph=True)[0].squeeze(-1)
    return J

def weighted_gradients(W, v, x, detach=False):
    # v, x: bs x n x 1
    # DWDx: bs x n x n x n
    assert v.size() == x.size()
    bs = x.shape[0]
    if detach:
        return (Jacobian_Matrix(W, x).detach() * v.view(bs, 1, 1, -1)).sum(dim=3)
    else:
        return (Jacobian_Matrix(W, x) * v.view(bs, 1, 1, -1)).sum(dim=3)

if 'Bbot_func' not in locals():
    def Bbot_func(x): # columns of Bbot forms a basis of the null space of B^T
        bs = x.shape[0]
        Bbot = torch.cat((torch.eye(num_dim_x-num_dim_control, num_dim_x-num_dim_control),
            torch.zeros(num_dim_control, num_dim_x-num_dim_control)), dim=0)
        if args.use_cuda:
            Bbot = Bbot.cuda()
        Bbot.unsqueeze(0)
        return Bbot.repeat(bs, 1, 1)

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

def forward(x, xref, uref, _lambda, verbose=False, acc=False, detach=False):
    # x: bs x n x 1
    bs = x.shape[0]
    W = W_func(x)
    M = torch.inverse(W)
    f = f_func(x)
    B = B_func(x)
    Bw = Bw_func(x) #added
    DfDx = Jacobian(f, x)
    DBDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control).type(x.type())
    for i in range(num_dim_control):
        DBDx[:,:,:,i] = Jacobian(B[:,:,i].unsqueeze(-1), x)

    DBwDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_distb).type(x.type()) #added
    for i in range(num_dim_distb):
        DBwDx[:,:,:,i] = Jacobian(Bw[:,:,i].unsqueeze(-1), x)

    _Bbot = Bbot_func(x)
    u = u_func(x, x - xref, uref) # u: bs x m x 1 # TODO: x - xref
    K = Jacobian(u, x)
    w = torch.rand(bs,num_dim_distb).unsqueeze(-1) #added for dot_x for computing dot_M

    A = DfDx + sum([u[:, i, 0].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i] for i in range(num_dim_control)]) + sum([torch.rand(1) * DBwDx[:, :, :, i] for i in range(num_dim_distb)]) # DBwDx * torch.rand(num_dim_distb) # w = torch.rand(1)?

    si = C.repeat(bs, 1, 1) + D.repeat(bs, 1, 1).matmul(K)

    dot_x = f + B.matmul(u) + Bw.matmul(w) #added
    dot_M = weighted_gradients(M, dot_x, x, detach=detach) # DMDt
    dot_W = weighted_gradients(W, dot_x, x, detach=detach) # DWDt

    Contraction = dot_M + (A + B.matmul(K)).transpose(1, 2).matmul(M) + M.matmul(A + B.matmul(K)) + _lambda * M # n x n

    LMI1 = cp.bmat([[Contraction.squeeze(0).detach().numpy(), M.matmul(Bw).squeeze(0).detach().numpy()],
                    [Bw.transpose(1, 2).matmul(M).squeeze(0).detach().numpy(), -1 * mu * np.identity(num_dim_distb)]])
    LMI2 = cp.bmat([[_lambda * M.squeeze(0).detach().numpy(),np.zeros((num_dim_x, num_dim_distb)),si.transpose(1, 2).squeeze(0).detach().numpy()],
                    [np.zeros((num_dim_distb,num_dim_x)),(alpha - mu) * np.identity(num_dim_distb),np.zeros((num_dim_distb, si.shape[1]))],
                    [si.squeeze(0).detach().numpy(),np.zeros((si.shape[1],num_dim_distb)), alpha * np.identity(C.shape[0])]])

    LMI1_ = []
    LMI2_ = []

    for i in range(bs):
        LMI1_.append(cp.bmat([[Contraction[i].detach().numpy(), M[i].matmul(Bw[i]).detach().numpy()],
                    [Bw.transpose(1, 2)[i].matmul(M[i]).detach().numpy(), -1 * mu * np.identity(num_dim_distb)]]))
        LMI2_.append(cp.bmat([[_lambda * M[i].detach().numpy(),np.zeros((num_dim_x, num_dim_distb)),si.transpose(1, 2)[i].detach().numpy()],
                    [np.zeros((num_dim_distb,num_dim_x)),(alpha - mu) * np.identity(num_dim_distb),np.zeros((num_dim_distb, si.shape[1]))],
                    [si[i].detach().numpy(),np.zeros((si.shape[1],num_dim_distb)), alpha * np.identity(C.shape[0])]]))

    constraints = [0 <= alpha] #[0 <= alpha, alpha <= 5]
    #constraints = [0 <= mu, mu <= 5]
    constraints += [LMI1_[i] << 0 for i in range(bs)]
    constraints += [LMI2_[i] >> 0 for i in range(bs)]
    constraints += [LMI1_[i] == LMI1_[i].T for i in range(bs)]
    constraints += [LMI2_[i] == LMI2_[i].T for i in range(bs)]


    prob = cp.Problem(cp.Minimize(alpha),constraints)
    prob.solve(solver=cp.MOSEK)








# Problem data.
m = 30
n = 20
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Construct the problem.
x = cp.Variable(n)

# right code
alpha = cp.Variable()
mu = cp.Variable()
cons1 = LMI1 <= 0
cons2 = LMI2 >= 0
cons3 = 0 <= alpha
cons4 = alpha <=5

optprob = cp.Problem(cp.Minimize(alpha), constraints=[cons1, cons2, cons3,cons4])
optprob.solve(solver=cp.MOSEK)

with torch.no_grad():
    model_W[0].weight.copy_(CCM['model_W']['0.weight'])
    model_W[0].bias.copy_(CCM['model_W']['0.bias'])
    model_W[2].weight.copy_(CCM['model_W']['2.weight'])
    model_Wbot[0].weight.copy_(CCM['model_Wbot']['0.weight'])
    model_Wbot[0].bias.copy_(CCM['model_Wbot']['0.bias'])
    model_Wbot[2].weight.copy_(CCM['model_Wbot']['2.weight'])

    model_u_w1[0].weight.copy_(controller['model_u_w1']['0.weight'])
    model_u_w1[0].bias.copy_(controller['model_u_w1']['0.bias'])
    model_u_w1[2].weight.copy_(controller['model_u_w1']['2.weight'])
    model_u_w1[2].bias.copy_(controller['model_u_w1']['2.bias'])
    model_u_w2[0].weight.copy_(controller['model_u_w2']['0.weight'])
    model_u_w2[0].bias.copy_(controller['model_u_w2']['0.bias'])
    model_u_w2[2].weight.copy_(controller['model_u_w2']['2.weight'])
    model_u_w2[2].bias.copy_(controller['model_u_w2']['2.bias'])