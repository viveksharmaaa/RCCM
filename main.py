import torch
from torch.autograd import grad
import torch.nn.functional as F

import importlib
import numpy as np
import time
from tqdm import tqdm

import os
import sys
import pickle
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')
import argparse

#added
# import argparse as args
# args.task = 'QUADROTOR_9D' #
# args.use_cuda = 'False'
# args.bs = 1024
# args.num_train = 131072
# args.num_test = 32768
# args.learning_rate = 0.001
# args.epochs = 15
# args.lr_step = 5
# args._lambda = 0.5
# args.w_ub = 10
# args.w_lb = 0.1
# args.use_cuda = 'False'
train_loss = []
test_loss = []
Alpha = []
Mu = []
Cont = []
Condition1 = []
Condition2 = []
alpha_grad =[]
mu_grad = []
#added

np.random.seed(1024)

parser = argparse.ArgumentParser(description="")
parser.add_argument('--task', type=str,
                        default='CAR', help='Name of the model.')
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false', help='Disable cuda.')
parser.set_defaults(use_cuda=False)
parser.add_argument('--bs', type=int, default=1024, help='Batch size.')
parser.add_argument('--num_train', type=int, default=131072, help='Number of samples for training.') # 4096 * 32
parser.add_argument('--num_test', type=int, default=32768, help='Number of samples for testing.') # 1024 * 32
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='Base learning rate.')
parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
parser.add_argument('--lr_step', type=int, default=5, help='')
parser.add_argument('--lambda', type=float, dest='_lambda', default=0.5, help='Convergence rate: lambda')
parser.add_argument('--w_ub', type=float, default=10, help='Upper bound of the eigenvalue of the dual metric.')
parser.add_argument('--w_lb', type=float, default=0.1, help='Lower bound of the eigenvalue of the dual metric.')
parser.add_argument('--log', type=str, help='Path to a directory for storing the log.')


args = parser.parse_args()

os.system('cp *.py '+args.log)
os.system('cp -r models/ '+args.log)
os.system('cp -r configs/ '+args.log)
os.system('cp -r systems/ '+args.log)

epsilon = args._lambda * 0.1
#mu = torch.tensor(3,requires_grad=False) #0.01 #added
mu = torch.nn.Linear(1,1,bias=False)
alpha = torch.nn.Linear(1,1,bias=False)

alpha.weight = torch.nn.Parameter(torch.tensor([10.0]))
mu.weight = torch.nn.Parameter(torch.tensor([10.0]))

#Initialize weights to uniform distribution
# torch.nn.init.uniform_(alpha.weight)
# torch.nn.init.uniform_(mu.weight)
print("alpha:",alpha.weight.item())
print("mu",mu.weight.item())

config = importlib.import_module('config_'+args.task)
X_MIN = config.X_MIN
X_MAX = config.X_MAX
U_MIN = config.UREF_MIN
U_MAX = config.UREF_MAX
XE_MIN = config.XE_MIN
XE_MAX = config.XE_MAX

system = importlib.import_module('system_'+args.task)
f_func = system.f_func
B_func = system.B_func
Bw_func = system.Bw_func
C,D,C_ref,D_ref= system.DgDxu()
#g_func =  system.g_func
num_dim_x = system.num_dim_x
num_dim_control = system.num_dim_control
num_dim_distb = system.num_dim_distb
if hasattr(system, 'Bbot_func'):
    Bbot_func = system.Bbot_func

model = importlib.import_module('model_'+args.task)
get_model = model.get_model

model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func = get_model(num_dim_x, num_dim_control, w_lb=args.w_lb, use_cuda=args.use_cuda)

# constructing datasets
def sample_xef():
    return (X_MAX-X_MIN) * np.random.rand(num_dim_x, 1) + X_MIN

def sample_x(xref):
     xe = (XE_MAX-XE_MIN) * np.random.rand(num_dim_x, 1) + XE_MIN
     x = xref + xe
     x[x>X_MAX] = X_MAX[x>X_MAX]
     x[x<X_MIN] = X_MIN[x<X_MIN]
     return x

def sample_uref():
    return (U_MAX-U_MIN) * np.random.rand(num_dim_control, 1) + U_MIN

def sample_full():
    xref = sample_xef()
    uref = sample_uref()
    x = sample_x(xref)
    return (x, xref, uref)

X_tr = [sample_full() for _ in range(args.num_train)]
X_te = [sample_full() for _ in range(args.num_test)]

if 'Bbot_func' not in locals():
    def Bbot_func(x): # columns of Bbot forms a basis of the null space of B^T
        bs = x.shape[0]
        Bbot = torch.cat((torch.eye(num_dim_x-num_dim_control, num_dim_x-num_dim_control),
            torch.zeros(num_dim_control, num_dim_x-num_dim_control)), dim=0)
        if args.use_cuda:
            Bbot = Bbot.cuda()
        Bbot.unsqueeze(0)
        return Bbot.repeat(bs, 1, 1)

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

N = 2 * 1024
def loss_pos_matrix_random_sampling(A):
    # A: bs x d x d   #added d = n + p
    # z: K x d
    z = torch.randn(N, A.size(-1)) #.cuda()
    z = z / z.norm(dim=1, keepdim=True)
    zTAz = (z.matmul(A) * z.view(1,N,-1)).sum(dim=2).view(-1)
    negative_index = zTAz.detach().cpu().numpy() < 0
    if negative_index.sum()>0:
        negative_zTAz = zTAz[negative_index]
        return -1.0 * (negative_zTAz.mean())
    else:
        return torch.tensor(0.).type(z.type()).requires_grad_()

def loss_pos_matrix_eigen_values(A):
    # A: bs x d x d
    eigv = torch.linalg.eigh(A,UPLO = 'U')[0].view(-1) #torch.symeig(A, eigenvectors=True)[0].view(-1)
    negative_index = eigv.detach().cpu().numpy() < 0
    negative_eigv = eigv[negative_index]
    return negative_eigv.norm()

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
    if detach:
        #Contraction = dot_M + (A + B.matmul(K)).transpose(1,2).matmul(M.detach()) + M.detach().matmul(A + B.matmul(K)) + 2 * _lambda * M.detach()
        Contraction = dot_M + (A + B.matmul(K)).transpose(1,2).matmul(M.detach()) + M.detach().matmul(A + B.matmul(K)) + _lambda * M.detach() #n x n

        #Cond1 (n+p) x (n+p)
        Cond1 = torch.cat((torch.cat((Contraction, M.detach().matmul(Bw)), 2), torch.cat((Bw.transpose(1, 2).matmul(M.detach()),-1 * mu(torch.ones(1)) * torch.eye(num_dim_distb).repeat(bs,1,1)), 2)), 1)
        Cond2 = torch.cat((torch.cat((_lambda * M.detach(), torch.zeros(bs, num_dim_x, num_dim_distb)), 2),torch.cat((torch.zeros(bs, num_dim_distb, num_dim_x), (alpha(torch.ones(1)) - mu(torch.ones(1))) *torch.eye(num_dim_distb).repeat(bs,1,1)), 2)), 1) - (1 / alpha(torch.ones(1))) * (torch.cat((si.transpose(1, 2), torch.zeros((bs, num_dim_distb, si.shape[1]))), 1)).matmul(torch.cat((si, torch.zeros((bs, si.shape[1], num_dim_distb))), 2))

        # Cond2 = torch.cat((torch.cat((_lambda * M.detach(), torch.zeros(bs, num_dim_x, num_dim_distb),si.transpose(1,2)), 2),
        #                        torch.cat((torch.zeros(bs, num_dim_distb, num_dim_x),
        #            (alpha(torch.ones(1)) - mu(torch.ones(1))) * torch.eye(num_dim_distb).repeat(bs,1,1),
        #            torch.zeros(bs, si.shape[1], num_dim_distb).transpose(1, 2)), 2),
        #                        torch.cat((si, torch.zeros(bs, si.shape[1], num_dim_distb),
        #            alpha(torch.ones(1)) * torch.eye(C.shape[0]).repeat(bs,1,1)), 2)),1)
    else:
        Contraction = dot_M + (A + B.matmul(K)).transpose(1,2).matmul(M) + M.matmul(A + B.matmul(K)) + _lambda * M
        #Contraction = dot_M + (A + B.matmul(K)).transpose(1, 2).matmul(M.detach()) + M.detach().matmul(A + B.matmul(K)) + 2 * _lambda * M.detach()
        Cond1 = torch.cat((torch.cat((Contraction, M.matmul(Bw)), 2), torch.cat((Bw.transpose(1, 2).matmul(M), -1 * mu(torch.ones(1)) * torch.eye(num_dim_distb).repeat(bs,1,1)), 2)), 1)

        Cond2 = torch.cat((torch.cat((_lambda * M, torch.zeros(bs, num_dim_x, num_dim_distb)), 2), torch.cat((torch.zeros(bs, num_dim_distb, num_dim_x),(alpha(torch.ones(1)) - mu(torch.ones(1))) * torch.eye(num_dim_distb).repeat(bs,1,1)), 2)), 1) - (1 / alpha(torch.ones(1))) * (torch.cat((si.transpose(1, 2), torch.zeros((bs, num_dim_distb, si.shape[1]))), 1)).matmul(torch.cat((si, torch.zeros((bs, si.shape[1], num_dim_distb))), 2))

        # Cond2 = torch.cat((torch.cat((_lambda * M, torch.zeros(bs, num_dim_x, num_dim_distb),si.transpose(1,2)), 2),
        #                        torch.cat((torch.zeros(bs, num_dim_distb, num_dim_x),
        #            (alpha(torch.ones(1)) - mu(torch.ones(1))) * torch.eye(num_dim_distb).repeat(bs,1,1),
        #            torch.zeros(bs, si.shape[1], num_dim_distb).transpose(1, 2)), 2),
        #                        torch.cat((si, torch.zeros(bs, si.shape[1], num_dim_distb),
        #            alpha(torch.ones(1)) * torch.eye(C.shape[0]).repeat(bs,1,1)), 2)),1)

    # #added

    # C1
    C1_inner = - weighted_gradients(W, f, x) + DfDx.matmul(W) + W.matmul(DfDx.transpose(1, 2)) + 2 * _lambda * W
    C1_LHS_1 = _Bbot.transpose(1, 2).matmul(C1_inner).matmul(_Bbot)  # this has to be a negative definite matrix

    # C2
    C2_inners = []
    C2s = []
    for j in range(num_dim_control):
        C2_inner = weighted_gradients(W, B[:, :, j].unsqueeze(-1), x) - (
                DBDx[:, :, :, j].matmul(W) + W.matmul(DBDx[:, :, :, j].transpose(1, 2)))
        C2 = _Bbot.transpose(1, 2).matmul(C2_inner).matmul(_Bbot)
        C2_inners.append(C2_inner)
        C2s.append(C2)

    loss = 0
    #loss += loss_pos_matrix_random_sampling(-Contraction - epsilon * torch.eye(Contraction.shape[-1]).unsqueeze(0).type(x.type()))
    loss += loss_pos_matrix_random_sampling(- Cond1  - epsilon * torch.eye(Cond1.shape[-1]).unsqueeze(0).type(x.type())) #loss_pos_matrix_eigen_values
    loss += loss_pos_matrix_random_sampling(Cond2)
    loss += alpha(torch.ones(1))
    #loss += mu(torch.ones(1))
    #loss += loss_pos_matrix_random_sampling(W - args.w_lb * torch.eye(W.shape[-1]).unsqueeze(0).type(x.type()))
    loss += loss_pos_matrix_random_sampling(-C1_LHS_1 - epsilon * torch.eye(C1_LHS_1.shape[-1]).unsqueeze(0).type(x.type()))
    # loss += loss_pos_matrix_random_sampling(args.w_ub * torch.eye(W.shape[-1]).unsqueeze(0).type(x.type()) - W)
    loss += 1. * sum([1. * (C2 ** 2).reshape(bs, -1).sum(dim=1).mean() for C2 in C2s])

    if verbose: # was torch.symeig(Contraction)[0].min(dim=1)[0].mean()
        print(torch.linalg.eigh(Contraction,UPLO = 'U')[0].min(dim=1)[0].mean(), torch.linalg.eigh(Contraction,UPLO = 'U')[0].max(dim=1)[0].mean(), torch.linalg.eigh(Contraction,UPLO = 'U')[0].mean()) #torch.linalg.eigvalsh
    if acc:
        #return loss, loss_pos_matrix_random_sampling(-Contraction).item(),loss_pos_matrix_random_sampling(-Cond1).item(),loss_pos_matrix_random_sampling(Cond2).item()#,loss_pos_matrix_random_sampling(-C1_LHS_1).item() #,1. * sum([1.*(C2**2).reshape(bs,-1).sum(dim=1).mean() for C2 in C2s])
        return loss, loss_pos_matrix_eigen_values(Contraction).item(), loss_pos_matrix_eigen_values(Cond1).item(), loss_pos_matrix_eigen_values(-Cond2).item()
        #return loss, ((torch.linalg.eigh(Contraction,UPLO = 'U')[0]>= 0).sum(dim=1) == 0).cpu().detach().numpy(), (
                     #(torch.linalg.eigh(Cond1,UPLO = 'U')[0] >= 0).sum(dim=1) == 0).cpu().detach().numpy(), ((torch.linalg.eigh(-Cond2,UPLO = 'U')[0] >= 0).sum(dim=1) == 0).cpu().detach().numpy()
        #return loss, ((torch.symeig(Contraction)[0]>=0).sum(dim=1)==0).cpu().detach().numpy(), ((torch.symeig(C1_LHS_1)[0]>=0).sum(dim=1)==0).cpu().detach().numpy(), sum([1.*(C2**2).reshape(bs,-1).sum(dim=1).mean() for C2 in C2s]).item()
    else:
        return loss, None, None, None #, None #, None

optimizer = torch.optim.Adam(list(model_W.parameters()) + list(model_Wbot.parameters()) + list(model_u_w1.parameters()) + list(model_u_w2.parameters())  +  list(alpha.parameters()) + list(mu.parameters()), lr=args.learning_rate)

#optimizer_ref = torch.optim.Adam(list(alpha.parameters()) + list(mu.parameters()), lr=args.learning_rate)
#optimizer = torch.optim.Adam([{'params':list(model_W.parameters())+ list(model_Wbot.parameters()) + list(model_u_w1.parameters()) + list(model_u_w2.parameters())},{'params':list(alpha.parameters()) + list(mu.parameters()),'lr': 1e-3}],lr=args.learning_rate)
def trainval(X, bs=args.bs, train=True, _lambda=args._lambda, acc=False, detach=False): # trainval a set of x
    #torch.autograd.set_detect_anomaly(True)

    if train:
        indices = np.random.permutation(len(X))
    else:
        indices = np.array(list(range(len(X))))

    total_loss = 0
    total_p1 = 0
    total_p2 = 0
    total_l3 = 0
    total_c1 = 0
    total_c2 = 0

    if train:
        _iter = tqdm(range(len(X) // bs))
    else:
        _iter = range(len(X) // bs)
    for b in _iter:
        start = time.time()
        x = []; xref = []; uref = [];
        for id in indices[b*bs:(b+1)*bs]:
            if args.use_cuda:
                x.append(torch.from_numpy(X[id][0]).float().cuda())
                xref.append(torch.from_numpy(X[id][1]).float().cuda())
                uref.append(torch.from_numpy(X[id][2]).float().cuda())
            else:
                x.append(torch.from_numpy(X[id][0]).float())
                xref.append(torch.from_numpy(X[id][1]).float())
                uref.append(torch.from_numpy(X[id][2]).float())

        x, xref, uref = (torch.stack(d).detach() for d in (x, xref, uref))
        x = x.requires_grad_()

        start = time.time()

        loss, p1, p2, l3 = forward(x, xref, uref, _lambda=_lambda, verbose=False if not train else False, acc=acc, detach=detach)

        start = time.time()
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('backwad(): %.3f s'%(time.time() - start))

        total_loss += loss.item() * x.shape[0]
        if acc:
            total_p1 += p1 * x.shape[0] #p1.sum()
            total_p2 += p2 * x.shape[0]   #p2.sum()
            total_l3 += l3 * x.shape[0]  #l3 * x.shape[0]
            #total_c1 += c1 * x.shape[0]
            #total_c2 += c2 * x.shape[0]
    return total_loss / len(X), total_p1 / len(X), total_p2 / len(X), total_l3/ len(X) #, total_c1/ len(X), total_c2/ len(X)

best_acc = 0

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by every args.lr_step epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // args.lr_step))  # lr = lr * (0.1)^(epoch//lr_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(args.epochs):
    #adjust_learning_rate(optimizer, epoch)
    loss, _, _, _ = trainval(X_tr, train=True, _lambda=args._lambda, acc=False, detach=True if epoch < args.lr_step else False) #True if epoch < args.lr_step else False
    alpha_grad.append(alpha.weight.grad.item())
    mu_grad.append(mu.weight.grad.item())
    print("Training loss: ", loss)
    print("Gradient alpha/mu:", mu.weight.grad.item(), alpha.weight.grad.item())
    print("Learning Rate:",optimizer.param_groups[0]['lr'])
    print("Alpha/Mu:", alpha.weight.item(), mu.weight.item())
    train_loss.append(loss)
    loss, p1, p2, l3 = trainval(X_te, train=False, _lambda=args._lambda, acc=True, detach=False) #args.lambda  was 0
    test_loss.append(loss)
    print("Epoch %d: Testing loss/Contraction/Cond1/Cond2: "%epoch, loss, p1, p2, l3)
    Alpha.append(alpha.weight.item())
    Mu.append(mu.weight.item())
    Cont.append(p1)
    Condition1.append(p2)
    Condition2.append(l3)


    if l3+p2 >= best_acc:
        best_acc = l3 + p2
        filename = args.log+'/model_best.pth.tar'
        filename_controller = args.log+'/controller_best.pth.tar'
        torch.save({'args':args, 'precs':(loss, p1, p2, l3), 'model_W': model_W.state_dict(), 'model_Wbot': model_Wbot.state_dict(), 'model_u_w1': model_u_w1.state_dict(), 'model_u_w2': model_u_w2.state_dict(), 'alpha': alpha.state_dict(),'mu': mu.state_dict()}, filename)
        torch.save(u_func, filename_controller)

    if epoch == args.epochs-1 :
        with open(args.task + '.pkl', 'wb') as f:
            pickle.dump([train_loss, test_loss, Alpha,Mu,Cont,Condition1, Condition2,alpha_grad,mu_grad], f)


# for OPT_REF (offline optimization for refining state and input tubes)
#
# alpha.weight = torch.nn.Parameter(torch.tensor([10.0]))
# mu.weight = torch.nn.Parameter(torch.tensor([10.0]))
#
# alpha_grad_ref = []
# mu_grad_ref = []
# train_loss_ref = []
# test_loss_ref = []
# Cont_ref = []
# Condition1_ref = []
# Condition2_ref = []
# Alpha_ref = []
# Mu_ref = []
#
# for epoch in range(args.epochs):
#     loss, _, _, _ = trainval(X_tr, train=True, _lambda=args._lambda, acc=False,detach=True,refine=True)
#     alpha_grad_ref.append(alpha.weight.grad.item())
#     mu_grad_ref.append(mu.weight.grad.item())
#     print("Training Ref loss: ", loss)
#     print("Gradient Ref alpha/mu:", mu.weight.grad.item(), alpha.weight.grad.item())
#     print("Learning Ref Rate:",optimizer_ref.param_groups[0]['lr'])
#     print("Alpha-Ref/Mu-Ref:", alpha.weight.item(), mu.weight.item())
#     train_loss_ref.append(loss)
#     loss, p1, p2, l3 = trainval(X_te, train=False, _lambda=0., acc=True, detach=False,refine=True)
#     test_loss_ref.append(loss)
#     print("Epoch %d: Ref Testing loss/Contraction/Cond1/Cond2: "%epoch, loss, p1, p2, l3)
#     Alpha_ref.append(alpha.weight.item())
#     Mu_ref.append(mu.weight.item())
#     Cont_ref.append(p1)
#     Condition1_ref.append(p2)
#     Condition2_ref.append(l3)
#
#     if l3+p2 >= best_acc:
#         best_acc = l3 + p2
#         filename_ref = args.log+'/model_best_ref.pth.tar'
#         filename_controller_ref = args.log+'/controller_best_ref.pth.tar'
#         torch.save({'args':args, 'precs':(loss, p1, p2, l3), 'model_W': model_W.state_dict(), 'model_Wbot': model_Wbot.state_dict(), 'model_u_w1': model_u_w1.state_dict(), 'model_u_w2': model_u_w2.state_dict(), 'alpha_ref': alpha.state_dict(),'mu_ref': mu.state_dict()}, filename_ref)
#         torch.save(u_func, filename_controller_ref)
#
#     if epoch == args.epochs-1 :
#         with open(args.task + '_refined.pkl', 'wb') as f:
#             pickle.dump([train_loss_ref, test_loss_ref, Alpha_ref,Mu_ref,Cont_ref,Condition1_ref, Condition2_ref,alpha_grad_ref,mu_grad_ref], f)













