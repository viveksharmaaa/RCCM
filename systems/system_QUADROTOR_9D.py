import torch

num_dim_x = 9
num_dim_control = 3
num_dim_distb = 3 #added

g = 9.81;

# the following dynamics is converted from the original Matlab code
# we noticed that xc(10) is a dead state variable
# b_T =  [sin(xc(9)); -cos(xc(9))*sin(xc(8)); cos(xc(9))*cos(xc(8))];
#
# f_ctrl =     [xc(4:6);
#               [0;0;g] - xc(7)*b_T;
#               zeros(4,1)];
# B_ctrl = @(xc)[zeros(6,4);
#                eye(4)];

def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    x1, x2, x3, x4, x5, x6, x7, x8, x9 = [x[:,i,0] for i in range(num_dim_x)]
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0, 0] = x4
    f[:, 1, 0] = x5
    f[:, 2, 0] = x6

    f[:, 3, 0] = - x7 * torch.sin(x9)
    f[:, 4, 0] = x7 * torch.cos(x9) * torch.sin(x8)
    f[:, 5, 0] = g - x7 * torch.cos(x9) * torch.cos(x8)

    f[:, 6, 0] = 0
    f[:, 7, 0] = 0
    f[:, 8, 0] = 0

    return f

def DfDx_func(x):
    raise NotImplemented('NotImplemented')

def B_func(x):
    bs = x.shape[0]
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 6, 0] = 1
    B[:, 7, 1] = 1
    B[:, 8, 2] = 1
    return B

def Bw_func(x):  #For Tube Certified Trajectory Tracking
    bs = x.shape[0]
    Bw = torch.zeros(bs, num_dim_x, num_dim_distb).type(x.type())

    x1, x2, x3, x4, x5, x6, x7, x8, x9 = [x[:,i,0] for i in range(num_dim_x)]

    Bw[:, 3, 0] = 1
    Bw[:, 4, 1] = 1
    Bw[:, 5, 2] = 1
    return Bw

def DgDxu():
    # All states and inputs
    C = torch.cat((torch.eye(num_dim_x),torch.zeros(num_dim_control,num_dim_x)))
    D = torch.cat((torch.zeros(num_dim_x,num_dim_control),0.01 * torch.diag(torch.tensor([2.0,5.0,5.0]))))

    # All States only and no inputs
    #C = torch.eye(num_dim_x)
    #D = torch.zeros(num_dim_x,num_dim_control)

    # Only position states and inputs
    # C = torch.cat((torch.cat((torch.eye(3),torch.zeros(3,num_dim_x-3)),1),torch.zeros(num_dim_control,num_dim_x)))
    # D = torch.cat((torch.zeros(3,num_dim_control),0.01 * torch.diag(torch.tensor([2.0,5.0,5.0]))))

    # Only position states
    C_ref = torch.cat((torch.eye(3),torch.zeros(3,num_dim_x-3)),1)
    D_ref = torch.zeros(3,num_dim_control)

    # Refined States
    #C_ref = torch.cat((torch.cat((torch.eye(3),torch.zeros(3,num_dim_x-3)),1),torch.zeros(num_dim_control,num_dim_x)))
    #D_ref = torch.cat((torch.zeros(3,num_dim_control),0.01 * torch.diag(torch.tensor([2.0,5.0,5.0]))))

    return C,D,C_ref,D_ref

def DBDx_func(x):
    raise NotImplemented('NotImplemented')
