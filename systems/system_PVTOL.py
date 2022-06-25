import torch

num_dim_x = 6
num_dim_control = 2
num_dim_distb = 1


m = 0.486
g = 9.81
l = 0.25
J = 0.00383


def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    x, z, phi, vx, vz, phi_dot = [x[:,i,0] for i in range(num_dim_x)]
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0, 0] = vx * torch.cos(phi) - vz * torch.sin(phi)
    f[:, 1, 0] = vx * torch.sin(phi) + vz * torch.cos(phi)
    f[:, 2, 0] = phi_dot
    f[:, 3, 0] = vz * phi_dot - g * torch.sin(phi)
    f[:, 4, 0] = -1 * vx * phi_dot - g * torch.cos(phi)
    f[:, 5, 0] = 0
    return f

def DfDx_func(x):
    raise NotImplemented('NotImplemented')

def B_func(x):
    bs = x.shape[0]
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 4, 0] = 1/m
    B[:, 4, 1] = 1/m
    B[:, 5, 0] = l/J
    B[:, 5, 1] = -1 * l/J
    return B

def Bw_func(x):  #For Tube Certified Trajectory Tracking
    bs = x.shape[0]
    Bw = torch.zeros(bs, num_dim_x, num_dim_distb).type(x.type())

    x, z, phi, vx, vz, phi_dot = [x[:, i, 0] for i in range(num_dim_x)]

    Bw[:, 3, 0] = torch.cos(phi)
    Bw[:, 4, 0] = -1 * torch.sin(phi)

    return Bw

def DBDx_func(x):
    raise NotImplemented('NotImplemented')

def g_func(x):
    bs = x.shape[0]
    x, z, phi, vx, vz, phi_dot = [x[:, i, 0] for i in range(num_dim_x)]
    g = torch.zeros(bs, num_dim_x, 1).type(x.type())
    g[:, 0, 0] = x
    g[:, 1, 0] = z
    g[:, 2, 0] = phi
    g[:, 3, 0] = vx
    g[:, 4, 0] = vz
    g[:, 5, 0] = phi_dot

    return g

def DgDxu():
    # All states and inputs
    C = torch.cat((torch.eye(num_dim_x,num_dim_x),torch.zeros(num_dim_control,num_dim_x)))
    D = torch.cat((torch.zeros(num_dim_x,num_dim_control),torch.ones(num_dim_control,num_dim_control)))

    # Only position states and inputs
    #C = torch.cat((torch.cat((torch.ones(num_dim_control,num_dim_control),torch.zeros(2,4)),1),torch.zeros(num_dim_control,num_dim_x)))
    #D = torch.cat((torch.zeros(2, num_dim_control), torch.ones(num_dim_control, num_dim_control)))

    return C,D