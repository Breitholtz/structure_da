import numpy as np
import torch
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
    
    
class XEncodeNet(nn.Module): ### raw data encoder: take X and output encoding of it
    def __init__(self, opt):
        super(GNet, self).__init__()
        self.use_g_encode = opt.use_g_encode
        if self.use_g_encode:
            G = np.zeros((opt.num_domain, opt.nt))
            for i in range(opt.num_domain):
                G[i] = opt.g_encode[str(i)]
            self.G = torch.from_numpy(G).float().to(device=opt.device)
        else:
            self.fc1 = nn.Linear(opt.num_domain, opt.nh)
            self.fc_final = nn.Linear(opt.nh, opt.nt)

    def forward(self, x):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        if self.use_g_encode:
            x = torch.matmul(x.float(), self.G)
        else:
            x = F.relu(self.fc1(x.float()))
            # x = nn.Dropout(p=p)(x)
            x = self.fc_final(x)
        return x
class FeatureNet(nn.Module): ### take x and z(encoded domain index and distance matrix); and output an encoding
    def __init__(self, opt):
        super(FeatureNet, self).__init__()

        nx, nh, nt, p = opt.nx, opt.nh, opt.nt, opt.p
        self.p = p

        self.fc1 = nn.Linear(nx, nh)
        self.fc2 = nn.Linear(nh * 2, nh * 2)
        self.fc3 = nn.Linear(nh * 2, nh * 2)
        self.fc4 = nn.Linear(nh * 2, nh * 2)
        self.fc_final = nn.Linear(nh * 2, nh)

        # here I change the input to fit the change dimension
        self.fc1_var = nn.Linear(nt, nh)
        self.fc2_var = nn.Linear(nh, nh)

    def forward(self, x, t):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)
            t = t.reshape(T * B, -1)

        x = F.relu(self.fc1(x))
        t = F.relu(self.fc1_var(t))
        t = F.relu(self.fc2_var(t))

        # combine feature in the middle
        x = torch.cat((x, t), dim=1)

        # main
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc_final(x)

        if re:
            return x.reshape(T, B, -1)
        else:
            return x
        
class PredNet(nn.Module):
    def __init__(self, opt):
        super(PredNet, self).__init__()

        #nh, nc = opt.nh, opt.nc
        nh, nc = 512, 2
        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)
        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)
        self.fc_final = nn.Linear(nh, nc)
        if True:#  opt.no_bn
            self.bn3 = Identity()
            self.bn4 = Identity()

    def forward(self, x, return_softmax=False):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc_final(x)
        x_softmax = F.softmax(x, dim=1)

        # x = F.log_softmax(x, dim=1)
        # x = torch.clamp_max(x_softmax + 1e-4, 1)
        # x = torch.log(x)
        x = torch.log(x_softmax + 1e-4)

        if re:
            x = x.reshape(T, B, -1)
            x_softmax = x_softmax.reshape(T, B, -1)

        if return_softmax:
            return x, x_softmax
        else:
            return x
