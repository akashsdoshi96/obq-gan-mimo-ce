import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data
from torch.nn import functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import sys
import scipy.io as sio
import pdb
import copy
%matplotlib inline

#Wireless Parameters
N_t = 64
N_r = 16
latent_dim = 65
conditional = 1
channel_model = 'A'

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor

class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# custom con2d, because pytorch don't have "padding='same'" option.
def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):

    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows - input_rows)
    padding_rows = max(0, (out_rows - 1) * stride[0] + (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    # same for padding_cols
    input_cols = input.size(3)
    filter_cols = weight.size(3)
    effective_filter_size_cols = (filter_cols - 1) * dilation[0] + 1
    out_cols = (input_cols + stride[0] - 1) // stride[0]
    padding_needed = max(0, (out_cols - 1) * stride[0] + effective_filter_size_cols - input_cols)
    padding_cols = max(0, (out_cols - 1) * stride[0] + (filter_cols - 1) * dilation[0] + 1 - input_cols)
    cols_odd = (padding_cols % 2 != 0)
    if rows_odd or cols_odd:
        input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])
    return F.conv2d(input, weight, bias, stride, padding=(padding_rows // 2, padding_cols // 2),dilation=dilation, groups=groups)
    
class View(torch.nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
    
class Concatenate(torch.nn.Module):
    def __init__(self, axis):
        super(Concatenate, self).__init__()
        self.axis = axis

    def forward(self, x):
        return torch.cat((x[0],x[1]),self.axis)
    
length = int(N_t/4)
breadth = int(N_r/4)
class Generator(torch.nn.Module):
    def __init__(self,mb_size):
        super(Generator,self).__init__()
        self.mb_size = mb_size
        self.embedding = torch.nn.Embedding(2,10)
        self.linear_e = torch.nn.Linear(10,length*breadth)
        self.view_e = View([mb_size,1,length,breadth])
        self.linear_g1 = torch.nn.Linear(latent_dim, 127*length*breadth)
        self.relu = torch.nn.ReLU()
        self.view_g = View([mb_size,127,length,breadth])
        self.upsample_g = torch.nn.Upsample(scale_factor=2)
        self.batchnorm_g1 = torch.nn.BatchNorm2d(128,momentum=0.8)
        self.batchnorm_g2 = torch.nn.BatchNorm2d(128,momentum=0.8)
        self.batchnorm_g3 = torch.nn.BatchNorm2d(128,momentum=0.8)
        self.conv2d_g1 = Conv2d(128,128,4,bias=False)
        self.conv2d_g2 = Conv2d(128,128,4,bias=False)
        self.conv2d_g4 = Conv2d(128,128,4,bias=False)
        self.conv2d_g3 = Conv2d(128,2,4,bias=False)
        self.concat = Concatenate(1)
        
    def forward(self,z,c):
        c_e = self.embedding(c)
        c_l = self.linear_e(c_e)
        c_v = self.view_e(c_l)
        
        z1 = self.linear_g1(z)
        z1 = self.relu(z1)
        z1_v = self.view_g(z1)
        
        z_c = self.concat([z1_v,c_v])
        z_c_1 = self.upsample_g(z_c)
        z_c_1 = self.conv2d_g1(z_c_1)
        z_c_1 = self.batchnorm_g1(z_c_1)
        z_c_1 = self.relu(z_c_1)
        z_c_2 = self.upsample_g(z_c_1)
        z_c_2 = self.conv2d_g2(z_c_2)
        z_c_2 = self.batchnorm_g2(z_c_2)
        z_c_2 = self.relu(z_c_2)
        z_c_2 = self.conv2d_g4(z_c_2)
        z_c_2 = self.batchnorm_g3(z_c_2)
        z_c_2 = self.relu(z_c_2)
        output = self.conv2d_g3(z_c_2)
        
        return output
    
G_test = torch.nn.Sequential(
    torch.nn.Linear(latent_dim, 128*length*breadth),
    torch.nn.ReLU(),
    View([1,128,length,breadth]),
    torch.nn.Upsample(scale_factor=2),
    Conv2d(128,128,4,bias=False),
    torch.nn.BatchNorm2d(128,momentum=0.8),
    torch.nn.ReLU(),
    torch.nn.Upsample(scale_factor=2),
    Conv2d(128,128,4,bias=False),
    torch.nn.BatchNorm2d(128,momentum=0.8),
    torch.nn.ReLU(),
    Conv2d(128,2,4,bias=False),
)

def fft_op(H_extracted):
    for i in range(H_extracted.shape[0]):
        H_extracted[i] = np.transpose(np.matmul(np.matmul(A_R.conj().T,H_extracted[i].T,dtype='complex64'),A_T))
    return H_extracted

if conditional:
    H_org_A = sio.loadmat("../data/H_16x64_MIMO_CDL_A_ULA_clean.mat")
    H_ex_A = H_org_A['hest']
    H_extracted_A = np.transpose(copy.deepcopy(H_ex_A),(2,1,0))
    H_extracted_A = fft_op(H_extracted_A)
    H_org_B = sio.loadmat("../data/H_16x64_MIMO_CDL_B_ULA_clean.mat")
    H_ex_B = H_org_B['hest']
    H_extracted_B = np.transpose(copy.deepcopy(H_ex_B),(2,1,0))
    H_extracted_B = fft_op(H_extracted_B)
    H_org_C = sio.loadmat("../data/H_16x64_MIMO_CDL_C_ULA_clean.mat")
    H_ex_C = H_org_C['hest']
    H_extracted_C = np.transpose(copy.deepcopy(H_ex_C),(2,1,0))
    H_extracted_C = fft_op(H_extracted_C)
    H_org_D = sio.loadmat("../data/H_16x64_MIMO_CDL_D_ULA_clean.mat")
    H_ex_D = H_org_D['hest']
    H_extracted_D = np.transpose(copy.deepcopy(H_ex_D),(2,1,0))
    H_extracted_D = fft_op(H_extracted_D)
    H_org_E = sio.loadmat("../data/H_16x64_MIMO_CDL_E_ULA_clean.mat")
    H_ex_E = H_org_E['hest']
    H_extracted_E = np.transpose(copy.deepcopy(H_ex_E),(2,1,0))
    H_extracted_E = fft_op(H_extracted_E)

    H_extracted = np.concatenate([H_extracted_A,H_extracted_B,H_extracted_C,H_extracted_D,H_extracted_E],axis=0)

    img_np_real = np.real(H_extracted)
    img_np_imag = np.imag(H_extracted)
    mu_real = np.mean(img_np_real,axis=0)
    mu_imag = np.mean(img_np_imag,axis=0)
    std_real = np.std(img_np_real,axis=0)
    std_imag = np.std(img_np_imag,axis=0)
    
    H_org_A = sio.loadmat("../data/H_16x64_MIMO_CDL_A_ULA_test.mat")
    H_ex_A = H_org_A['hest']
    H_extracted_A = np.transpose(copy.deepcopy(H_ex_A),(2,1,0))
    H_org_B = sio.loadmat("../data/H_16x64_MIMO_CDL_B_ULA_test.mat")
    H_ex_B = H_org_B['hest']
    H_extracted_B = np.transpose(copy.deepcopy(H_ex_B),(2,1,0))
    H_org_C = sio.loadmat("../data/H_16x64_MIMO_CDL_C_ULA_test.mat")
    H_ex_C = H_org_C['hest']
    H_extracted_C = np.transpose(copy.deepcopy(H_ex_C),(2,1,0))
    H_org_D = sio.loadmat("../data/H_16x64_MIMO_CDL_D_ULA_test.mat")
    H_ex_D = H_org_D['hest']
    H_extracted_D = np.transpose(copy.deepcopy(H_ex_D),(2,1,0))
    H_org_E = sio.loadmat("../data/H_16x64_MIMO_CDL_E_ULA_test.mat")
    H_ex_E = H_org_E['hest']
    H_extracted_E = np.transpose(copy.deepcopy(H_ex_E),(2,1,0))

    H_extracted = np.concatenate([H_extracted_A,H_extracted_B,H_extracted_C,H_extracted_D,H_extracted_E],axis=0)
    H_extracted = fft_op(H_extracted)
    H_ex = np.concatenate([H_ex_A,H_ex_B,H_ex_C,H_ex_D,H_ex_E],axis=2)
    
    size = int(H_extracted.shape[0]/5)
    CDL_NLOS = np.zeros((3*size,1))
    CDL_LOS = np.ones((2*size,1))
    CDL = np.concatenate((CDL_NLOS,CDL_LOS),axis=0)
    
else:
    H_org = sio.loadmat("../data/H_16x64_MIMO_CDL_%s_ULA_clean.mat"%channel_model)
    H_ex = H_org['hest']
    H_extracted = np.transpose(copy.deepcopy(H_ex),(2,1,0))
    dft_basis = sio.loadmat("../../data/dft_basis.mat")
    A_T = dft_basis['A1']/np.sqrt(N_t)
    A_R = dft_basis['A2']/np.sqrt(N_r)
    for i in range(H_ex.shape[2]):
        H_extracted[i] = np.transpose(np.matmul(np.matmul(A_R.conj().T,H_extracted[i].T,dtype='complex64'),A_T))

    img_np_real = np.real(H_extracted)
    img_np_imag = np.imag(H_extracted)

    mu_real = np.mean(img_np_real,axis=0)
    mu_imag = np.mean(img_np_imag,axis=0)
    std_real = np.std(img_np_real,axis=0)
    std_imag = np.std(img_np_imag,axis=0)
    
    H_org = sio.loadmat("../data/H_16x64_MIMO_CDL_%s_ULA_test.mat"%channel_model)
    H_ex = H_org['hest']
    H_extracted = np.transpose(copy.deepcopy(H_ex),(2,1,0))
    for i in range(H_ex.shape[2]):
        H_extracted[i] = np.transpose(np.matmul(np.matmul(A_R.conj().T,H_extracted[i].T,dtype='complex64'),A_T))
    img_np_real = np.real(H_extracted)
    img_np_imag = np.imag(H_extracted)
    img_np_real = (img_np_real - mu_real)/std_real
    img_np_imag = (img_np_imag - mu_imag)/std_imag
    
N_s = N_r
N_rx_rf = N_r
Nbit_t = 6
Nbit_r = 2
angles_t = np.linspace(0,2*np.pi,2**Nbit_t,endpoint=False)
angles_r = np.linspace(0,2*np.pi,2**Nbit_r,endpoint=False)

def training_precoder(N_t,N_s):
    angle_index = np.random.choice(len(angles_t),(N_t,N_s))
    return (1/np.sqrt(N_t))*np.exp(1j*angles_t[angle_index])

def training_combiner(N_r,N_rx_rf):
    angle_index = np.random.choice(len(angles_r),(N_r,N_rx_rf))
    W = (1/np.sqrt(N_r))*np.exp(1j*angles_r[angle_index])
    return np.matrix(W).getH()

ntest = 20              
nrepeat = 5 #Different noise realizations
SNR_vec = range(-15,20,5)
alpha = 0.4
ct = 0
N_p = int(alpha*N_t)
qpsk_constellation = (1/np.sqrt(2))*np.array([1+1j,1-1j,-1+1j,-1-1j])
identity = np.identity(N_r)
