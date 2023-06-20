import torch, scipy, time, math
from torch.autograd import Variable
from torch.utils import data  # from resnet import FCN
from torch.utils.data import DataLoader 
import tqdm, psutil 
import tensorflow as tf 
import platform, glob, argparse
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
from PIL import Image
import cv2, nibabel as nb
import numpy as np
import numpy as np
from keras.optimizers import *
from skimage import io, transform
from os.path import splitext as split
from pynvml import *
from numpy import linalg as LA
#====================================================================================
print('\n CMIT 2023:  Deep learning Setup Test File -- if no module error messages are shown\n')
print('               you have to install by the usual command   pip install Xmodule ... Ke\n ')
#====================================================================================
start_time = time.time()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
s24=1014**2

if use_cuda:
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0);    info = nvmlDeviceGetMemoryInfo(h)
    mem = f'Free {info.free/s24:.1f} MB'
else:
    free = int(psutil.virtual_memory().total - psutil.virtual_memory().available)
    mem = f'Free {free/s24:.1f} MB'
print('\n\nRunning:',__file__,' on', device, '=', platform.node())

A = scipy.linalg.hilbert(128)
V, E = LA.eig( A )
a=V[-1].real
b=V[-1].imag
print('\tFirst and Last eigenvalue of Hilert Matrix = %.3f and (%.4e + %.4ei)\n'
       % (V[0].real,  a,b) )


end_time = time.time()

print('Memory usage:',mem, ' and  Time(sec) used = %.2f' % (end_time-start_time) )