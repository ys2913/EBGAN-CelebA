from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--m', type=int, default=20, help='m value for loss calculation')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./checkpoints', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

opt = {'batchSize': opt.batchSize,	#64 
       'beta1': opt.beta1, #0.5, 
       'cuda': opt.cuda,	#False 
       'dataroot': opt.dataroot + '/', #'./celebA/', 
       'dataset': opt.dataset, #'celebA', 
       'imageSize': opt.imageSize, #64, 
       'lr': opt.lr, #0.0002, 
       'manualSeed': opt.manualSeed,	#None,
       'outf': opt.outf, #'./checkpoints',
       'ndf': opt.ndf, #64, 
       'netD': opt.netD, #'', 
       'netG': opt.netG, #'', 
       'ngf': opt.ngf, #64, 
       'ngpu': opt.ngpu, #1, 
       'niter': opt.niter, #25, 
       'nz': opt.nz, #100, 
       'm': opt.m,
       'workers': opt.workers}

dataset = dset.ImageFolder(root=opt['dataroot'], transform=transforms.Compose([
                                   transforms.Scale(64),
                                   transforms.CenterCrop(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                         shuffle=True, num_workers=int(2))

ngpu = int(opt['ngpu'])
nz = int(opt['nz'])
ngf = int(opt['ngf'])
ndf = int(opt['ndf'])
nc = 3
m = int(opt['m'])
fixed_noise = Variable(torch.FloatTensor(50, nz, 1, 1).normal_(0, 1))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    return

class gen(nn.Module):
    def __init__(self):
        super(gen, self).__init__()
        self._gen = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        output = self._gen(input)
        return output

class enc(nn.Module):
    def __init__(self):
        super(enc, self).__init__()
        self._enc = nn.Sequential(
            # 3 x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
            # (ndf*4) x 8 x 8
        )
    
    def forward(self,input):
        output = self._enc(input)
        return output.view(-1,1)

    
class dec(nn.Module):
    def __init__(self):
        super(dec, self).__init__()
        self._dec = nn.Sequential(
            # (ndf*4) x 8 x 8
            nn.ConvTranspose2d( ndf * 4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            # (ndf*2) x 16 x 16
            nn.ConvTranspose2d( ndf * 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            # (ndf) x 32 x 32
            nn.ConvTranspose2d( ndf, 3, 4, 2, 1, bias=False),
            # (3) x 64 x 64
        )
    
    def forward(self,input):
        output = self._dec(input)
        return output
    

class autoenc(nn.Module):
    def __init__(self):
        super(autoenc, self).__init__()
        self._enc = enc()
        self._dec = dec()
    
    def forward(self,input):
        output = self._enc(input)
        output = output.view(-1,ndf*4, 8, 8)
        output = self._dec(output)
        return output

netG = gen()
netD = autoenc()

if opt['cuda']:
    netD.cuda()
    netG.cuda()
    fixed_noise = fixed_noise.cuda()

netG.apply(weights_init)
if opt['netG'] != '':
    netG.load_state_dict(torch.load(opt['netG']))
    
netD.apply(weights_init)
if opt['netD'] != '':
    netD.load_state_dict(torch.load(opt['netD']))

criterion = nn.BCELoss()

if opt['cuda']:
    criterion.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=opt['lr'], betas=(opt['beta1'], 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt['lr'], betas=(opt['beta1'], 0.999))

def reset_grad():
    netG.zero_grad()
    netD.zero_grad()
    return

def D(input):
    x_rec = netD(input)
    output = torch.mean(torch.sum((input - x_rec)**2, 1))
    return output

for epoch in range(opt['niter']):
    for i, data in enumerate(dataloader, 0):
        X, _ = data
        X = Variable(X)
        
        batch = X.size()[0]
        noise = Variable(torch.randn(batch, nz, 1, 1))
        noise.data.normal_(0,1)
        
        if opt['cuda']:
            X = X.cuda()
            noise = noise.cuda()
        
        # Dicriminator
        G_sample = netG(noise)
        D_real = D(X)
        D_fake = D(G_sample)
        
        # EBGAN D loss. D_real and D_fake is energy, i.e. a number
        D_loss = D_real + f.relu(m - D_fake)
        
        # Reuse D_fake for generator loss
        D_loss.backward()
        optimizerD.step()
        reset_grad()
        
        # Generator
        G_sample = netG(noise)
        D_fake = D(G_sample)
        
        G_loss = D_fake
        
        G_loss.backward()
        optimizerG.step()
        reset_grad()
        
        if i%10 == 0:
            print('D_loss: ', D_loss.data[0], ' , G_loss: ', G_loss.data[0])
            break
    
    if epoch%5 == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt['outf'], epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt['outf'], epoch))
        fake_images = netG(fixed_noise)
        vutils.save_image(fake_images.data,'%s/fake_samples_epoch_%03d.png' % (opt['outf'], epoch),
                          nrow=5, padding=2)