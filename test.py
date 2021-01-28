import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from models import *
from utils import *


def test():
    transform = transforms.Compose(
        [transforms.Resize((128, 128)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


    a_test_data = dsets.ImageFolder(os.path.join('./input', 'testA', 'A1'), transform=transform)
    b_test_data = dsets.ImageFolder(os.path.join('./input', 'testB'), transform=transform)
    a2_test_data = dsets.ImageFolder(os.path.join('./input', 'testA', 'A2'), transform=transform)
    a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=1, shuffle=True, num_workers=0)
    a2_test_loader = torch.utils.data.DataLoader(a2_test_data, batch_size=1, shuffle=True, num_workers=0)
    b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=1, shuffle=True, num_workers=0)

    Generator_AtoB = define_Gen(6, 3, 64)
    Generator_BtoA = define_Gen(6, 3, 64)


    print_networks([Generator_AtoB, Generator_BtoA], ['Gab', 'Gba'])

    try:
        ckpt = load_checkpoint('%s/latest.ckpt' % ('./'))
        Generator_AtoB.load_state_dict(ckpt['Gab'])
        Generator_BtoA.load_state_dict(ckpt['Gba'])
    except:
        print(' [*] No checkpoint!')
    for i in range(0, 2):
        """ run """
        a1_real_test = iter(a_test_loader).next()[0]
        a2_real_test = iter(a2_test_loader).next()[0]

        a_real_test = Variable(torch.cat((a1_real_test, a2_real_test), 1), requires_grad=True)
        # a_real_test = Variable(iter(a_test_loader).next()[0], requires_grad=True)
        b_real_test = Variable(iter(b_test_loader).next()[0], requires_grad=True)

        a_real_test, b_real_test = cuda([a_real_test, b_real_test])
        #a_real_test = cuda([a_real_test])
        Generator_AtoB.eval()
        Generator_BtoA.eval()

        with torch.no_grad():
            a_fake_test = Generator_AtoB(b_real_test)
            b_fake_test = Generator_BtoA(a_real_test)
            a_recon_test = Generator_AtoB(b_fake_test)
            b_recon_test = Generator_BtoA(a_fake_test)

        pic = (torch.cat([a_real_test, b_fake_test, a_recon_test, b_real_test, a_fake_test, b_recon_test],
                         dim=0).data + 1) / 2.0

        if not os.path.isdir('./output'):
            os.makedirs('./output')

        torchvision.utils.save_image(pic, './output' + '/sample' + str(i) +'.jpg', nrow=3)

if __name__ == '__main__' and '__file__' in globals():
    test()