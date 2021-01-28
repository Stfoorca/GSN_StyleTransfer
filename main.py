from models import *
from utils import *
import torch
from torch import nn
import itertools
import functools
import os

from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

class CycleGAN(object):
    def __init__(self):
        self.start_epoch = 0
        self.epochs = 100

        self.Generator_AtoB = define_Gen(3, 3, 64)
        self.Generator_BtoA = define_Gen(3, 3, 64)

        self.Discriminator_A = define_Dis(3, 64, 3)
        self.Discriminator_B = define_Dis(3, 64, 3)

        # Learning Rate 0.0002 - uzyty w artykule
        self.G_Optimizer = torch.optim.Adam(
            itertools.chain(self.Generator_AtoB.parameters(), self.Generator_BtoA.parameters()), lr=0.0002,
            betas=(0.5, 0.999))
        self.D_Optimizer = torch.optim.Adam(
            itertools.chain(self.Discriminator_A.parameters(), self.Discriminator_B.parameters()), lr=0.0002,
            betas=(0.5, 0.999))

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_Optimizer, lr_lambda=LambdaLR(200, 0, 100).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_Optimizer, lr_lambda=LambdaLR(200, 0, 100).step)


        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        try:
            ckpt = load_checkpoint('./latest.ckpt')
            self.start_epoch = ckpt['epoch']
            self.Discriminator_A.load_state_dict(ckpt['Da'])
            self.Discriminator_B.load_state_dict(ckpt['Db'])
            self.Generator_AtoB.load_state_dict(ckpt['Gab'])
            self.Generator_BtoA.load_state_dict(ckpt['Gba'])
            self.D_Optimizer.load_state_dict(ckpt['d_optimizer'])
            self.G_Optimizer.load_state_dict(ckpt['g_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0

    def train(self):
        transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        dataset_path = get_train_directories('./input')

        a_loader = DataLoader(datasets.ImageFolder(os.path.join('./input', 'trainA', 'A1'), transform=transform),
                              batch_size=1, shuffle=True, num_workers=0)

        a2_loader = DataLoader(datasets.ImageFolder(os.path.join('./input', 'trainA', 'A2'), transform=transform),
                              batch_size=1, shuffle=True, num_workers=0)

        b_loader = DataLoader(datasets.ImageFolder(os.path.join('./input', 'trainB'), transform=transform),
                              batch_size=1, shuffle=True, num_workers=0)

        a_fake_sample = PoolSample()
        b_fake_sample = PoolSample()

        for epoch in range(self.start_epoch, 200):
            lr = self.G_Optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            for i, (a1_real, a2_real, b_real) in enumerate(zip(a_loader, a2_loader, b_loader)):
                step = epoch * min(len(a_loader), len(b_loader)) * i + 1


                #GENERATOREK
                set_grad([self.Discriminator_A, self.Discriminator_B], False)
                self.G_Optimizer.zero_grad()
                
                a_real = Variable(a_real)
                b_real = Variable(b_real[0])

                a_real, b_real = cuda([a_real, b_real])

                a_fake = self.Generator_AtoB(b_real)
                b_fake = self.Generator_BtoA(a_real)

                a_recon = self.Generator_AtoB(b_fake)
                b_recon = self.Generator_BtoA(a_fake)

                a_idt = self.Generator_AtoB(a_real)
                b_idt = self.Generator_BtoA(b_real)


                a_idt_loss = self.L1(a_idt, a_real) * 10 * 0.5
                b_idt_loss = self.L1(b_idt, b_real) * 10 * 0.5


                a_fake_dis = self.Discriminator_A(a_fake)
                b_fake_dis = self.Discriminator_B(b_fake)

                real_label = cuda(Variable(torch.ones(a_fake_dis.size())))

                a_gen_loss = self.MSE(a_fake_dis, real_label)
                b_gen_loss = self.MSE(b_fake_dis, real_label)

                a_cycle_loss = self.L1(a_recon, a_real) * 10
                b_cycle_loss = self.L1(b_recon, b_real) * 10

                gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                gen_loss.backward()
                self.G_Optimizer.step()

                #DYSKRYMINATOREK
                set_grad([self.Discriminator_A, self.Discriminator_B], True)

                self.D_Optimizer.zero_grad()

                a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                a_fake, b_fake = cuda([a_fake, b_fake])


                a_real_dis = self.Discriminator_A(a_real)
                a_fake_dis = self.Discriminator_A(a_fake)
                b_real_dis = self.Discriminator_B(b_real)
                b_fake_dis = self.Discriminator_B(b_fake)
                real_label = cuda(Variable(torch.ones(a_real_dis.size())))
                fake_label = cuda(Variable(torch.zeros(a_fake_dis.size())))

                a_dis_real_loss = self.MSE(a_real_dis, real_label)
                a_dis_fake_loss = self.MSE(a_fake_dis, fake_label)
                b_dis_real_loss = self.MSE(b_real_dis, real_label)
                b_dis_fake_loss = self.MSE(b_fake_dis, fake_label)

                a_dis_loss = (a_dis_real_loss + a_dis_fake_loss)*0.5
                b_dis_loss = (b_dis_real_loss + b_dis_fake_loss)*0.5

                a_dis_loss.backward()
                b_dis_loss.backward()
                self.D_Optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %
                                            (epoch, i + 1, min(len(a_loader), len(b_loader)),
                                                            gen_loss,a_dis_loss+b_dis_loss))
            print('Saving checkpoint on epoch: (%3d).' % (epoch))
            save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Discriminator_A.state_dict(),
                                   'Db': self.Discriminator_B.state_dict(),
                                   'Gab': self.Generator_AtoB.state_dict(),
                                   'Gba': self.Generator_BtoA.state_dict(),
                                   'd_optimizer': self.D_Optimizer.state_dict(),
                                   'g_optimizer': self.G_Optimizer.state_dict()},
                                  '%s/latest.ckpt' % ('./'))

            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()


if __name__ == '__main__' and '__file__' in globals():
    print(torch.cuda.is_available())
    xd = CycleGAN()

    xd.train()