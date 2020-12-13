from models import *
from utils import *
import torch
import itertools


class CycleGAN(object):
    def __init__(self):
        self.start_epoch = 0
        self.epochs = 100


        self.Generator_AtoB = Generator(3, 3, 64)
        self.Generator_BtoA = Generator(3, 3, 64)

        self.Discriminator_A = Discriminator(3, 64, 3)
        self.Discriminator_B = Discriminator(3, 64, 3)

        #Learning Rate 0.0002 - uzyty w artykule
        self.G_Optimizer = torch.optim.Adam(itertools.chain(self.Generator_AtoB.parameters(), self.Generator_BtoA.parameters()), lr=0.0002, betas=(0.5, 0.999))
        self.D_Optimizer = torch.optim.Adam(itertools.chain(self.Discriminator_A.parameters(), self.Discriminator_B.parameters()), lr=0.0002, betas=(0.5, 0.999))



    def train(self):
        pass
        #TODO:Wczytywanie zdjec
        #Preprocess wczytywanych zdjecia
        #Zastosowanie dataloadera z pytorcha z uzyciem atrybutu transform

        #Stworzenie fake pooli zdjec

        #Trening:
            #Forward pass - generatory
            #Obliczanie lossa
            #Backward pass - generatory

            #Forward pass - dyskriminatory
            #Obliczanie lossa
            #Backward pass - dyskriminatory

            #Checkpoint

            #Learning rate update - zgodnie z artykulem





