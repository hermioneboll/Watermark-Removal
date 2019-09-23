import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # cgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            # gan
#             self.netD = networks.define_D(opt.output_nc, opt.ndf,
#                                           opt.which_model_netD,
#                                           opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
        if self.isTrain:
#             self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            if opt.lambdaVggContent > 0.0:
                self.vgg = networks.Vgg16()
                self.vgg.load_state_dict(torch.load(r'../vgg16.weight')) # use your own path of vgg16.weight
                self.vgg.cuda()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))
        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input, fineSize_r = 0):
        AtoB = self.opt.which_direction == 'AtoB'
        if fineSize_r == 0:
            input_A = input['A' if AtoB else 'B']
            input_B = input['B' if AtoB else 'A']
        elif fineSize_r == 1:
            input_A = input['A_2' if AtoB else 'B_2']
            input_B = input['B_2' if AtoB else 'A_2']
        elif fineSize_r == 2:
            input_A = input['A_4' if AtoB else 'B_4']
            input_B = input['B_4' if AtoB else 'A_4']
        # input_Anew = input_A.resize_(1,3,128,128)
        # input_Bnew = input_B.resize_(1,3,128,128)
        # sizeA = input_Anew.size()
        # sizeB = input_Bnew.size()
        # self.input_A.resize_(sizeA).copy_(input_Anew)
        # self.input_B.resize_(sizeB).copy_(input_Bnew)
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)
        assert self.real_B.size() == self.fake_B.size(),self.image_paths
       # print  "fake_B:",self.fake_B.size(), "real_B:", self.real_B.size(), "real_A:", self.real_A.size()
    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths
    # cgan
    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        try:
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
        except:
            print("loss_D_fake error : ", self.real_A.shape)
            print(self.fake_B.shape)
            print(fake_AB.shape)
            print(pred_fake.shape)
            print(self.criterionGAN.fake_label_var.shape)
            print(self.criterionGAN.Tensor.shape)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()
    # gan    
#     def backward_D(self):
# #         print(self.fake_B.shape)
#         pred_fake = self.netD(self.fake_B)
# #         print(pred_fake.shape)
#         try:
#             self.loss_D_fake = self.criterionGAN(pred_fake, False)
#         except:
#             print("loss_D_fake error : ", self.real_A.shape)
#             print(self.fake_B.shape)
#             print(fake_AB.shape)
#             print(pred_fake.shape)
#             print(self.criterionGAN.fake_label_var.shape)
#             print(self.criterionGAN.Tensor.shape)
#         pred_real = self.netD(self.real_B)
#         self.loss_D_real = self.criterionGAN(pred_real, True)
        
#         # Combined loss
#         self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        
#         self.loss_D.backward(retain_graph=True)
    # cgan
    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
#         print("compute loss G GAN")
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambdaGAN
        mse_loss = torch.nn.MSELoss()
        if self.opt.lambdaVggContent > 0.0:
            real_B_vgg_input = (self.real_B + 1) / 2.0 * 255.0
            fake_B_vgg_input = (self.fake_B + 1) / 2.0 * 255.0
            
            real_B_vgg_input = real_B_vgg_input.cuda()
            fake_B_vgg_input = fake_B_vgg_input.cuda()
            
            real_B_vgg_input = networks.subtract_imagenet_mean_batch(real_B_vgg_input)
            fake_B_vgg_input = networks.subtract_imagenet_mean_batch(fake_B_vgg_input)

            real_B_vgg_features = self.vgg(real_B_vgg_input)
            fake_B_vgg_features = self.vgg(fake_B_vgg_input)

            real_B_vgg_content = Variable(real_B_vgg_features[1].data, requires_grad=False)
            try:
                self.vgg_content_loss = self.opt.lambdaVggContent * mse_loss(fake_B_vgg_features[1], real_B_vgg_content)
            except:
                print("realsize:",self.real_B.size())
        else:
            self.vgg_content_loss = 0
        if self.opt.lambdaL1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        # Second, G(A) = B
#             self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        else:
            self.loss_G_L1 = 0
        # self.loss_G = self.vgg_content_loss + self.loss_G_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.vgg_content_loss
#         self.loss_G = self.loss_G_L1
        self.loss_G.backward()
    # gan
#     def backward_G(self):
#         # First, G(A) should fake the discriminator
#         pred_fake = self.netD(self.fake_B)
#         self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambdaGAN
#         mse_loss = torch.nn.MSELoss()
#         if self.opt.lambdaVggContent > 0.0:
#             real_B_vgg_input = (self.real_B + 1) / 2.0 * 255.0
#             fake_B_vgg_input = (self.fake_B + 1) / 2.0 * 255.0
            
#             real_B_vgg_input = real_B_vgg_input.cuda()
#             fake_B_vgg_input = fake_B_vgg_input.cuda()
            
#             real_B_vgg_input = networks.subtract_imagenet_mean_batch(real_B_vgg_input)
#             fake_B_vgg_input = networks.subtract_imagenet_mean_batch(fake_B_vgg_input)

#             real_B_vgg_features = self.vgg(real_B_vgg_input)
#             fake_B_vgg_features = self.vgg(fake_B_vgg_input)

#             real_B_vgg_content = Variable(real_B_vgg_features[1].data, requires_grad=False)
#             try:
#                 self.vgg_content_loss = self.opt.lambdaVggContent * mse_loss(fake_B_vgg_features[1], real_B_vgg_content)
#             except:
#                 print("realsize:",self.real_B.size())
#         else:
#             self.vgg_content_loss = 0
#         if self.opt.lambdaL1 > 0.0:
#             self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
#         # Second, G(A) = B
# #             self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
#         else:
#             self.loss_G_L1 = 0
#         # self.loss_G = self.vgg_content_loss + self.loss_G_L1
#         self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.vgg_content_loss
# #         self.loss_G = self.loss_G_L1
#         self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        if self.opt.lambdaGAN > 0.0 and self.opt.lambdaVggContent > 0.0 and self.opt.lambdaL1 > 0.0:
            return OrderedDict([
                ('G_GAN', self.loss_G_GAN.data.item()),
                ('G_content', self.vgg_content_loss.data.item()),
                ('G_L1', self.loss_G_L1.data.item()),
                ('D_real', self.loss_D_real.data.item()),
                ('D_fake', self.loss_D_fake.data.item())
            ])
        elif self.opt.lambdaGAN > 0.0 and self.opt.lambdaL1 > 0.0:
            return OrderedDict([
                ('G_GAN', self.loss_G_GAN.data.item()),
                # ('G_content', self.vgg_content_loss.data[0]),
                ('G_L1', self.loss_G_L1.data.item()),
                ('D_real', self.loss_D_real.data.item()),
                ('D_fake', self.loss_D_fake.data.item())
            ])
        elif self.opt.lambdaVggContent > 0.0 and self.opt.lambdaL1 > 0.0:
            return OrderedDict([
                            # ('G_GAN', self.loss_G_GAN.data[0]),
                            ('G_content', self.vgg_content_loss.data.item()),
                            ('G_L1', self.loss_G_L1.data.item())
                            # ('D_real', self.loss_D_real.data[0]),
                            # ('D_fake', self.loss_D_fake.data[0])
                            ])
        elif self.opt.lambdaL1 > 0.0:
            return OrderedDict([
                # ('G_GAN', self.loss_G_GAN.data[0]),
                ('G_L1', self.loss_G_L1.data.item())
                # ('D_real', self.loss_D_real.data[0]),
                # ('D_fake', self.loss_D_fake.data[0])
            ])
        else:
            return OrderedDict([
                # ('G_GAN', self.loss_G_GAN.data[0]),
                ('G_content', self.vgg_content_loss.data.item())
                # ('D_real', self.loss_D_real.data[0]),
                # ('D_fake', self.loss_D_fake.data[0])
            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
