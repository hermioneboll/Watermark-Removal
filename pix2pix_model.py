import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from base_model import BaseModel
import networks
from . import face_recognition_networks
import pickle

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
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, 64,
                                            'n_layers',
                                            4, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            if opt.lambdaFace > 0.0:
                self.Face_recognition_network = face_recognition_networks.LightCNN_29Layers(num_classes=79077)
                self.Face_recognition_network = torch.nn.DataParallel(self.Face_recognition_network).cuda()
                checkpoint = torch.load(r'/data5/shentao/LightCNN/CNN_29.pkl')
                self.Face_recognition_network.load_state_dict(checkpoint)
                for param in self.Face_recognition_network.parameters():
                    param.requires_grad = False
                self.Face_recognition_network.eval()
                self.criterionFace = torch.nn.L1Loss()

            if opt.lambdaVggContent > 0.0:
                self.vgg = networks.Vgg16()
                self.vgg.load_state_dict(torch.load(r'/data5/shentao/StyleTransfer/experiments/models/vgg16.weight'))
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

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        _,_,self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        _, _,self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_) * self.opt.lambda_A
        lambda_Face = self.opt.lambdaFace
        if lambda_Face > 0:
            # G_A should be identity if real_B is fed.
            _, real_A_face_feature_fc, real_A_face_feature_conv = self.Face_recognition_network.forward((self.real_A + 1)/2.0)
            _, fake_B_face_feature_fc, fake_B_face_feature_conv  = self.Face_recognition_network.forward((self.fake_B + 1)/2.0)
            _, real_B_face_feature_fc, real_B_face_feature_conv = self.Face_recognition_network.forward((self.real_B + 1) / 2.0)
            loss_perceptual_A = ( self.criterionFace( fake_B_face_feature_fc,Variable(real_A_face_feature_fc.data, requires_grad=False)) +
                                     self.criterionFace(fake_B_face_feature_conv,Variable(real_A_face_feature_conv.data, requires_grad=False)))* lambda_Face

            loss_perceptual_B = ( self.criterionFace( fake_B_face_feature_fc,Variable(real_B_face_feature_fc.data, requires_grad=False)) +
                                     self.criterionFace(fake_B_face_feature_conv,Variable(real_B_face_feature_conv.data, requires_grad=False)))* lambda_Face

            self.loss_perceptual = loss_perceptual_B

        else:
            self.loss_perceptual = 0

        # lambda_identity = self.opt.identity
        # if lambda_identity > 0:
        #     # G_A should be identity if real_B is fed.
        #     self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * lambda_identity
        # else:
        #     self.loss_G_L1 = 0


        mse_loss = torch.nn.MSELoss()
        if self.opt.lambdaVggContent > 0.0:
            real_A_vgg_input = (torch.cat((self.real_A, self.real_A, self.real_A), 1) + 1) / 2.0 * 255.0
            fake_B_vgg_input = (torch.cat((self.fake_B, self.fake_B, self.fake_B), 1) + 1) / 2.0 * 255.0

            real_A_vgg_input = networks.subtract_imagenet_mean_batch(real_A_vgg_input)
            fake_B_vgg_input = networks.subtract_imagenet_mean_batch(fake_B_vgg_input)

            real_A_vgg_features = self.vgg(real_A_vgg_input)
            fake_B_vgg_features = self.vgg(fake_B_vgg_input)

            real_A_vgg_content = Variable(real_A_vgg_features[1].data, requires_grad=False)

            self.vgg_content_loss = self.opt.lambdaVggContent  * mse_loss(fake_B_vgg_features[1], real_A_vgg_content)
        else:
            self.vgg_content_loss = 0

        if self.opt.lambdaVggStyle > 0.0:
            real_B_vgg_input = (torch.cat((self.real_B, self.real_B, self.real_B), 1) + 1) / 2.0 * 255.0
            real_B_vgg_input = networks.subtract_imagenet_mean_batch(real_B_vgg_input)

            real_B_vgg_features = self.vgg(real_B_vgg_input)
            gram_style = [networks.gram_matrix(y) for y in real_B_vgg_features]

            style_loss = 0.
            for m in range(len(fake_B_vgg_features)):
                gram_y = networks.gram_matrix(fake_B_vgg_features[m])
                gram_s = Variable(gram_style[m].data, requires_grad=False).repeat(self.opt.batchSize, 1, 1, 1)
                style_loss += self.opt.lambdaVggStyle * mse_loss(gram_y, gram_s)

            self.vgg_style_loss = style_loss
        else:
            self.vgg_style_loss = 0

        if self.opt.pretrain:
            self.loss_G = self.loss_perceptual + self.vgg_content_loss + self.vgg_style_loss
        else:
            self.loss_G = self.loss_G_GAN + self.loss_perceptual + self.vgg_content_loss + self.vgg_style_loss

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                            ('G_perceptual', self.loss_perceptual.data[0]),
                            ('G_content', self.vgg_content_loss.data[0]),
                            ('G_style', self.vgg_style_loss.data[0]),
                            ('D_real', self.loss_D_real.data[0]),
                            ('D_fake', self.loss_D_fake.data[0])
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
