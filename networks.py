import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
from .NVIDIA_networks import *
###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnetResize_9blocks':
        netG = ResnetResizeGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnetResize_3blocks':
        netG = ResnetResizeGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_3blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_resize_128':
        netG = UnetNNResizeGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_resize_slim_128':
        netG = UnetNNResizeSlimGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_resize_256_8':
        netG = UnetNNResizeGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_3blocks_muti_resolution':
        netG = ResnetResizeMultiResolutionGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_3blocks_muti_resolution_PixelNorm':
        netG = ResnetResizeMultiResolutionPixelNormGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                                    use_dropout=use_dropout, n_blocks=3, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_3blocks_muti_resolution_PixelNorm_slim':
        netG = ResnetResizeMultiResolutionPixelNormSlimGenerator(input_nc, output_nc, 16, norm_layer=norm_layer,use_dropout=use_dropout, n_blocks=3, gpu_ids=gpu_ids)
    elif which_model_netG == 'PixelNormUnetGenerator':
        netG = PixelNormUnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, gpu_ids=gpu_ids)
    elif which_model_netG == 'AttGAN_G':
        netG = AttGANGenerator(ngf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(device_id=gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=4, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'AttGAN_D':
        netD = AttGANDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(device_id=gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)




class ResnetResizeGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetResizeGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias)]


        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.Upsample(scale_factor=2)]
            model += [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1,padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class ResnetResizeMultiResolutionGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=None, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetResizeMultiResolutionGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias)]
        self.fnum_32 = ngf * mult
        self.model_32 = nn.Sequential(*model)
        self.conv_32 = nn.Conv2d(self.fnum_32, 1, (3, 3), padding=1)
        self.tanh_32 = nn.Tanh()

        mult = 2 ** 2
        model_64 = [nn.Upsample(scale_factor=2),
                     nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                     norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        self.fnum_64 = int(ngf * mult / 2)
        self.model_64 = nn.Sequential(*model_64)
        self.conv_64 = nn.Conv2d(self.fnum_64, 1, (3, 3), padding=1)
        self.tanh_64 = nn.Tanh()


        mult = 2 ** 1
        model_128 = [nn.Upsample(scale_factor=2),
                     nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                     norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model_128 += [nn.ReflectionPad2d(3)]
        model_128 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_128 += [nn.Tanh()]
        self.model_128 = nn.Sequential(*model_128)

    def forward(self, input):
            feature_32 = self.model_32(input)
            output_32 = self.conv_32(feature_32)
            output_32 = self.tanh_32(output_32)

            feature_64 = self.model_64(feature_32)
            output_64 = self.conv_64(feature_64)
            output_64 = self.tanh_64(output_64)

            output_128 = self.model_128(feature_64)

            return output_32, output_64, output_128

class ResnetResizeMultiResolutionPixelNormGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=None, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetResizeMultiResolutionPixelNormGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [PixelNormBlock(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [PixelNormResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=True)]
        self.fnum_32 = ngf * mult
        self.model_32 = nn.Sequential(*model)
        self.conv_32 = nn.Conv2d(self.fnum_32, 1, (3, 3),stride=1, padding=1)
        self.tanh_32 = nn.Tanh()

        mult = 2 ** 2
        model_64 = [PixelNormUpscaleBlock(ngf * mult, int(ngf * mult / 2),  kernel_size=3, padding=1)]

        self.fnum_64 = int(ngf * mult / 2)
        self.model_64 = nn.Sequential(*model_64)
        self.conv_64 = nn.Conv2d(self.fnum_64, 1, (3, 3), stride=1, padding=1)
        self.tanh_64 = nn.Tanh()

        mult = 2 ** 1
        model_128 = [PixelNormUpscaleBlock(ngf * mult, int(ngf * mult / 2), kernel_size=3, padding=1)]

        model_128 += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1)]
        model_128 += [nn.Tanh()]
        self.model_128 = nn.Sequential(*model_128)

    def forward(self, input):
        feature_32 = self.model_32(input)
        output_32 = self.conv_32(feature_32)
        output_32 = self.tanh_32(output_32)

        feature_64 = self.model_64(feature_32)
        output_64 = self.conv_64(feature_64)
        output_64 = self.tanh_64(output_64)

        output_128 = self.model_128(feature_64)

        return output_32, output_64, output_128

class ResnetResizeMultiResolutionPixelNormSlimGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=None, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetResizeMultiResolutionPixelNormSlimGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        # model = [PixelNormBlock(input_nc, ngf, kernel_size=3, stride=1, padding=1)]
        model = [nn.Conv2d(input_nc, ngf, 3, 1, 1, bias=False)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [PixelNormBlock(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [PixelNormResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                           use_dropout=use_dropout, use_bias=True)]
        self.fnum_32 = ngf * mult
        self.model_32 = nn.Sequential(*model)
        self.conv_32 = nn.Conv2d(self.fnum_32, 1, (3, 3), stride=1, padding=1)
        self.tanh_32 = nn.Tanh()

        mult = 2 ** 2
        model_64 = [PixelNormUpscaleBlock(ngf * mult, int(ngf * mult / 2), kernel_size=3, padding=1)]

        self.fnum_64 = int(ngf * mult / 2)
        self.model_64 = nn.Sequential(*model_64)
        self.conv_64 = nn.Conv2d(self.fnum_64, 1, (3, 3), stride=1, padding=1)
        self.tanh_64 = nn.Tanh()

        mult = 2 ** 1
        model_128 = [PixelNormUpscaleBlock(ngf * mult, int(ngf * mult / 2), kernel_size=3, padding=1)]
        model_128 += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1)]
        model_128 += [nn.Tanh()]

        self.model_128 = nn.Sequential(*model_128)

    def forward(self, input):
        feature_32 = self.model_32(input)
        output_32 = self.conv_32(feature_32)
        output_32 = self.tanh_32(output_32)

        feature_64 = self.model_64(feature_32)
        output_64 = self.conv_64(feature_64)
        output_64 = self.tanh_64(output_64)

        output_128 = self.model_128(feature_64)

        return output_32, output_64, output_128

class PixelNormUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=None, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect'):
        assert (n_blocks >= 0)
        super(PixelNormUnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        self.conv1_128 = nn.Conv2d(input_nc, ngf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU()

        self.conv2_64  = PixelNormBlock(ngf,   ngf * 2, kernel_size=3, stride=2, padding=1)
        self.conv3_32  = PixelNormBlock(ngf*2, ngf * 4, kernel_size=3, stride=2, padding=1)
        self.conv4_16  = PixelNormBlock(ngf*4, ngf * 4, kernel_size=3, stride=2, padding=1)

        self.resnet_block1_16 = PixelNormResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer,
                                                     use_dropout=use_dropout, use_bias=True)
        self.resnet_block2_16 = PixelNormResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer,
                                                     use_dropout=use_dropout, use_bias=True)
        self.resnet_block3_16 = PixelNormResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer,
                                                     use_dropout=use_dropout, use_bias=True)

        self.conv5_32 = PixelNormUpscaleBlock(ngf * 4, ngf * 4, kernel_size=3, padding=1)

        self.conv_32 = nn.Conv2d(ngf * 8, output_nc, (3, 3), stride=1, padding=1)
        self.tanh_32 = nn.Tanh()

        self.conv6_64 = PixelNormUpscaleBlock(ngf * 8, ngf * 2,  kernel_size=3, padding=1)

        self.conv_64 = nn.Conv2d(ngf*4, output_nc, (3, 3), stride=1, padding=1)
        self.tanh_64 = nn.Tanh()

        self.conv7_128 = PixelNormUpscaleBlock(ngf * 4, ngf, kernel_size=3, padding=1)
        self.conv8_128 = nn.Conv2d(ngf*2, output_nc, kernel_size=3, padding=1)
        self.tanh_128 = nn.Tanh()

    def forward(self, input):

        fmap_128 = self.conv1_128(input)
        fmap_128 = self.relu(fmap_128)

        fmap_64 = self.conv2_64(fmap_128)
        fmap_32 = self.conv3_32(fmap_64)
        fmap_16 = self.conv4_16(fmap_32)

        res_fmap_16_1 = self.resnet_block1_16(fmap_16)
        res_fmap_16_2 = self.resnet_block2_16(res_fmap_16_1)
        res_fmap_16_3 = self.resnet_block3_16(res_fmap_16_2)

        up_famp_32 = self.conv5_32(res_fmap_16_3)
        concat_fmap_32 = torch.cat([fmap_32, up_famp_32], 1)
        output_32 = self.conv_32(concat_fmap_32 )
        output_32 = self.tanh_32(output_32)

        up_famp_64 = self.conv6_64(concat_fmap_32)
        concat_fmap_64 = torch.cat([fmap_64, up_famp_64], 1)
        output_64 = self.conv_64(concat_fmap_64)
        output_64 = self.tanh_64(output_64)

        up_fmap_128 = self.conv7_128(concat_fmap_64)

        concat_fmap_128 = torch.cat([fmap_128, up_fmap_128], 1)
        output_128 = self.conv8_128(concat_fmap_128)

        output_128 = self.tanh_128(output_128)

        return output_32, output_64, output_128

class AttGANGenerator(nn.Module):
    def __init__(self,ngf = 64, norm_layer= None):
        super(AttGANGenerator, self).__init__()
        kw = 4
        padw = 1
        encoder = [nn.Conv2d(1, ngf, kernel_size=kw, stride=2, padding=padw),norm_layer(ngf),nn.LeakyReLU(0.2, True),#64
                    nn.Conv2d(ngf, ngf*2, kernel_size=kw, stride=2, padding=padw),norm_layer(ngf*2),nn.LeakyReLU(0.2, True),#32
                    nn.Conv2d(ngf*2, ngf*4, kernel_size=kw, stride=2, padding=padw), norm_layer(ngf*4),nn.LeakyReLU(0.2, True),#16
                    nn.Conv2d(ngf*4, ngf*8, kernel_size=kw, stride=2, padding=padw), norm_layer(ngf*8), nn.LeakyReLU(0.2, True),#8
                    nn.Conv2d(ngf*8, ngf*16, kernel_size=kw, stride=2, padding=padw), norm_layer(ngf*16), nn.LeakyReLU(0.2, True)]#4
        self.encoder = nn.Sequential(*encoder)

        decoder = [       nn.ConvTranspose2d(ngf*16 + 40, ngf*16, kernel_size=4, stride=2, padding=1),
                          norm_layer(ngf*16),nn.LeakyReLU(0.2, True),#8
                          nn.ConvTranspose2d(ngf*16, ngf*8, kernel_size=4, stride=2, padding=1),
                          norm_layer(ngf*8),nn.LeakyReLU(0.2, True),#16
                          nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1),
                          norm_layer(ngf*4), nn.LeakyReLU(0.2, True),#32
                          nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1),
                          norm_layer(ngf*2), nn.LeakyReLU(0.2, True),#64
                          nn.ConvTranspose2d(ngf*2, 1, kernel_size=4, stride=2, padding=1),
                          nn.Tanh()]# 128
        self.decoder= nn.Sequential(*decoder)

        label_embedding = [nn.Linear(40, 4*4*40),nn.LeakyReLU(0.2, True)]
        self.label_embedding = nn.Sequential(*label_embedding)

    def forward(self, input, Z_recon, Z_target):
        Z_recon = self.label_embedding(Z_recon)
        Z_target = self.label_embedding(Z_target)

        Z_recon = Z_recon.view(Z_recon.size(0), 40,4,4)
        Z_target = Z_target.view(Z_target.size(0), 40, 4,4)

        encode = self.encoder(input)
        recon = torch.cat([encode, Z_recon], 1)
        target = torch.cat([encode, Z_target], 1)

        target = self.decoder(target)
        recon = self.decoder(recon)
        return recon, target


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

    # Define a resnet block
class PixelNormResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(PixelNormResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [PixelNormBlock(dim, dim, kernel_size=3,stride=1, padding=p)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [PixelNormBlock(dim, dim, kernel_size=3, stride=1, padding=p)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# Defines the glassremove Unet
class UnetNNResizeGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetNNResizeGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetNNResizeSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                                     norm_layer=norm_layer, innermost=True)
        unet_block = UnetNNResizeSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                     norm_layer=norm_layer)
        unet_block = UnetNNResizeSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                     norm_layer=norm_layer)
        unet_block = UnetNNResizeSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                     norm_layer=norm_layer)
        unet_block = UnetNNResizeSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                     outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetNNResizeSlimGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetNNResizeSlimGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetNNResizeSlimSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True)
        unet_block = UnetNNResizeSlimSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetNNResizeSlimSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetNNResizeSlimSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetNNResizeSlimSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetNNResizeSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetNNResizeSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            unsample = nn.Upsample(scale_factor=2)
            upconv  = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)

            down =  [downconv]
            up = [uprelu, unsample ,upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            unsample = nn.Upsample(scale_factor=2)
            upconv  = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)

            down = [downrelu, downconv]
            up = [uprelu, unsample, upconv,upnorm]
            model = down + up
        else:
            unsample = nn.Upsample(scale_factor=2)
            upconv  = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)

            down = [downrelu, downconv, downnorm]
            up = [uprelu, unsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

# Defines the slimsubmodule with skip connection.
class UnetNNResizeSlimSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetNNResizeSlimSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            unsample = nn.Upsample(scale_factor=2)
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)

            down = [downconv]
            up = [uprelu, unsample, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            unsample = nn.Upsample(scale_factor=2)
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)

            down = [downrelu, downconv]
            up = [uprelu, unsample, upconv, upnorm]
            model = down + up
        else:
            unsample = nn.Upsample(scale_factor=2)
            upconv1x1 = nn.Conv2d(inner_nc * 2, inner_nc * 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
            upconv = nn.Conv2d(inner_nc * 1, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)

            down = [downrelu, downconv, downnorm]
            up = [uprelu, unsample, upconv1x1, uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)



# Defines the PatchGAN discriminator with the specified arguments.
class AttGANDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(AttGANDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, 64, kernel_size=kw, stride=2, padding=padw),norm_layer(64),nn.LeakyReLU(0.2, True),#64
                    nn.Conv2d(64, 128, kernel_size=kw, stride=2, padding=padw),norm_layer(128),nn.LeakyReLU(0.2, True),#32
                    nn.Conv2d(128, 256, kernel_size=kw, stride=2, padding=padw), norm_layer(256),nn.LeakyReLU(0.2, True),#16
                    nn.Conv2d(256, 512, kernel_size=kw, stride=2, padding=padw), norm_layer(512), nn.LeakyReLU(0.2, True),#8
                    nn.Conv2d(512, 1024, kernel_size=kw, stride=2, padding=padw), norm_layer(1024), nn.LeakyReLU(0.2, True)]#4
        self.base_model = nn.Sequential(*sequence)

        classify_fc = [nn.Linear(1024 * 4 * 4, 1024),norm_layer(1024),nn.LeakyReLU(0.2, True),
                       nn.Linear(1024 , 40)]
        self.classify_fc = nn.Sequential(*classify_fc)

        adverse_fc =  [nn.Linear(1024 * 4 * 4, 1024)]
        self.adverse_fc = nn.Sequential(*adverse_fc)

    def forward(self, input):
        x = self.base_model(input)
        x = x.view(x.size(0), -1)
        classify_output = self.classify_fc(x)
        # classify_output
        adverse_output = self.adverse_fc(x)

        return classify_output, adverse_output

#=======================================================================================================================
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class GramMatrix(nn.Module):
    # for style transfer
    def forward(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


def subtract_imagenet_mean_batch(batch):
    # """Subtract ImageNet mean pixel-wise from a BGR image."""
    # tensortype = type(batch.data)
    # mean = tensortype(batch.data.size())
    # mean[:, 0, :, :] = 103.939
    # mean[:, 1, :, :] = 116.779
    # mean[:, 2, :, :] = 123.680
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 2, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 0, :, :] = 123.680
    return batch - Variable(mean)


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        relu4_3 = h

        return [relu1_2, relu2_2, relu3_3, relu4_3]
