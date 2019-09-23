import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))


        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        # pretrained model mean and variance
#         mean = [0.485, 0.456, 0.406]
#         std = [0.229, 0.224, 0.225]
#         transform_list = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        loadsize = self.opt.loadSize
        if self.opt.randomloadSize == True:
            loadsize = random.sample([self.opt.fineSize, 2 * self.opt.loadSize], 1)[0]
            # loadsize = random.randint(self.opt.fineSize, 2*self.opt.loadSize)
       # if AB.size[0]/2 < AB.size[1]:
       #     new_w = loadsize*2
       #     new_h = int(float(AB.size[1])/float(AB.size[0]) * new_w)
       # else:
       #     new_h = loadsize*2
       #     new_w = int(float(AB.size[0]) /float(AB.size[1]) * new_h)
        if AB.size[0]/2 < AB.size[1]:
            new_h = loadsize * 32
            new_w = int(float(AB.size[0])/float(AB.size[1]) * new_h)
            if new_w <= 32:
                new_w = 32
            while new_w % 32 != 0:
                new_w += 1
        else:
            new_w = loadsize * 32
            new_h = int(float(AB.size[1]) /float(AB.size[0]) * new_w)
            if new_h <= 15:
                new_h = 16
            while new_h % 16 != 0:
                new_h += 1
       #print("new_h:",new_h,"             new_w:",new_w, "           w:", AB.size[0],"       h:", AB.size[1])
        AB = AB.resize((new_w, new_h), Image.BICUBIC)
       # if self.opt.randomloadSize == True:
       #     loadsize = random.randint(self.opt.fineSize, 10*self.opt.loadSize)
       # AB = AB.resize((loadsize * 2, loadsize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
        A = AB[:, :, 0:w]
        B = AB[:, :, w:w_total]
        assert(w%16==0)
        assert(h%16==0)
       # A = AB[:, h_offset:h_offset + self.opt.fineSize,
       #        w_offset:w_offset + self.opt.fineSize]
       # B = AB[:, h_offset:h_offset + self.opt.fineSize,
       #        w + w_offset:w + w_offset + self.opt.fineSize]
        if self.opt.semifineSize > 0:
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
            A_2 = AB[:, h_offset:h_offset + self.opt.fineSize/2,
               w_offset:w_offset + self.opt.fineSize/2]
            B_2 = AB[:, h_offset:h_offset + self.opt.fineSize/2,
                w + w_offset:w + w_offset + self.opt.fineSize/2]
        else:
            A_2 = A
            B_2 = B

        if self.opt.quartfineSize > 0:
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
            A_4 = AB[:, h_offset:h_offset + self.opt.fineSize/4,
               w_offset:w_offset + self.opt.fineSize/4]
            B_4 = AB[:, h_offset:h_offset + self.opt.fineSize/4,
                w + w_offset:w + w_offset + self.opt.fineSize/4]
        else:
            A_4 = A
            B_4 = B

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
    
        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path,
                'A_2': A_2, 'B_2':B_2,
                'A_4': A_4, 'B_4': B_4}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
