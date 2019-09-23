import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')
        loadsize = self.opt.fineSize
        if A.size[0] < A.size[1]:
            new_h = loadsize
            new_w = int(float(A.size[0]) / float(A.size[1]) * new_h)
            if new_w <= 3:
                new_w = 4
            while new_w % 4 != 0:
                new_w += 1
        else:
            new_w = loadsize
            new_h = int(float(A.size[1]) / float(A.size[0]) * new_w)
            if new_h <= 3:
                new_h = 4
            while new_h % 4 != 0:
                new_h += 1
                # print("new_h:",new_h,"             new_w:",new_w, "           w:", AB.size[0],"       h:", AB.size[1])
        A = A.resize((new_w, new_h), Image.BICUBIC)
        A = self.transform(A)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
