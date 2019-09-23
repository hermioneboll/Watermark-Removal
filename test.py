import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
# from util.visualizer import Visualizer
# from util import html
import cv2
import numpy as np

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    # visualizer = Visualizer(opt)
    # create website
    # web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        img_path = model.get_image_paths()
        realA = model.get_current_visuals()['real_A']
        # realB = model.get_current_visuals()['real_B']
        fakeB = model.get_current_visuals()['fake_B']
        savepath = r'./test-result/' + str(opt.which_epoch) + '/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        cv2.imwrite(savepath + img_path[0].split('/')[-1].split('.')[0] + '_' + 'realA.png', realA[:,:,::-1])
        # cv2.imwrite(r'./test-result/' + img_path[0].split('/')[-1].split('.')[0] + '_' + 'realB.png', realB)
        cv2.imwrite(savepath + img_path[0].split('/')[-1].split('.')[0] + '_' + 'fakeB.png', fakeB[:,:,::-1])

        # visuals = model.get_current_visuals()

        print('process image... %s' % img_path)
        # visualizer.save_images(webpage, visuals, img_path)

    # webpage.save()
