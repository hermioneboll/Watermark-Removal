import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import cv2
# from util.visualizer import Visualizer

if __name__ == '__main__':

    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    # visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        fineSize_r = -1
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            # visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            if i % 30 == 0 :
                fineSize_r += 1
                fineSize_r = fineSize_r%3
            model.set_input(data)
            model.optimize_parameters()
#             print(data['A'].shape)
#             print(model.loss_G_GAN.data.item())
 #           print("optimize====================", i)
            if total_steps % opt.display_freq == 0:
 #               print("display==display========")
                save_result = total_steps % opt.update_html_freq == 0
                # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                # print(model.get_current_visuals(), epoch)
                realA = model.get_current_visuals()['real_A']
                realB = model.get_current_visuals()['real_B']
                fakeB = model.get_current_visuals()['fake_B']
                r, g, b = cv2.split(realA)
                realA = cv2.merge([b, g, r])
                r, g, b = cv2.split(realB)
                realB = cv2.merge([b, g, r])
                r, g, b = cv2.split(fakeB)
                fakeB = cv2.merge([b, g, r])
                cv2.imwrite(r'./train-result/epoch' + str(epoch) + '_' + 'realA.jpg', realA)
                cv2.imwrite(r'./train-result/epoch' + str(epoch) + '_' + 'realB.jpg', realB)
                cv2.imwrite(r'./train-result/epoch' + str(epoch) + '_' + 'fakeB.jpg', fakeB)

            if total_steps % opt.print_freq == 0:
 #               print("error==print=============================error")
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
        
                print('epoch:%d\titers:%.3f\tG_GAN:%.5f\tD_Real:%.5f\tD_Fake:%.5f\tG_L1:%.5f\tG_PL:%.5f'%(epoch, float(epoch_iter)/dataset_size,errors['G_GAN'], errors['D_real'], errors['D_fake'], errors['G_L1'], errors['G_content']))
                # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                # if opt.display_id > 0:
                #     visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
#                print('epoch:', epoch, "    iters: ", float(epoch_iter)/dataset_size,
#                      "   G_L1: ", errors['G_L1'])
                # , "   D_real: ", errors['D_real'], "   D_fake: ", errors['D_fake'])
    
    
#                 if opt.lambdaVggContent > 0.0 and opt.lambdaL1 > 0.0:
# 			print('epoch:%d\titers:%.3f\tG_PL:%.5f\tG_L1:%.5f'%(epoch, float(epoch_iter)/dataset_size,errors['G_content'], errors['G_L1']))
# 		elif  opt.lambdaVggContent > 0.0:
# 			print('epoch:%d\titers:%.3f\tG_PL:%.5f'%(epoch, float(epoch_iter)/dataset_size,errors['G_content']))
# 		elif opt.lambdaL1 > 0.0:
# 			print('epoch:%d\titers:%.3f\tG_L1:%.5f'%(epoch, float(epoch_iter)/dataset_size,errors['G_L1']))
# 		else:
# 			print('no loss function')
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
