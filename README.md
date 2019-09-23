# Watermark-Removal(Updating...)
## pix2pix train/test
### Train a model:
```
python train.py --dataroot ../data/Local_Watermark/trainAB --semifineSize 0 --quartfineSize 0 --resize_or_crop resize_and_crop --loadSize 16 --fineSize 16 --nThreads 0 --checkpoints_dir checkpoint_UNET_L1_cGAN_PL_NNresize4 --which_model_netG unet_16_nnresize
```
### Test the model:
```
python test.py --dataroot ../newopenset  --resize_or_crop scale_width --nThreads 0 --dataset_mode single --model test --how_many 12000 --checkpoints_dir checkpoint_UNET_L1_cGAN_PL_NNresize4 --which_model_netG unet_16_nnresize --which_epoch 40
```

## Related Projects 
[pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix): Image-to-image translation with conditional adversarial nets
