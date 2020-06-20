def predict():
  import sys
  import argparse
  opt = argparse.Namespace(checkpoint='gmm_final.pth',
                          data_root='Database',
                          out_dir='output/first',
                          name='GMM',
                          batch_size=16,
                          n_worker=4,
                          gpu_id='0',
                          log_freq=100,
                          radius=5,
                          fine_width=192,
                          fine_height=256,
                          grid_size=5)
  from run_gmm import run,GMM,GMMDataset,load_checkpoint,DataLoader,torch
  model = GMM(opt)
  load_checkpoint(model, opt.checkpoint)
  # model.cuda()
  model.eval()
  print('Run on {} data'.format("VAL"))
  dataset = GMMDataset(opt, "val", data_list='val_pairs.txt', train=False)
  dataloader = DataLoader(dataset, batch_size=opt.batch_size,
                          num_workers=opt.n_worker, shuffle=False)
  with torch.no_grad():
      run(opt, model, dataloader, "val")
  print('Successfully completed')
  opt = argparse.Namespace(checkpoint='tom_final.pth',
                          data_root='Database',
                          out_dir='output/second',
                          name='TOM',
                          batch_size=16,
                          n_worker=4,
                          gpu_id='0',
                          log_freq=100,
                          radius=5,
                          fine_width=192,
                          fine_height=256,
                          grid_size=5)
  from run_tom import run,UnetGenerator,nn,load_checkpoint,TOMDataset,DataLoader
  model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
  load_checkpoint(model, opt.checkpoint)
  # model.cuda()
  model.eval()
  mode = 'val'
  print('Run on {} data'.format(mode.upper()))
  dataset = TOMDataset(opt, mode, data_list=mode+'_pairs.txt', train=False)
  dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.n_worker, shuffle=False)  
  with torch.no_grad():
    run(opt, model, dataloader, mode)
  print('Successfully completed')
#Resive image code
# from PIL import Image
# im = Image.open("output/second")
# width, height = im.size  
  
# # Setting the points for cropped image  
# left = width / 3
# top = 2 * height / 3
# right = 2 * width / 3
# bottom = height
  
# # Cropped image of above dimension  
# # (It will not change orginal image)  
# im1 = im.crop((left, top, right, bottom)) 
# newsize = (200, 300) 
# im1 = im1.resize(newsize) 
# # Shows the image in image viewer  
# im1.save('output/final')
