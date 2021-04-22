import os
import tensorflow as tf
import utils as utils
from PIL import Image
import numpy as np
import scipy.io as sio
import h5py

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index        
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
sess = tf.Session(config=run_config)
flags = FLAGS
model = utils.net_arch(sess, flags, (2000,2000))
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
model_out_dir = "model"
if utils.load_model(sess, saver, model_out_dir):
    best_auc_sum = sess.run(model.best_auc_sum)
    print('====================================================')
    print(' Best auc_sum: {:.3}'.format(best_auc_sum))
    print('=============================================')    
    print(' [*] Load Success!\n')
#names = utils.all_files_under('/mnt/projects/CSE_BME_AXM788/data/TCGA-LUAD-hist-DX/tileExt/tiles/','.mat')
names = utils.all_files_under('/scratch/users/xxw345/data/','.mat')
for name in names:
    print(name)
    savingName = '/scratch/users/crb138/chemoXX/mask/' + os.path.basename(name)[:-4] + '_mask'
    if os.path.isfile(savingName) or os.path.getsize(name)>>20 < 2:
        print('size less than 2MB or file existed , continue to next')
        continue       
        
    mat_contents = sio.loadmat(name)
    data = mat_contents['data']
	#data = mat_contents['tileStruct/data']
    #name = mat_contents['tileStruct/name']
    
    
    mask_dataset_shape = (data.shape[3], 2000, 2000)
    hdf5_file = h5py.File(savingName, mode='w')
    hdf5_file.create_dataset("mask", mask_dataset_shape, np.int8, compression="gzip")

    #if data.shape == (3,2000,2000):
    #    tileCount = 1
    #else:
    #    tileCount = data.shape[0]
    tileCount = data.shape[3]
    
    for i in range(tileCount):
        print('predicting on image number', i+1, '/', tileCount, end="\r")
        #iif tileCount == 1:
        #    image = mat_contents['tileStruct/data'][:]
        #else:
        #    image = mat_contents[data[i][0]].value        
        image = data[:,:,:,i]
        #image = np.expand_dims(image, axis=0)
        #image = np.swapaxes(image,0,2)
        #image = image[...,::-1]
        image = np.expand_dims(image, axis=0)
        samples = sess.run(tf.get_default_graph().get_tensor_by_name('g_/Sigmoid:0'), \
                feed_dict={tf.get_default_graph().get_tensor_by_name('image:0'): image})
        samples = np.squeeze(samples*255).astype(np.uint8)        
        #print(samples.shape)
        hdf5_file["mask"][i, ...] = samples
    hdf5_file.close()
    #mat_contents.close()
        
        
'''        
    try:
        imagearray = utils.read_img_from_mat(name)
    except:
        print('skiping file', name, 'due to image issue')
        continue
        
    
    imagearray = np.swapaxes(imagearray,1,3)
    if imagearray.shape[0] > 200:
        print('too many tiles, skiping now and will calculate later', imagearray.shape[0])
        continue    
    maskarray = np.zeros((imagearray.shape[0], imagearray.shape[1], imagearray.shape[2]),dtype=np.float32)
    for i in range(imagearray.shape[0]):
        print('predicting on image number', i+1, '/', imagearray.shape[0])
        image = np.expand_dims(imagearray[0,...], axis=0)
        samples = sess.run(tf.get_default_graph().get_tensor_by_name('g_/Sigmoid:0'), \
                feed_dict={tf.get_default_graph().get_tensor_by_name('image:0'): image})
        maskarray[i] = Image.fromarray(np.squeeze(samples*255).astype(np.uint8))    
    #savingName = os.path.basename(name)[:-4] + '_mask.mat'
    #print(savingName)
    hdf5 = h5py.File('/mnt/projects/CSE_BME_AXM788/data/CCF/TCGA_LUSC/maskGAN/'+savingName,'w')
    hdf5.create_dataset('mask', data = maskarray)
    hdf5.close()
    print('done')
'''   
'''    
    try:
        sio.savemat('/mnt/projects/CSE_BME_AXM788/data/CCF/TCGA_LUSC/maskGAN/'+savingName,
                {"mask":maskarray},format='5',do_compression=True)
    except:
        continue                
'''


#imagearray = utils.read_img_from_mat('/mnt/ccipd_data/CCF/TCGA_LUAD/tiles/TCGA-05-4250-01Z-00-DX1.90f67fdf-dff9-46ca-af71-0978d7c221ba.mat')
#swap the 2nd axis to the last, not sure why hdf5 save the 3d matrix with axis change
#imagearray = np.swapaxes(imagearray,1,3)
#maskarray = np.zeros((imagearray.shape[0], imagearray.shape[1], imagearray.shape[2]),dtype=np.float32)
#for i in range(imagearray.shape[0]):
#    print('predicting on image number', i+1, '/', imagearray.shape[0])
    #current memory issue if the DL input with more than 1 image, just read 1 by 1 for now
#    imagearray = np.expand_dims(imagearray[0,...], axis=0)
#    samples = sess.run(tf.get_default_graph().get_tensor_by_name('g_/Sigmoid:0'), \
#                feed_dict={tf.get_default_graph().get_tensor_by_name('image:0'): imagearray})
#    maskarray[i] = Image.fromarray(np.squeeze(samples*255).astype(np.uint8))
#print(maskarray.shape)
#utils.add_prediction_backTo_mat('/mnt/ccipd_data/CCF/TCGA_LUAD/tiles/TCGA-05-4250-01Z-00-DX1.90f67fdf-dff9-46ca-af71-0978d7c221ba.mat',\
#                                maskarray)


#loading all images under current foder as numpy array
#imagearray = utils.load_images_under_dir('data')
#casting the prediction results into Image format and save to disk
#im = Image.fromarray(np.squeeze(samples*255).astype(np.uint8))
#im.save("results/test_nuclei.png")    
#Image.fromarray((imagearray[0,...]).astype(np.uint8)).save("results/nuclei.png")

