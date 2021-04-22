import os
import tensorflow as tf
import utils as utils
from PIL import Image
import numpy as np

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

imagearray = utils.read_img_from_mat('/mnt/projects/CSE_BME_AXM788/data/CCF/TCGA_LUAD/tiles/TCGA-05-4250-01Z-00-DX1.90f67fdf-dff9-46ca-af71-0978d7c221ba.mat')
#swap the 2nd axis to the last, not sure why hdf5 save the 3d matrix with axis change
imagearray = np.swapaxes(imagearray,1,3)
maskarray = np.zeros((imagearray.shape[0], imagearray.shape[1], imagearray.shape[2]),dtype=np.float32)
for i in range(imagearray.shape[0]):
    print('predicting on image number', i+1, '/', imagearray.shape[0])
    #current memory issue if the DL input with more than 1 image, just read 1 by 1 for now
    imagearray = np.expand_dims(imagearray[0,...], axis=0)
    samples = sess.run(tf.get_default_graph().get_tensor_by_name('g_/Sigmoid:0'), \
                feed_dict={tf.get_default_graph().get_tensor_by_name('image:0'): imagearray})
    maskarray[i] = Image.fromarray(np.squeeze(samples*255).astype(np.uint8))
print(maskarray.shape)
utils.add_prediction_backTo_mat('/mnt/projects/CSE_BME_AXM788/data/CCF/TCGA_LUAD/tiles/TCGA-05-4250-01Z-00-DX1.90f67fdf-dff9-46ca-af71-0978d7c221ba.mat',\
                                maskarray)


#loading all images under current foder as numpy array
#imagearray = utils.load_images_under_dir('data')
#casting the prediction results into Image format and save to disk
#im = Image.fromarray(np.squeeze(samples*255).astype(np.uint8))
#im.save("results/test_nuclei.png")    
#Image.fromarray((imagearray[0,...]).astype(np.uint8)).save("results/nuclei.png")

