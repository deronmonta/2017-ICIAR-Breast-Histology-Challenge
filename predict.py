import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse


# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=r'C:\Users\FurryMonster Yang\Documents\Python Scripts\testing_data'
filename = dir_path +'/' +image_path
image_size= 256
num_channels=3
images = []
correct = 0
# Reading the image using OpenCV

print(filename)
path = os.path.join(image_path,'*tif')
files = glob.glob(path)
results = []
for fl in files:
    image = cv2.imread(fl)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    #images.append(image)
    image = np.array(image, dtype=np.uint8)
    image = image.astype('float32')
    image = np.multiply(image, 1.0/255.0) 
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = image.reshape(1, image_size,image_size,num_channels)
    
    ## Let us restore the saved model 
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('Breast_model2-3_aug.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    
    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()
    
    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")
    
    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0") 
    y_true = graph.get_tensor_by_name("y_true:0") 
    y_test_images = np.zeros((1, 4)) 
    
    
    ### Creating the feed_dict that is required to be fed to calculate y_pred 
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    results.append(result)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]
    #print(result)
    if 'b0' in fl:
        print('True Label: Benign')
    elif 'is0' in fl:
        print('True Label: InSitu')
    elif 'iv0' in fl:
        print('True Label: Invasive')
    elif 'n0' in fl:
        print('True Label: Normal')
        
    if np.argmax(result) == 0:
        print('Prediction: Benign')
    elif np.argmax(result) == 1:
         print('Prediction: InSitu')
    elif np.argmax(result) == 2:
         print('Prediction: Invasive')
    elif np.argmax(result) == 3:
         print('Prediction: Normal')
    
    if 'b0' in fl and np.argmax(result) == 0:
        correct += 1
        print('Correct')
    elif 'is0' in fl and np.argmax(result) == 1:
        correct += 1
        print('Correct')
    elif 'iv0' in fl and np.argmax(result) == 2:     
        correct += 1
        print('Correct')
    elif 'n0' in fl and np.argmax(result) == 3:
        correct += 1
        print('Correct')
    else:
        print('Wrong')
        
with open('aug_model2_4_results.py', 'w') as results: #Save results for ensemble learning
    results.write(str(results))
