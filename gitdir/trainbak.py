import time
import math
import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import random
import shelve
from decimal import Decimal
import math
import argparse
import sys
import shutil
from datetime import datetime
parser = argparse.ArgumentParser(description="Machine learning algorithm for generating audio")

parser.add_argument("--logdir",type=str, default="Saves" ,help='directory for saving')
parser.add_argument("--data_dir",type=str,default="aud" ,help='directory with sound data (default aud)')
parser.add_argument("--generations",type=int,default=10000,help='number of generations (default 10000)')
parser.add_argument("--num_files",type=int,default=-1,help='number of files or -1 for all of them(default -1)')
parser.add_argument("--checkpoint_every", type=int,default=50,help="number of generations until checkport")
parser.add_argument("--Sample_rate", type=int,default=5000,help="Sample rate")
parser.add_argument("--file_split", type=int,default=1000,help="Number of files per input")
parser.add_argument("--learning_rate", type=float,default=0.01,help="learning rate (default 0.01)")

#time_format = "{0:%Y-%m-%dT%H-%M-%S}"

full_start = time.time()

args = parser.parse_args()

if not os.path.isdir(args.logdir):
    os.mkdir(args.logdir)
else:
    if(raw_input("a directory with the chosen name already exists. Do you want to overide? Y/n:").lower() == "y"):
        shutil.rmtree(os.path.join("/home/guyknaan/voiceswap",args.logdir))
        os.mkdir(args.logdir)
    else:
        print "not overiding"
        sys.exit()

if not os.path.isdir(args.data_dir):
    raise ValueError("the chose data dir: %s does not exist" %args.data_dir)    

SAMPLE_RATE= args.Sample_rate



d = shelve.open("data")

keys_train = d["train"]
keys_test = d["test"]



for i in range(len(keys_train)):
     keys_train[i][0]=keys_train[i][0]+".flac"

for i in range(len(keys_test)):
     keys_test[i][1]=keys_test[i][1]+".flac"
d.close()

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

emp0 = np.empty(0)

emp1 = np.empty(1)

def join_sound(file_in, file_out,size=300, mode=0):
#    print mode
    try:
        if mode == 0:
    
            inp,_ = librosa.load(file_in, sr=SAMPLE_RATE)
            dur=librosa.get_duration(inp,SAMPLE_RATE)
            x,_ = librosa.load(file_out,sr=SAMPLE_RATE)
            #print x
            new_dur= librosa.get_duration(x,sr=SAMPLE_RATE)

            out, _ = librosa.load(file_out, sr=int(SAMPLE_RATE*dur/new_dur))
#            print out.shape
    #        print inp.shape
            if(len(inp)>len(out)):
                inp=inp[0:len(out)]
            else:
                out=out[0:len(inp)]

        else:
            inp = file_in
            out = file_out


        for i in range(len(inp)):
            inp[i]= (inp[i]+1)/2.0
    #        print inp[i]

        for i in range(len(out)):
            out[i]= (out[i]+1)/2.0
    #    print out
    #    print inp

        newInp = []
        newOut = []
        index = 0

    
        for i in range(len(inp)):
            if (i + size < len(inp)):
                newInp.append(inp[i+int(size/2)])
                newOut.append(out[i:i+size].tolist())

        for i in range(len(newOut)):
            newOut[i].append(float(i)/len(newOut))

    except ValueError as e:
        print e
        return emp1,emp1
    except MemoryError as e:
        print e
        return emp0,emp0
    return np.array(newOut[:-2]),np.array(newInp[:-2])
#print (join_sound("aud/KXEGWMOFSFoutput179.mp3","aud/ITMUVRTUURoutput561.mp3"),"val")



def parse_audio_files(files,dir,ind=None,size=300):
    if ind is None:
        ind = len(files)
    inputs, outputs = [], []
    
    count_num=0
    
    #print files
    #print ind
    #print len(files[0:ind])
    for fn in files[0:ind]:
        count_num+=1
        print "loading the %sth file: %s" %(str(count_num),fn[0])
        if len(fn) == 2:
            try:
                inp, out = join_sound(dir + "/" + fn[0],dir + "/" + fn[1])
                if inp is emp0:
                    return np.array(inputs),np.vstack(np.array(outputs))
                if inp is not emp1:
                    
                    for i in out:
                        outputs.append(i)

                    for i in inp:
                        inputs.append(i)
                        
            except ValueError as e:
                print e
                #raise
            except MemoryError as e:
                return np.array(inputs[0:-10]),np.vstack(np.array(outputs[0:-10]))
    return np.array(inputs),np.vstack(np.array(outputs))




def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode



tr_features, tr_labels = parse_audio_files(keys_train,args.data_dir, ind=args.num_files if args.num_files != -1 else None,size=args.file_split)

#print args.num_files if args.num_files != -1 else None

#print tr_features.shape
#print tr_labels.shape



n_dim = tr_features.shape[1]
n_classes = tr_labels.shape[1]
training_epochs = args.generations
n_hidden_units_one = 500
n_hidden_units_two = 550
sd = 1
learning_rate = args.learning_rate

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)


W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)


W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
#y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)
y_ = tf.nn.sigmoid(tf.matmul(h_2,W) + b)
init = tf.global_variables_initializer()

#cost_function = -tf.reduce_mean(Y * tf.log(y_))

cost_function=tf.reduce_mean(tf.square(tf.sqrt(y_)-tf.sqrt(Y))/(2**0.5))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        start_time = time.time()
        #print sess.run(y_)
        _,cost = sess.run([optimizer,cost_function],feed_dict={X:tr_features,Y:tr_labels})
        #cost_history = np.append(cost_history,cost)
        duration = time.time() - start_time
        print('step {:d} - loss = {:.2E}, ({:.3f} sec/stp)'.format(epoch, cost, duration))

        if epoch%args.checkpoint_every==0:
            print "starting Save"
            saver.save(sess, os.path.join(args.logdir, 'model.ckpt'), global_step=epoch)
            print "Susessfully Saved"

        if epoch+1==training_epochs:
                print "starting Save"
                saver.save(sess, os.path.join(args.logdir, 'export'))
                print "Susessfully Saved"

end_time = time.time()
diffrence = end_time - full_start
hours = int(math.floor(diffrence/(60**2)))
diffrence-=60*60*hours
minutes = int(math.floor(diffrence/60))
diffrence-=60*minutes
seconds = diffrence
print "total time for training: {} hours, {} minutes, and {} seconds".format(hours,minutes,round(seconds,2))
