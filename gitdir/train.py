

from scipy.io import wavfile
import subprocess
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

# test
import matplotlib.pyplot as plt

from datetime import datetime
parser = argparse.ArgumentParser(description="Machine learning algorithm for generating audio")

def run(cmd, shell=True):
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=shell).communicate()[0]

#parser.add_argument("--logdir",type=str, default="Saves" ,help='directory for saving')
parser.add_argument("--data_dir",type=str,default="aud" ,help='directory with sound data (default aud)')
parser.add_argument("--generations",type=int,default=10000,help='number of generations (default 10000)')
parser.add_argument("--num_files",type=int,default=-1,help='number of files or -1 for all of them(default -1)')
parser.add_argument("--checkpoint_every", type=int,default=50,help="number of generations until checkport")
parser.add_argument("--Sample_rate", type=int,default=5000,help="Sample rate")
parser.add_argument("--file_split", type=int,default=10000,help="Number of files per input")
parser.add_argument("--learning_rate", type=float,default=0.01,help="learning rate (default 0.01)")
parser.add_argument("--action", type=int,default=3,help="1 for turning files to data, 2 for learning and 3 for both")
parser.add_argument("--restore", type=str,default=None,help="restore previus session")
parser.add_argument("--generate_path", type=str,default=None,help="Generateed file origanal path")
parser.add_argument("--generate_new", type=str,default=None,help="Path to new file")
args = parser.parse_args()


if bool(args.generate_path)^bool(args.generate_new):
    raise ValueError("You must specify either both generate_path and generate_new or None")



if args.restore:
    date_start = "/".join(args.restore.split("/")[:-1])
else:
    date_start = "{0:%Y-%m-%dT%H-%M-%S}::".format(datetime.now())+str(run("ls checkpoints| wc -l")[:-1])

#os.mkdir("checkpoints/%s" %date_start)

#os.mkdir("summarys/%s" %date_start)



#time_format = "{0:%Y-%m-%dT%H-%M-%S}"

full_start = time.time()


args.logdir = "checkpoints/%s" %date_start

print args.logdir

if not os.path.isdir(args.logdir):
    os.mkdir(args.logdir)
#else:
 #   if(raw_input("a directory with the chosen name already exists. Do you want to overide? Y/n:").lower() == "y"):
  #      shutil.rmtree(os.path.join("/home/guyknaan/voiceswap",args.logdir))
   #     os.mkdir(args.logdir)
    #else:
     #   print "not overiding"
      #  sys.exit()

#if not os.path.isdir(args.data_dir):
#    raise ValueError("the chose data dir: %s does not exist" %args.data_dir)    

SAMPLE_RATE= args.Sample_rate



d = shelve.open("data")

keys_train = d["train"]
keys_test = d["test"]


"""
for i in range(len(keys_train)):
     keys_train[i][0]=keys_train[i][0]#+".flac"

for i in range(len(keys_test)):
     keys_test[i][1]=keys_test[i][1]#+".flac"
d.close()
"""

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
            
            #replace maybe
            #out, _ = librosa.load(file_out, sr=int(SAMPLE_RATE*dur/new_dur))
            out, _ = librosa.load(file_out, sr=SAMPLE_RATE)
    #        print inp.shape
            #turn the numbers to the range from 0 to 1
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

 #       print inp.shape
 #       print out.shape

        newInp = []
        newOut = []
        index = 0

                
        out = out[0:int(size*math.floor(len(out)/size))]
        inp = inp[0:int(size*math.floor(len(inp)/size))]
        inp=np.split(inp,len(inp)/size)
        out=np.split(out,len(out)/size)    
        
        for i in range(len(out)):
            wavfile.write("out_test/out_file%03d.wav" %i,SAMPLE_RATE,np.array(out[i]))
            wavfile.write("out_test/in_file%03d.wav" %i,SAMPLE_RATE,np.array(inp[i]))
            out[i]=np.append(out[i],float(i)/len(out))
        
        
        
        
        
    except ValueError as e:
        print e
        raise
        return emp1,emp1
    except MemoryError as e:
        print e
        return emp0,emp0
    return np.array(out[:-2]),np.array(inp[:-2])
#a,b = join_sound("aud/KXEGWMOFSFoutput179.mp3","aud/ITMUVRTUURoutput561.mp3")

join_sound("tester/DLPTOAUSIQ0211.flac","tester/DLPTOAUSIQ0211_output.mp3",size=10000)
raise
#print a.shape
#print b.shape
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
                #inp, out = join_sound(dir + "/" + fn[0],dir + "/" + fn[1],size=size)
                # temp temp
                inp, out = join_sound(fn[0],fn[1],size=size)
                
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

d = shelve.open("preloaded")
if args.action == 1 or args.action == 3: 
    try:
        tr_features, tr_labels = parse_audio_files(keys_train,args.data_dir, ind=args.num_files if args.num_files != -1 else None,size=args.file_split)
    except Exception as e:
        raise
        d["features"] = tr_features
        d["labels"] = tr_labels
         
        raise
    d["features"] = tr_features
    d["labels"] = tr_labels
else:
    tr_features = d["features"]    
    tr_labels = d["labels"]


#tr_features, tr_labels = np.random.rand(100,20),np.random.rand(100,20)

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
b = tf.Variable(tf.random_normal([n_classes]))
y_ = tf.nn.sigmoid(tf.matmul(h_2,W)+b)

#cost_function = -tf.reduce_mean(Y * tf.log(y_))

cost_function=tf.reduce_mean(tf.square(tf.sqrt(y_)-tf.sqrt(Y))/(2**0.5))
tf.summary.scalar('cost', cost_function)
#adapt_rate = tf.placeholder(tf.float32, shape=[])
#optimizer = tf.train.GradientDescentOptimizer(adapt_rate).minimize(cost_function)

optimizer=tf.train.AdagradOptimizer(args.learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None

saver = tf.train.Saver()

init = tf.global_variables_initializer()

sumarizer = tf.summary.FileWriter("summarys/%s" %date_start)

def run_training():
    #rate=float(learning_rate)
    with tf.Session() as sess:
        done=0
        if args.restore:
            saver.restore(sess,args.restore)
        else:
            sess.run(init)
        
        if args.generate_path:
            loaded, _  = join_sound("aud/XPLQAKERFH0403.flac",args.generate_path)
            print loaded.shape
            output = sess.run(y_, feed_dict={X:loaded})
            output = output.reshape(output.size)
            output = (output*2) -1 
            print output
            print output.shape
            wavfile.write(args.generate_new,SAMPLE_RATE,output)
            
            return
        try:
            for epoch in range(training_epochs):
                start_time = time.time()
                #print sess.run(y_)
                feed_dict={X:tr_features,Y:tr_labels}
                _,cost = sess.run([optimizer,cost_function],feed_dict=feed_dict)
                #cost_history = np.append(cost_history,cost)
                duration = time.time() - start_time
                print('step {:d} - loss = {:e}, ({:.3f} sec/stp)'.format(epoch, cost, duration))
    
                if epoch%args.checkpoint_every==0:
                    print "Saving"
                    saver.save(sess, os.path.join(args.logdir, 'model.ckpt'), global_step=epoch)
                if epoch%10 == 0:
                    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict)
                    sumarizer.add_summary(summary, epoch)
                if epoch+1==training_epochs:
                    print "Saving"
                    saver.save(sess, os.path.join(args.logdir, 'export'))
        except:
            saver.save(sess, os.path.join(args.logdir, 'export'))
            raise
if args.action == 2 or args.action == 3:
    run_training()

end_time = time.time()
diffrence = end_time - full_start
hours = int(math.floor(diffrence/(60**2)))
diffrence-=60*60*hours
minutes = int(math.floor(diffrence/60))
diffrence-=60*minutes
seconds = diffrence
print "total time for training: {} hours, {} minutes, and {} seconds".format(hours,minutes,round(seconds))
