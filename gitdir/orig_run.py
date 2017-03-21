import math
import glob
import os
import sys

#os.system("apt-get -y update")
#os.system("apt-get -y dist-upgrade")
#os.system("apt-get -y upgrade")
#os.system("apt-get install -y python-tk")
#os.system("pip install youtube-dl")
#os.system("youtube-dl -U")
#os.system("apt-get install -y ffmpeg")
os.system("rm -rf videos")
os.system("rm -rf Saves")
os.mkdir("Saves")
os.mkdir("videos")

#if not os.path.isdir("aud"):
#    os.mkdir("aud")    
    
    
    

import json
import tensorflow as tf
import numpy as np
#from sklearn.metrics import precision_recall_fscore_support
#import matplotlib
import librosa
import os
import json
import subprocess
import voice
import time
import shelve

SAMPLE_RATE= 5000

d = shelve.open("data")

def variable_summaries(var):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def run(cmd, shell=True):
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=shell).communicate()[0]

#keys_train = []
#keys_test=[]
#vids = run("youtube-dl --get-id https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw -i")
#vids = vids.split()
#vids_=vids[0:2]
#print vids_
#for i in vids_:
#    keys_train = keys_train + voice.main(i)
#keys_test.append(voice.main(vids[6]))



#print keys_train
#print keys_test
#d["train"] = keys_train
#d["test"] = keys_test

keys_train = d["train"]
keys_test = d["test"]

for i in range(len(keys_train)):
     keys_train[i][0]=keys_train[i][0]+".flac"
   
for i in range(len(keys_test)):
     keys_test[i][1]=keys_test[i][1]+".flac"
d.close()




def run(cmd, shell=True):
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=shell).communicate()[0]



def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    #chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    #tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    #return mfccs,chroma,mel,contrast,tonnetz
    return mfccs,mel,contrast






def join_sound(file_in, file_out,size=300, mode=0):
#    print mode
    if mode == 0:
        
        inp,_ = librosa.load(file_in, sr=SAMPLE_RATE)
        dur=librosa.get_duration(inp,SAMPLE_RATE)
        x,_ = librosa.load(file_out,sr=SAMPLE_RATE)
        #print x
        new_dur= librosa.get_duration(x,sr=SAMPLE_RATE)

        out, _ = librosa.load(file_out, sr=int(SAMPLE_RATE*dur/new_dur))
        print out.shape
        print inp.shape
        if(len(inp)>len(out)):
            inp=inp[0:len(out)]
        else:
            out=out[0:len(inp)]
        #print len(out)
        #print len(inp)



    else:
        inp = file_in
        out = file_out

    
    for i in range(len(inp)):
        inp[i]= (inp[i]+1)/2

    for i in range(len(out)):
        out[i]= (out[i]+1)/2
    
    out = out[0:int(size*math.floor(len(out)/size))]
    inp = inp[0:int(size*math.floor(len(inp)/size))]
    inp=np.split(inp,len(inp)/size)
    out=np.split(out,len(out)/size)

    newInp = []
    newOut = []    
    index = 0
    for i in out:
        print i 
        for j in range(len(i)):
            print "j "+ str(j)
            newOut.append(i.tolist())
            newOut[-1].append(1/(j+1))
            newInp.append(inp[index][j])
        index+=1
            
    
#    for i in range(len(inp)):
#        inp[i]=np.append(inp[i],1/(i+1))
#        inp[i]=np.append(inp[i],1/(len(inp)+1))

    for i in range(len(out)):
        out[i]=np.append(out[i],1/(i+1))
        out[i]=np.append(out[i],1/(len(out)+1))
    

    
    return np.array(newOut[:-1]),np.vstack(newInp[:-1])
    
#inp,out = join_sound("aud/" + keys_train[10][0],"aud/" + keys_train[10][1])
#print inp
#print out

#print inp.shape
#print out.shape

#raise

def parse_audio_files(files,dir,ind=None):
    if ind is None:
        ind = len(files)
    inputs, outputs = [], []
    
    #print files
    #print len(files[0:ind])
    for fn in files[0:ind]:
        print "in"
        if len(fn) == 2:
            try:
                inp, out = join_sound(dir + "/" + fn[0],dir + "/" + fn[1])
            
            
                #print ("inp",inp)
                #print ("out",out)
                #time.sleep(2)
            
                #nexts = np.append(nexts, nxt)
                for i in out:
                    outputs.append(i.tolist())
                
                for i in inp:
                    inputs.append(i.tolist())
         
            except Exception as e:
                print e                    
    return np.array(inputs),np.array(outputs)




#print join_sound(np.array((1000,1000,1000,1000)), np.array((1,2,3,4,5,6,7,8,9,10)),mode=1)



train_x, train_y = parse_audio_files(keys_train,"aud",1)

print (train_x, "trainx")
print (train_x.shape, "trainx.shape")
print (train_y, "trainy")
print (train_y.shape, "trainy.shape")


"""
x_list = test_x.tolist()

print "len \n"
print len(x_list)
y_list = test_y.tolist()
#len(x_list)

json_ = open("test_data.json","w")

for i in xrange(len(x_list)):
    json_.write(json.dumps({"sound":x_list[i], "key":y_list[i][1]}) + "\n")
    #json_.write(json.dumps({"sound":x_list[i], "key":0}) + "\n")
#x_json = x_json[1:-1]





#json_.write(x_json)

json_.close()
"""

training_epochs = 5000
print train_x.shape
#n_dim = train_x.shape[1]
#n_classes = train_y.shape[1]
n_dim=1
n_classes=2
n_hidden_units_one = 2
n_hidden_units_two = 2
sd = 1 / np.sqrt(n_dim)
learning_rate = 100
def run_training():
    
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
    y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)
    init = tf.global_variables_initializer()
    
    
    #keys_placeholder = tf.placeholder(tf.int64, shape=(None,))
    #keys = tf.identity(keys_placeholder)
    
    
    #inputs = {'key': keys_placeholder.name, 'sound': X.name}
    #tf.add_to_collection('inputs', json.dumps(inputs))
    
    
    #prediction = tf.argmax(y_, 1)
    #scores = tf.nn.softmax(y_)
    #outputs = {'key': keys.name,
    #       'prediction': prediction.name,
    #       'scores': scores.name}
    #tf.add_to_collection('outputs', json.dumps(outputs))

    cost_function = -tf.reduce_sum(Y * tf.log(y_))
    #cost_function = tf.reduce_sum(tf.pow(Y-y_,2))
#    tf.summary.scalar('cost', cost_function)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)



    cost_history = np.empty(shape=[1],dtype=float)
    y_true, y_pred = None, None
    #with tf.name_scope('accuracy'):
        #with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_, 1))
    #with tf.name_scope('accuracy'):
    
    correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        test_writer = tf.summary.FileWriter('Saves/test')
        for epoch in range(training_epochs):
            feed_dict={X:[[1],[1/2],[1/3],[1/4],[1/5],[1/6],[1/7],[1/8],[1/9],[1/10],[1/11],[1/12],[1/13],[1/14]],Y:[[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,1]]}
            _,cost = sess.run([optimizer,cost_function],feed_dict=feed_dict)
            cost_history = np.append(cost_history,cost)
            print(epoch, cost)   
  #          if ((epoch%10) == 0) or epoch+1==training_epochs:
#                summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict)
 #               test_writer.add_summary(summary, epoch)
            if ((epoch%1000) == 0):    
                print "saving"
                saver.save(sess, os.path.join("Saves", 'model.ckpt'), global_step=epoch)
            if epoch+1==training_epochs:
                saver.save(sess, os.path.join("Saves", 'export'))

run_training()               
  
    

