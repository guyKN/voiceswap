import voice
import sys
import subprocess
import shelve
d = shelve.open("data")
def run(cmd, shell=True):
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=shell).communicate()[0]

run("mkdir aud")
try:
    num_train=int(sys.argv[1])
    num_test=int(sys.argv[2])
except:
    raise ValueError("you must provide two arguments for downloading")

print num_train
print num_test

keys_train = []
keys_test=[]
print "youtube-dl  --max-downloads %s --get-id https://www.youtube.com/channel/UCiUHGAbtjCrLO3ZPrTfle3Q" %str(num_train+num_test+3)
vids = run("youtube-dl  --max-downloads %s --get-id https://www.youtube.com/channel/UCiUHGAbtjCrLO3ZPrTfle3Q" %str(num_train+num_test+3))
vids = vids.split()
print vids
print vids

try:

    vids_=vids[1:num_train+1]
    print vids_
    for i in vids_:
        if(not(i.startswith("-"))):
            keys_train = keys_train + voice.main(i)
        else:
            print "there is a - %s" %i
    


    vids__=vids[num_train+2:num_train+num_test+2]
    for i in vids__:
        if(not(i.startswith("-"))):
            keys_test = keys_test + voice.main(i)
        else:
            print "there is a -"

    print keys_train
    print keys_test
except:
    print "error"
    d["train"] = keys_train
    d["test"] = keys_test
    d.close()
    raise

d["train"] = keys_train
d["test"] = keys_test

#keys_train = d["train"]
#keys_test = d["test"]


d.close()

#run("gsutil rm gs://voice-recognition-151100-ml/*.flac")
