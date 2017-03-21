import json
import subprocess
import string
import random
import sys
import shelve
import time
import subs
BUCKET = "gs://voice-recognition-151100-ml"

#d["voice"]=[]



    
def run(cmd, shell=True):
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=shell).communicate()[0]


def find_code(fileName):
    token = run("gcloud auth print-access-token", True)
    token = token[:-1]
    req = open("sync-req.json","w")    
    req.write("""{'config': {'encoding':'FLAC','sampleRate': 44100,'languageCode': 'en-US'},'audio': {'uri':'%s/%s'}}""" %(BUCKET,fileName))
    #req.write("""{'config': {'encoding':'FLAC','sampleRate': 44100,'languageCode': 'en-US'},'audio': {'uri':'gs://my-voices/thing.flac'}""")
    req.close()    
    code = ("""curl -s -k -H "Content-Type: application/json" -H "Authorization: Bearer %s" https://speech.googleapis.com/v1beta1/speech:syncrecognize -d @sync-req.json""" %(token))    
    #print(code)
    return(run(code))



def secs(s):
    s=s.split(":")
    return int(s[0])*3600+int(s[1])*60+float(s[2])

    

def main(vid):
    keys = []
    words = []
    fileName=''.join(random.choice(string.ascii_uppercase) for _ in range(10))
    run("rm subs/*")
    run("rm videos/*")
    print "youtube-dl --extract-audio --audio-format mp3 --write-auto-sub " + vid
    run("youtube-dl --extract-audio --audio-format mp3 --write-auto-sub " + vid, True)
        
    run("mv *.mp3 videos/" + fileName+".mp3", True)
    run("mv *.vtt subs/{}.vtt".format(fileName))

    print("done")


    print("num 1")

    run('ffmpeg -i videos/' + fileName + '.mp3 -acodec flac -bits_per_raw_sample 16 -ar 44100 -ac 1 -y videos/' + fileName + '.flac', True)

    print("num 2")
    prev = 0
    try:
        subtitles = subs.main("subs/%s.vtt" %fileName)
    except:
        return []
    print subtitles
    #time.sleep(5)
    ind =  1#note : subtitles are shifted in 1 (?)
    # create input audio using sub title sentences
    for cur_sub in subtitles:
        file_name_with_zeros = fileName + str(ind).zfill(4) + ".flac"      
        print cur_sub
        keys.append([file_name_with_zeros])
        start = secs(cur_sub["start"])
        end = secs(cur_sub["end"])
        words.append(cur_sub["sub"])
        
        cmd = "ffmpeg -ss {} -i videos/{}.flac  -t {} videos/{}".format(prev,fileName, end-prev, file_name_with_zeros)
        print cmd
        run(cmd)
        prev = end
        ind += 1
        
    # delete the origanal files
    run("rm videos/" + fileName + ".flac", True)
    run("rm videos/" + fileName + ".mp3", True)
    
    # create output audio using sub title sentences in text to speach conversion
    for word_idx in range(len(words)-1):
        used_idx = word_idx +1
        name =  fileName + str(used_idx).zfill(4) + "_output.mp3"
        keys[word_idx].append(name)
        run('wget -q -U Mozilla -O ' + name + ' "http://translate.google.com/translate_tts?ie=UTF-8&total=1&idx=0&textlen=32&client=tw-ob&q=' + words[used_idx] + '&tl=En-gb"', True)


    print "moving"
    run("mv *.mp3 aud", True)

    run("mv videos/* aud")
    keys = keys[2:-2]
    print keys
    
    #prev = 0
    #shifted_keys =[]
    #for cur_key in keys:
    #    n1 = prev
    #    n2 = cur_key[1]
    #    shifted_keys.append([n1,n2])
    #    prev = cur_key[0]
    #    run("play ")

    #shifted_keys = shifted_keys[2:-2]
    
    return keys



def convert():
    foldName = "small"
    files = run("ls " + foldName).split("\n")[:-1]    
    for i in files:
        split = i.split(".")[0]+".wav"
        print ("i", i)
        print ("split", split)
        print "ffmpeg -y -hide_banner -i %s/%s %s/%s" %(foldName, i,foldName,split)
        
        run("ffmpeg -y -hide_banner -i %s/%s %s/%s" %(foldName, i,foldName,split))
        run("rm " + i)
    
if __name__=='__main__':
    #print(find_code("none.flac"))
    main("fMFtgo2HpDY"  )
