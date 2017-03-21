import shelve
import librosa
import subprocess
d = shelve.open("data")
good_list=[]
def run(cmd, shell=True):
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=shell).communicate()[0]
files_in = run("ls tester/*.flac").split()[:-1]
files_out = run("ls tester/*.mp3").split()[:-1]
print files_in
print files_out
for i in range(len(files_in)):
    try:
        l1=librosa.core.get_duration(filename="%s" %files_in[i])
        l2 = librosa.core.get_duration(filename="%s" %files_out[i])
    except:
	break
    print l1
    print l2
    print files_in[i]
    print files_out[i]
    
    div=  max(l1,l2)/(min(l1,l2)+0.0000001)
    min_file = min(l1,l2)
    print div

    if div>1.3 or min_file <1:
	run("rm %s " %files_in[i])
	run("rm %s" %files_out[i])
    else:
	good_list.append([files_in[i],files_out[i]])	

d["train"] = good_list

d.close()
