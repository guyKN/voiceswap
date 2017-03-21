def rm_extra(val):
    brackets = 0
    print val
    output=""
    for i in val:
        print "in"
        if i == "<":
            brackets+=1
            print "<"
        elif i==">":
            print ">"
            brackets-=1
        elif(brackets==0):
            output=output+i
    return output 




def main(name):
    #did=0
    current=""
    output =[]
    vtt=open(name,"r").read()
    vtt=vtt.split("\n")
    vtt = list(reversed(vtt))
    for i in vtt:
    	split = i.split("-->")
    	if len(split)==2:
    	    did = 1
            output.append({"start":split[0], "end":split[1].split("align")[0], "sub":current})
            current = ""		
        else:
            try:
                print rm_extra(i)
                current = rm_extra(i)+" "+current
            except TypeError as e:
                print e

    return list(reversed(output))


if __name__ == "__main__":
	print(main("subs/HCQAZABQYB.vtt"))
    #print rm_extra("<hello> good <bad> great")
