import argparse

parser = argparse.ArgumentParser(description="do stuff")
parser.add_argument("hi-hi",type=int,help='no help hahahahahahahaha')

args = parser.parse_args()

print(args["hi-hi"])
