import pickle
import sys

name = sys.argv[1]
with open(name, 'rb') as f:
    contents = pickle.load(f)

print 'There are', len(contents), 'rows'

empty_row = 0
num_of_features = len(contents[0])
print num_of_features, 'number of features'
for r in contents:
    if(len(r) == 0):
        empty_row += 1
    if(len(r) != num_of_features):
        print 'Inconsistent columns on ', r

print 'there are ' + str(empty_row) + ' empty rows'
