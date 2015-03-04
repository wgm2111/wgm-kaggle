# import the data



import numpy as np


# Read in data as txt
# --
ftrain = 'train.csv'
ftest = 'test.csv'

# Read in training set
# --
vecs = []
vals = []
vlines = []
Icards = []

 
for i, line in enumerate(open(ftrain)):
    if i==0 : 
        continue
    else:
        sline = line.strip().split(',')
        vline = np.array(sline, dtype='int')
        vlines.append(vline)
        card, val = vline[:-1], vline[-1]
        Icard = 13 * (card[::2]-1) + (card[1::2]-1)
        val = vline[-1]
        vec = np.zeros(52, dtype='int')
        vec[Icard] = 1
        vecs.append(vec)
        vals.append(val)
        Icards.append(Icard)
        


train_data = np.array(vlines, dtype='int')
training_card_array = np.array(vecs, dtype='int')
training_value_array = np.array(vals, dtype='int')


# Read in test data
# --
vlines = []
vecs = []
vals = []
 
for i, line in enumerate(open(ftest)):
    if i==0 : 
        continue
    else:
        sline = line.strip().split(',')
        vline = np.array(sline, dtype='int')[1:]
        vlines.append(vline)
        Icard = 13 * (vline[::2] - 1) + (vline[1::2] - 1)
        vec = np.zeros(52, dtype='int')
        vec[Icard] = 1
        vecs.append(vec)

test_data = np.array(vlines, dtype='int')
test_card_array = np.array(vecs, dtype='int')


# Define a function for checking the true answer

# Define testing function 
# --
def true_predictor(card_array):

    # Test correct answers
    # --
    suit_matches_arr = card_array.reshape((-1, 4,13)).sum(-1)
    number_matches_arr = card_array.reshape((-1,4,13)).sum(1)
    
    # initialize output
    out = -np.ones((card_array.shape[0]), dtype='int')

    for (i, suit_matches, number_matches) in zip(
        range(1000000), suit_matches_arr, number_matches_arr):
    

        flush = (suit_matches.max()==5)
        
        two = ((number_matches==2).sum()==1)
        twotwo = ((number_matches==2).sum()==2)
        three = ((number_matches==3).sum()==1)
        full = two and three
        four = ((number_matches==4).sum()==1)
        Iunmatched = np.arange(13)[number_matches==1]
        
        strait = (
            (Iunmatched.size==5) and ((Iunmatched.max()-Iunmatched.min())==4) 
            or ((Iunmatched.size==5) and Iunmatched[0]==0 and Iunmatched[1:].min()==9))
        
        # _cn = np.arange(13)[number_matches==1] # for straits
        # strait = ((number_matches==1).sum()==5) and ((_cn.max()-_cn.min())==4)
        
        royal = flush and strait and (Iunmatched.min()==0) and (Iunmatched.max()==12)
        
        if royal:
            _val = 9 
        elif flush * strait:
            _val = 8
        elif four:
            _val = 7
        elif full:
            _val = 6
        elif flush:
            _val = 5
        elif strait:
            _val = 4
        elif three:
            _val = 3
        elif twotwo:
            _val= 2
        elif two:
            _val=1
        else:
            _val = 0  
        
        # append store the best hand indicator
        out[i] = _val
    return out


# Compute the true output
# --
test_card_array_true = true_predictor(test_card_array)







# Test the true output if run as a script
# --
if __name__ == "__main__":
    

    # 
    my_training_value_array = true_predictor(training_card_array)
    for i, _my, _tr in zip(range(100000),my_training_value_array, training_value_array):
        if _my==_tr:
            pass
        else:
            print("card index = {2}, my_training_value = {0}, true_training_value = {1}".format(
                    _my, _tr, i))

