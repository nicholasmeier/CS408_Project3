import pickle;

with open('data/y_train.pickle', 'rb') as handle :
    Y_train = pickle.load(handle)

with open('data/y_valid.pickle', 'rb') as handle :
    Y_valid = pickle.load(handle)

with open('data/y_test.pickle', 'rb') as handle :
    Y_test = pickle.load(handle)

def total_ones(df):
    total_buggy = 0
    for x in range(0,len(df)):
        if df[x] == 1:
            total_buggy += 1
    return total_buggy


print ("Y_train",total_ones(Y_train), "/", len(Y_train) , "(" , total_ones(Y_train) / len(Y_train) * 100, "%", ")", "buggy")
print ("Y_valid", total_ones(Y_valid) , "/", len(Y_valid) ,"(" ,total_ones(Y_valid) / len(Y_valid) * 100, "%", ")" ,"buggy")
print ("Y_test", total_ones(Y_test) , "/",  len(Y_test),"(" , total_ones(Y_test) / len(Y_test) * 100, "%", ")" ,  "buggy")

