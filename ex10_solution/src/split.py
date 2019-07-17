import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.utils

def read():
    data = pd.read_csv('/home/madanm/Exercise_10/ex10/scenario/features.txt')
    print(data.head())

    print('the different columns in features.txt\n')
    print(data.columns)
    #data.to_csv(r'feat.csv')
    print(' ')
    print('total number of unique features')
    print(data.nunique())

    print('splitting based on the value of instances')
    mask = data.instance.str.contains("factor")
    df_factor = data[mask]
    
    df_cbmc = data[~mask]
    train_factor, test_factor = train_test_split(df_factor, test_size=0.5)
    train_cbmc, test_cbmc = train_test_split(df_cbmc, test_size=0.5)

    training_df = [train_factor,train_cbmc]
    testing_df = [test_factor,test_cbmc]

    df_train = pd.concat(training_df)
    df_test = pd.concat(testing_df)
    df_train = sklearn.utils.shuffle(df_train)
    df_test = sklearn.utils.shuffle(df_test)
    print(df_train.head())
    df_train = df_train['instance']
    df_test = df_test['instance']
    
    

    #df_train.to_csv(r'/home/madanm/Exercise_10/ex10/scenario/train_feat.csv',index=False,header=False)
    #df_test.to_csv(r'/home/madanm/Exercise_10/ex10/scenario/test_feat.csv',index=False,header=False)
    #df_train.to_csv(r'/home/madanm/Exercise_10/ex10/scenario/train_feat.txt',index=False,header=False)
    #df_test.to_csv(r'/home/madanm/Exercise_10/ex10/scenario/test_feat.txt',index=False,header=False)

def main():
    print("calling the function to read the data: features.txt")
    read()


if __name__ == "__main__":
    main()
