# 
def preprocessData(data_df, sRatio=None):  
    r'''  
    Preprocess data frame  
  
    @param data_df(DataFrame):  
        Training DataFrame  
  
    @param sRatio(float):  
        if splitRation is not None:  
            (train_data, train_label, test_data, test_label)  
        else:  
            (train_data, train_label)  
    '''  
    # Remove column 'Name'  
    data_df = data_df.drop(['Name'], axis=1)  
  
    # Show number of rows with null value  
    print("\t[Info] Show number of rows with null value:")  
    print(data_df.isnull().sum())  
    print("")  
  
    # Fill null with age mean value on 'Age' column  
    print("\t[Info] Handle null value of Age column...")  
    age_mean = data_df['Age'].mean()  
    data_df['Age'] = data_df['Age'].fillna(age_mean)  
  
    # Show number of rows with null value  
    print("\t[Info] Show number of rows with null value:")  
    print(data_df.isnull().sum())  
    print("")  
  
    print("\t[Info] Translate value of column Sex into (0,1)...")  
    data_df['Sex'] = data_df['Sex'].map({'female':0, 'male':0}).astype(int)  
  
    print("\t[Info] OneHot Encoding on column Embarked...")  
    data_df = pd.get_dummies(data=data_df, columns=['Embarked'])  
  
    # Show top 2 records  
    print("\t[Info] Show top 2 records:")  
    pprint(data_df.as_matrix()[:2])  
    print("")  
  
    ndarray = data_df.values  
    print("\t[Info] Translate into ndarray(%s) with shape=%s" % (ndarray.__class__, str(ndarray.shape)))  
    print("\t[Info] Show top 2 records:\n%s\n" % (ndarray[:2]))  
  
    # Separate labels with features  
    Label = ndarray[:,0]  
    Features = ndarray[:,1:]  
  
    # Normalized features  
    print("\t[Info] Normalized features...")  
    from sklearn import preprocessing  
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))  
    scaledFeatures = minmax_scale.fit_transform(Features)  
    print("\t[Info] Show top 2 records:\n%s\n" % (scaledFeatures[:2]))  
  
    if sRatio:  
        # Splitting data into training/testing part  
        print("\t[Info] Split data into training/testing part")  
        msk = np.random.rand(len(scaledFeatures)) < sRatio  
        trainFeatures = scaledFeatures[msk]  
        trainLabels = Label[msk]  
        testFeatures = scaledFeatures[~msk]  
        testLabels = Label[~msk]  
        print("\t[Info] Total %d training instances; %d testing instances!" % (trainFeatures.shape[0], testFeatures.shape[0]))  
        return (trainFeatures, trainLabels, testFeatures, testLabels)  
    else:  
        return (scaledFeatures, Label)  