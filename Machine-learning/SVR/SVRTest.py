def Model(train_dataset, train_labels, test_dataset, test_labels, index):
    print("Ind", index)
# The SVR Model
    regressor = SVR(kernel=Dict[Feature]['kernel'][index[0]],
                    C=Dict[Feature]['C'][index[1]],
                    gamma=Dict[Feature]['gamma'][index[2]],
                    epsilon=Dict[Feature]['epsilon'][index[3]])
# Fitting the lodel
    regressor.fit(train_dataset, train_labels)
# Prediction from the model
    y_pred = regressor.predict(test_dataset)
# Obtaining the MSE and MAPE
    MSE = ((y_pred-test_labels)**2).mean()
    MAPE = mean_absolute_percentage_error(test_labels, y_pred)
    return MAPE, MSE