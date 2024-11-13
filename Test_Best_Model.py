# Best Model
rf= RandomForestRegressor(n_jobs=-1, random_state=123)
rf.fit(x_nonvintage_train, y_nonvintage_train)
rf.score(x_nonvintage_test, y_nonvintage_test)


def predict_price(manufacturer, condition, fuel, odometer, transmission, drive, paint_color, state, vehicle_age):

    
    # Initialize an array of the size of all the columns in x_nonvintage
    x = np.zeros(len(x_nonvintage.columns)) 

    #set first two indices to our numeric features
    x[0] = odometer              
    x[1] = vehicle_age

    # list of categorical features
    features = [state, manufacturer, condition, fuel, transmission, paint_color, drive]
    
    # for all categores in a categorical feature, loop over the categories
    for category in features:
        
        # Check if the category is in our data x_nonvintage
        if category in x_nonvintage.columns:
            
            # Get the index for the one hot encoded category in that data
            idx = np.where(x_nonvintage.columns == category)[0][0]

            #set that category index=1 (categories for that categorical feature that are not in x_nonvintage will be 0)
            x[idx] = 1
       
    # prediction
    return rf.predict([x])[0]


# Test
predict_price('bmw', 'like-new', 'diesel', 120000, 'automatic', '4wd', 'black', 'la', 5)
