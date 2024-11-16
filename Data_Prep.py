def data_cleaning(data):
    
    # Drop irrelevant features
    data = data.drop(['id', 'url', 'region_url', 'image_url', 'description', 'lat', 
                      'long', 'region', 'VIN', 'title_status', 'type', 'cylinders', 'county', 
                      'model', 'size'], axis='columns')

    # Drop rows with missing values
    data = data.dropna()

    # Remove 'harley-davidson' since it's not a car brand
    data = data[~(data.manufacturer == 'harley-davidson')]

    # Group manufacturers with low counts as 'other'
    manufacturer_count = data['manufacturer'].value_counts()
    manufacturer_less_than_100 = manufacturer_count[manufacturer_count < 100].index
    data['manufacturer'] = data['manufacturer'].apply(lambda x: 'other' if x in manufacturer_less_than_100 else x)
    
    # Add vehicle age column
    data['posting_date'] = pd.to_datetime(data['posting_date'], utc=True)
    data['posting_year'] = data['posting_date'].dt.year
    data = data.drop('posting_date', axis='columns')
    data['year'] = data['year'].astype('int32')
    data['vehicle_age'] = data['posting_year'] - data['year']
    data = data.drop(['year', 'posting_year'], axis='columns')
    
    # Remove rows with price equal to 0
    data = data[data.price != 0]

    # Function to remove outliers based on the IQR method within each state
    def remove_outliers(df):
        df_out = pd.DataFrame()
        for key, subdf in df.groupby('state'):
            Q1 = subdf['price'].quantile(0.25)
            Q3 = subdf['price'].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            reduced_df = subdf[(subdf.price >= lower) & (subdf.price <= upper)]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out

    # Apply outlier removal function
    data = remove_outliers(data)

    # Remove samples with unrealistic odometer readings
    data = data[(data.odometer > 1000) & (data.odometer <= 500000)]
    
    # Split data into vintage and non-vintage cars
    df_nonvintage = data[(data.vehicle_age > 0) & (data.vehicle_age < 30)]
    df_vintage = data[(data.vehicle_age >= 30) & (data.vehicle_age <= 100)]
    
    # Remove duplicates
    df_nonvintage = df_nonvintage.drop_duplicates()
    df_vintage = df_vintage.drop_duplicates()

    # One hot encoding
    df_nonvintage_encoded= pd.get_dummies(df_nonvintage, columns=['manufacturer', 'condition', 'fuel','transmission', 'drive',
                                      'paint_color', 'state'],  dtype=int, prefix='', prefix_sep='')
    df_vintage_encoded= pd.get_dummies(df_vintage, columns=['manufacturer', 'condition', 'fuel','transmission', 'drive',
                                      'paint_color', 'state'],  dtype=int, prefix='', prefix_sep='')


    # Remove 1 level from each categorical feature to avoid multicollinearity
    # state= nd (least count), pain_color=custom, fuel= other, transmission= other, drive= rwd  (least count), 
    # condition= salvage (least count), manufacturer= other 
    df_nonvintage_encoded.drop(['nd','other','custom','other','other','rwd','salvage'], axis='columns', inplace=True)
    df_vintage_encoded.drop(['nd','other','custom','other','other','rwd','salvage'], axis='columns', inplace=True)
    
        
    # Split into x(input) and y (target/output)
    x_nonvintage = df_nonvintage_encoded.drop('price', axis='columns')
    y_nonvintage = df_nonvintage_encoded.price
    x_vintage = df_vintage_encoded.drop('price', axis='columns')
    y_vintage = df_vintage_encoded.price


    # Since we have large samples, 80/20 split will suffice
    x_nonvintage_train, x_nonvintage_test,y_nonvintage_train, y_nonvintage_test = train_test_split(x_nonvintage, y_nonvintage, 
                                                                                                  test_size= 0.2, random_state=123) 
    x_vintage_train, x_vintage_test,y_vintage_train, y_vintage_test = train_test_split(x_vintage, y_vintage, 
                                                                                                  test_size= 0.2, random_state=123) 
    


    return  x_nonvintage_train, x_nonvintage_test,y_nonvintage_train, y_nonvintage_test, x_vintage_train, x_vintage_test,y_vintage_train, y_vintage_test                         



















