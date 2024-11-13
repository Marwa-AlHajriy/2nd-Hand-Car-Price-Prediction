
def graphical_numerical_summaries(data, variable)
    plt.figure(figsize=(8, 4))
    plt.subplot(1,2,1)
    sns.histplot(data[variable], bins=50, kde=True)
    plt.title(f'Histogram of {variable} Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    
    plt.subplot(1,2,2)
    sns.boxplot(x=data[variable])
    plt.title(f'Box Plot of {variable} Distribution')

    return data[variable].describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1])
