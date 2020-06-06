import pandas as pd
import numpy as np
import numpy.random as nr
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

Features = np.array(pd.read_csv('Auto_Data_Features.csv'))
Labels = np.array(pd.read_csv('Auto_Data_Labels.csv'))
Labels = np.log(Labels)
scaler = StandardScaler()
Labels = scaler.fit_transform(Labels)
Auto_Data = np.concatenate((Features, Labels), 1)
print(Auto_Data.shape)

#%%
auto_prices = pd.read_csv('Automobile price data _Raw_.csv')

def clean_auto_data(auto_prices):
    #Function to load the auto price data set from a .csv file
    import pandas as pd
    import numpy as np
    
    #Remove rows with missing values, accounting for missing values coded as '?'
    cols = ['price', 'bore', 'stroke',
            'horsepower', 'peak-rpm']
    for column in cols:
        auto_prices.loc[auto_prices[column] == '?', column] = np.nan
    auto_prices.dropna(axis = 0, inplace = True)
    
    #Convert some columns to numeric values
    for column in cols:
        auto_prices[column] = pd.to_numeric(auto_prices[column])
    
    #Fix column names so the '-' character becomes '_'
    cols = auto_prices.columns
    auto_prices.columns = [str.replace('-', '_') for str in cols]
    
    return auto_prices
auto_prices = clean_auto_data(auto_prices)

print(auto_prices.columns)

#%%
auto_prices['price'] = np.log(auto_prices['price'])

#%%
marker_dic = {('gas','std'):'o', ('gas','turbo'):'s', ('diesel','std'):'x', ('diesel','turbo'):'^'}
markers = [marker_dic[(x,y)] for x,y in zip(auto_prices['fuel_type'], auto_prices['aspiration'])]

#%%
#Apply K-Means Clustering
nr.seed(2233)
col_dic = {0:'blue',1:'green',2:'orange',3:'gray',4:'magenta',5:'black'}
kmeans_2 = KMeans(n_clusters=2, random_state=0)
assignments_km2 = kmeans_2.fit_predict(Auto_Data)
assign_color_km2 = [col_dic[x] for x in assignments_km2]

#%%
def plot_auto_cluster(auto_prices, assign_color, markers):
    fig, ax = plt.subplots(2, 2, figsize=(12,11)) # define plot area
    x_cols = ['city_mpg', 'curb_weight', 'curb_weight', 'horsepower']
    y_cols = ['price', 'price', 'city_mpg', 'price']
    for x_col, y_col, i, j in zip(x_cols, y_cols, [0,0,1,1], [0,1,0,1]):
        for x,y,c,m in zip(auto_prices[x_col], auto_prices[y_col], assign_color, markers):
            ax[i,j].scatter(x,y,color = c, marker = m)
        ax[i,j].set_title('Scatter plot of ' + y_col + ' vs. ' + x_col) #Give the plot a main title
        ax[i,j].set_xlabel(x_col)
        ax[i,j].set_ylabel(y_col)
    plt.show()
    
plot_auto_cluster(auto_prices, assign_color_km2, markers)

#%%
#Trying with a 3 cluster model
nr.seed(4455)
kmeans_3 = KMeans(n_clusters=3, random_state=0)
assignments_km3 = kmeans_3.fit_predict(Auto_Data)
assign_color_km3 = [col_dic[x] for x in assignments_km3]
plot_auto_cluster(auto_prices, assign_color_km3, markers)

#%%
#4 Cluster model
nr.seed(223)
kmeans_4 = KMeans(n_clusters=4, random_state=0)
assignments_km4 = kmeans_4.fit_predict(Auto_Data)
assign_color_km4 = [col_dic[x] for x in assignments_km4]
plot_auto_cluster(auto_prices, assign_color_km4, markers)

#%%
#5 Cluster model
nr.seed(4443)
kmeans_5 = KMeans(n_clusters=5, random_state=0)
assignments_km5 = kmeans_5.fit_predict(Auto_Data)
assign_color_km5 = [col_dic[x] for x in assignments_km5]
plot_auto_cluster(auto_prices, assign_color_km5, markers)

#%%
#6 Cluster model
nr.seed(2288)
kmeans_6 = KMeans(n_clusters=6, random_state=0)
assignments_km6 = kmeans_6.fit_predict(Auto_Data)
assign_color_km6 = [col_dic[x] for x in assignments_km6]
plot_auto_cluster(auto_prices, assign_color_km6, markers)


#%%
#Evaluate the models
km_models = [kmeans_2, kmeans_3, kmeans_4, kmeans_5, kmeans_6]

def plot_WCSS_km(km_models, samples):
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    
    ## Plot WCSS
    wcss = [mod.inertia_ for mod in km_models]
    n_clusts = range(2,len(wcss) + 2)
    ax[0].bar(n_clusts, wcss)
    ax[0].set_xlabel('Number of clusters')
    ax[0].set_ylabel('WCSS')
    
    ## Plot BCSS
    ## Compute BCSS as TSS - WCSS 
    n_1 = (float(samples.shape[0]) * float(samples.shape[1])) - 1.0
    tss = n_1 * np.var(samples)
    bcss = [tss - x for x in wcss]
    ax[1].bar(n_clusts, bcss)
    ax[1].set_xlabel('Number of clusters')
    ax[1].set_ylabel('BCSS')
    plt.show()
    
plot_WCSS_km(km_models, Auto_Data)

#%%
#Silhouette Coefficients
assignment_list = [assignments_km2, assignments_km3, assignments_km4, assignments_km5, assignments_km6]

def plot_sillohette(samples, assignments, x_lab = 'Number of clusters'):
    silhouette = [silhouette_score(samples, a) for a in assignments]
    n_clusts = range(2, len(silhouette) + 2)
    plt.bar(n_clusts, silhouette)
    plt.xlabel(x_lab)
    plt.ylabel('SC')
    plt.show()

plot_sillohette(Auto_Data, assignment_list)



#%%
#Hierarchical Agglomerative Clustering
nr.seed(2233)
agc_2 = AgglomerativeClustering(n_clusters=2)
assignments_ag2 = agc_2.fit_predict(Auto_Data)
assign_color_ag2 = [col_dic[x] for x in assignments_ag2]
plot_auto_cluster(auto_prices, assign_color_ag2, markers)

#%%
#3 Clusters
nr.seed(4433)
agc_3 = AgglomerativeClustering(n_clusters=3)
assignments_ag3 = agc_3.fit_predict(Auto_Data)
assign_color_ag3 = [col_dic[x] for x in assignments_ag3]
plot_auto_cluster(auto_prices, assign_color_ag3, markers)

#%%
#4 Clusters
nr.seed(2663)
agc_4 = AgglomerativeClustering(n_clusters=4)
assignments_ag4 = agc_4.fit_predict(Auto_Data)
assign_color_ag4 = [col_dic[x] for x in assignments_ag4]
plot_auto_cluster(auto_prices, assign_color_ag4, markers)

#%%
#5 Clusters
nr.seed(6233)
agc_5 = AgglomerativeClustering(n_clusters=5)
assignments_ag5 = agc_5.fit_predict(Auto_Data)
assign_color_ag5 = [col_dic[x] for x in assignments_ag5]
plot_auto_cluster(auto_prices, assign_color_ag5, markers)

#%%
#6 Clusters
nr.seed(2288)
agc_6 = AgglomerativeClustering(n_clusters=6)
assignments_ag6 = agc_6.fit_predict(Auto_Data)
assign_color_ag6 = [col_dic[x] for x in assignments_ag6]
plot_auto_cluster(auto_prices, assign_color_ag6, markers)

#%%
assignment_list = [assignments_ag2, assignments_ag3, assignments_ag4, assignments_ag5, assignments_ag6]
plot_sillohette(Auto_Data, assignment_list)