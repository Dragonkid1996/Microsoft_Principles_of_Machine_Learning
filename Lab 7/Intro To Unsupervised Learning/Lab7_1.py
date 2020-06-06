import numpy as np
import numpy.random as nr
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

#%%
def make_dist(mean, cov, dist_num, n = 100, seed = 123):
    nr.seed(seed)
    sample = nr.multivariate_normal(mean, cov, n) #Compute 2-D Normally distributed data
    sample = np.column_stack((sample, np.array([dist_num]*n))) #Add distribution identifier
    print("Shape of sample = " + str(sample.shape))
    return(sample)

cov = np.array([[1.0, 0.4], [0.4, 1.0]])
mean = np.array([0.0, 0.0])
sample1 = make_dist(mean, cov, 1)

#%%
cov = np.array([[1.0, 0.8], [0.8, 1.0]])
mean = np.array([3.0, 0.0])
sample2 = make_dist(mean, cov, 2, 100, 3344)

mean = np.array([-3.0, 0.0])
cov = np.array([[1.0, 0.8], [0.8, 1.0]])
sample3 = make_dist(mean, cov, 3, 100, 5566)

#%%
def plot_dat(sample1, sample2, sample3):
    plt.scatter(sample1[:,0], sample1[:,1], color = 'blue')
    plt.scatter(sample2[:,0], sample2[:,1], color = 'orange')
    plt.scatter(sample3[:,0], sample3[:,1], color = 'green')
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Sample Data")
    plt.show()
    
plot_dat(sample1, sample2, sample3)

#%%
sample = np.concatenate((sample1, sample2, sample3))

for i in range(1):
    mean_col = np.mean(sample[:,i])
    std_col = np.std(sample[:,i])
    sample[:,i] = [(x - mean_col)/std_col for x in sample[:,i]]
    
sample.shape

#%%
# Creating the K-Means clustering model
kmeans_3 = KMeans(n_clusters=3, random_state=0)
assignments_km3 = kmeans_3.fit_predict(sample[:,0:2])
print(assignments_km3)

#%%
def plot_clusters(sample, assignment):
    col_dic = {0:'blue', 1:'green', 2:'orange', 3:'gray', 4:'magenta', 5:'black'}
    colors = [col_dic[x] for x in assignment]
    plt.scatter(sample[:,0], sample[:,1], color = colors)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Sample Data")
    plt.show()

plot_clusters(sample, assignments_km3)

#%%
# Creating the Hierarchical Agglomerative Clustering model
agglomerative_3 = AgglomerativeClustering(n_clusters=3)
assignments_ag3 = agglomerative_3.fit_predict(sample[:,0:2])

plot_clusters(sample, assignments_ag3)

#%%
# KMeans with 4 clusters
kmeans_4 = KMeans(n_clusters=4, random_state=0)
assignments_km4 = kmeans_4.fit_predict(sample[:,0:2])
plot_clusters(sample, assignments_km4)

#%%
#Hierarchical Agglomerative Clustering with 4 clusters
agglomerative_4 = AgglomerativeClustering(n_clusters=4)
assignments_ag4 = agglomerative_4.fit_predict(sample[:,0:2])
plot_clusters(sample, assignments_ag4)

#%%
# KMeans with 4 clusters
kmeans_5 = KMeans(n_clusters=5, random_state=0)
assignments_km5 = kmeans_5.fit_predict(sample[:,0:2])
plot_clusters(sample, assignments_km5)

#%%
#Hierarchical Agglomerative Clustering with 4 clusters
agglomerative_5 = AgglomerativeClustering(n_clusters=5)
assignments_ag5 = agglomerative_5.fit_predict(sample[:,0:2])
plot_clusters(sample, assignments_ag5)

#%%
km_models = [kmeans_3, kmeans_4, kmeans_5]

def plot_WCSS_km(km_models, samples):
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    
    #Plot WCSS
    wcss = [mod.inertia_ for mod in km_models]
    print(wcss)
    n_clusts = [x+1 for x in range(2, len(wcss) + 2)]
    ax[0].bar(n_clusts, wcss)
    ax[0].set_xlabel("Number of Clusters")
    ax[0].set_ylabel("WCSS")
    
    #Plot BCSS
    tss = np.sum(sample[:,0:1]**2, axis = 0)
    print(tss)
    #Compute BCSS as TSS - WCSS
    bcss = np.concatenate([tss - x for x in wcss]).ravel()
    ax[1].bar(n_clusts, bcss)
    ax[1].set_xlabel("Number of Clusters")
    ax[1].set_ylabel("BCSS")
    plt.show()

plot_WCSS_km(km_models, sample)

#%%
assignment_list = [assignments_km3, assignments_km4, assignments_km5]

def plot_silhouette(samples, assignments, x_lab = 'Number of Clusters', start = 3):
    silhouette = [silhouette_score(samples[:,0:1], a) for a in assignments]
    n_clusts = [x + start for x in range(0, len(silhouette))]
    plt.bar(n_clusts, silhouette)
    plt.xlabel(x_lab)
    plt.ylabel("SC")
    plt.show()

plot_silhouette(sample, assignment_list)

#%%
assignment_list = [assignments_ag3, assignments_ag4, assignments_ag5]
plot_silhouette(sample, assignment_list)


#%%
#Another Example
nr.seed(3344)
cov = np.array([[1.0, -0.98], [-0.98, 1.0]])
mean = np.array([-1.0, 0.0])
sample1 = make_dist(mean, cov, 1, 100, 3344)

nr.seed(5566)
cov = np.array([[1.0, -0.8], [-0.8, 1.0]])
mean = np.array([6.0, 0.0])
sample2 = make_dist(mean, cov, 1, 100, 6677)

nr.seed(7777)
cov = np.array([[1.0, 0.9], [0.9, 1.0]])
mean = np.array([-4.0, 0.0])
sample3= make_dist(mean, cov, 3, 100, 367)

## Plot the distributions
plot_dat(sample1, sample2, sample3)

#%%
sample_2 = np.concatenate((sample1, sample2, sample3))

for i in range(1):
    mean_col = np.mean(sample_2[:,i])
    std_col = np.std(sample_2[:,i])
    sample_2[:,i] = [(x - mean_col)/std_col for x in sample_2[:,i]]

sample_2.shape

#%%
#3 Cluster KMeans
nr.seed(3344)
kmeans_3 = KMeans(n_clusters=3, random_state=0)
assignments_km3 = kmeans_3.fit_predict(sample_2[:,0:2])
plot_clusters(sample_2, assignments_km3)
print(silhouette_score(sample_2[:,0:1], assignments_km3))

#%%
#3 Cluster Agglomerative
nr.seed(3344)
agglomerative3_w = AgglomerativeClustering(n_clusters=3)
assignments3_w = agglomerative3_w.fit_predict(sample_2[:,0:2])
plot_clusters(sample_2, assignments3_w)

#%%
#Using average linkage
nr.seed(5555)
agglomerative3_a = AgglomerativeClustering(n_clusters=3, linkage = 'average')
assignments3_a = agglomerative3_a.fit_predict(sample_2[:,0:2])
plot_clusters(sample_2, assignments3_a)

#%%
#Using complete/maximal linkage
nr.seed(987)
agglomerative3_c = AgglomerativeClustering(n_clusters=3, linkage = 'complete')
assignments3_c = agglomerative3_c.fit_predict(sample_2[:,0:2])
plot_clusters(sample_2, assignments3_c)

#%%
#Using average linkage and Manhattan/l1 distance metric
nr.seed(3344)
agglomerative3_a_m = AgglomerativeClustering(n_clusters=3, linkage = 'average', affinity = 'manhattan')
assignments3_a_m = agglomerative3_a_m.fit_predict(sample_2[:,0:2])
plot_clusters(sample_2, assignments3_a_m)

#%%
#Using average linkage and cosine similarity
nr.seed(234)
agglomerative3_a_c = AgglomerativeClustering(n_clusters=3, linkage = 'average', affinity = 'cosine')
assignments3_a_c = agglomerative3_a_c.fit_predict(sample_2[:,0:2])
plot_clusters(sample_2, assignments3_a_c)

#%%
#Computing the silhouette coefficient for each model
assignment_list = [assignments3_w, assignments3_a, assignments3_c, assignments3_a_m, assignments3_a_c]
plot_silhouette(sample, assignment_list, x_lab = 'Model number', start = 1)