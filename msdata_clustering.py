import pandas as pd
import numpy as np
import statistics
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import Normalizer
import sklearn.metrics as m
import matplotlib.pyplot as plt


def goodness_of_model(class_labels, labels, x):
    score=m.normalized_mutual_info_score(class_labels, labels, average_method="geometric")
    print(f"NMI, when K/d/linkage={x}: {score}")
    return score

def make_kcluster(df,n_clusters):
    model=KMeans(n_clusters)
    model.fit(df)
    return model

def make_hcluster(df, lin):
    Z=linkage(df,lin, optimal_ordering=False)
    return Z


def run_pca(df,n):
    pca=PCA(n)
    principles=pca.fit_transform(df)
    principles=pd.DataFrame(principles)
    return principles

def read_data():
    '''
    Reads the data and removes the class labels from the dataframe. Class labels are returned as
    an separate array. Data is also normalized.
    '''

    df=pd.read_csv('msdata2.csv', index_col=0)
    labels=df["class"]
    df=df.drop(["class"], axis=1)
    df=Normalizer().transform(df)
    return df, labels

def save_labels(labels, file_path):
    with open(file_path, 'w') as f:
        for l in labels:
            f.write("%s\n" % int(l))
            

def main():
    df, labels=read_data()
    
    #change to True to run the section
    test_clusterings=False
    plot_data=False
    test_dimensions=False
    run_clustering=True


    if test_clusterings:
        #testing clustering methods, before data processing
        print("testing K-meaans with different Ks")
        for k in range(2,10,2):
            model=make_kcluster(df,k)
            goodness_of_model(labels,model.labels_, k)
            
        models2={}
        models2["single"]=make_hcluster(df, "single")
        models2["complete"]=make_hcluster(df, "complete")
        models2["average"]=make_hcluster(df, "average")
        models2["distance of centroid"]=make_hcluster(df, "centroid")
        print("testing hierarchial clustering with different linkages")
        for m in models2:
            model_labels=fcluster(models2[m], 3, criterion="maxclust")
            goodness_of_model(labels,model_labels, m)
            

    if plot_data:
        #making scatter plot of the data
        components=run_pca(df,3)
        fig=plt.figure()
        axs=fig.add_subplot(111, projection='3d')
        axs.set_title(f'Scatter plot of MS-data')
        axs.scatter(components[[0]], components[[1]], components[[2]])
        plt.show()
    

    k=3#number of clusters chosed based on the above figure
    if test_dimensions:
        #testing different dimension reductions
        scores2={}
        models={}
        print("testing k-means with different number of dimensions")
        for i in range(10,700,10):
            df2=run_pca(df,i)
            components=df2
            model=make_kcluster(df2,k)
            score=goodness_of_model(labels, model.labels_, i)
            scores2[i]=score
            models[i]=model
        
        fig=plt.figure()
        axs=fig.add_subplot(111)
        axs.set_title(f'NMI:s with different dimension reductions')
        axs.scatter(scores2.keys(), scores2.values())
        plt.show()

    if run_clustering:
        #reducing dimensions to 50 and running k-means 20 times, then best result is saves
        scores2={}
        models={}
        df2=run_pca(df,50)
        print("Running the final algorithm with K=3 and d=50")
        for i in range(20):
            components=df2
            model=make_kcluster(df2,k)
            score=goodness_of_model(labels, model.labels_, 3)
            scores2[i]=score
            models[i]=model

        d=max(scores2, key=lambda key:scores2[key])
        fig=plt.figure()
        axs=fig.add_subplot(111, projection='3d')
        axs.set_title(f'K={k}, dimensions={50}')
        axs.scatter(components[[0]], components[[1]], components[[2]], c=models[d].labels_)
        plt.show()
        print("Best NMI-score:")
        goodness_of_model(labels, models[d].labels_,3)
        save_labels(models[d].labels_, 'ms_labels.txt')
        print("Average of NMI-scores: ", statistics.mean(scores2.values()))
   
 
if __name__ == "__main__":
    main()