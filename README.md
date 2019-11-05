# Gryds
This is a simple interface to use as tunning mathematical model. It splits the 
data using a Stratified K-Fold and saves all useful information during tests.

## How to use 

The core of the project resides in the gridsearch module.

    ```python
	import gryds.gridsearch as gs

	path = 'path/to/store/results/'
	my_gs = gs.GS(nfolds=3, path)
    ```
This code snippet will create an instance of a grid search that will split the
data into **3** parts and save results to **path**.

    ```python
	from sklearn.cluster import KMeans
	from sklearn.datasets import make_blobs

	X, Y = make_blobs(n_samples=50, centers=2)
	model = KMeans()
	gs.tune(model, X, Y, n_clusters=[2, 4], max_iter=[100, 200]
    ```

Adding the code above to the first snippet, allows you to fine tune the KMeans
model from SkLearn package. Notice that, the given model already satisfies the
*GrydModel* interface, which is under the **gryds/models.py** module.

With that, you will have acces to the mean score of each configuration, and
the mapping of prediction, sample index, and expected output.

> This is a very simple project, still has no parallel processing so it may be a
bit slow.

