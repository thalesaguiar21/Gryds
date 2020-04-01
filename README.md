# Gryds
This is a simple interface to use as tunning mathematical model. It splits the 
data using a Stratified K-Fold and saves all useful information during tests.

> This is a very simple project, still has no parallel processing so it may be a
bit slow.

## How to use

The core of the project resides in the **gsearch** module. This code snippet
will create an instance of a grid search that will split the data into **3**
parts, using the given cross validator, and save results into.

```python
from gryds import gsearch
model = sklearn.cluster.KMeans(n_cluster=2)
crossval = sklearn.model_selection.StratifiedKFold(3)
X, Y = sklearn.datasets.make_blobs()
gsearch.tune(model, crossval, X, Y, n_clusters=[2,4],
             max_iter=[100, 200])
```

Notice that, the given model already satisfies the *GrydModel* interface, which
is under the **gryds/base.py** module.

With that, you will have acces to the mean score of each configuration, and
the mapping of prediction, sample index, and expected output. Besides, the
module also saves time elapsed for both training and testing. By default, files
are saved in _tests_ directory.

You can change the directory to save files by change the module configurations
```python
import gryds

gryds.confs.paths['savedir'] = 'path/to/save/results'
```

It is also possible to configure metrics used to save the file. For instance,
saving the elapsed times in nanoseconds
```python
import gryds

gryds.confs.metrics['timeunit'] = 'nano'
```

## Testing

You can easily test the module by typing
```bash
$ make test
```

inside the project directory. Also, you can clear the project after testing
using
```bash
$ make test-no-out
$ make clear
```

Finally, you can also test modules independently
```bash
$ make test-no-out module=path/to/module_to_test.py
$ make clear
```

