The Vinson & Vigliocco norms are made freely available [alongside their publication][1].  However, the format is that of a sparse integer matrix split over two text files, with separate text files labelling the row and column indices of that matrix.

This program lets you query that matrix, and can transform it into human-readable text-files.

To use, [download the data][1], and unzip it into a location.

## Usage

Invoke like this:

```commandline
python human_readable_vv.py --help
```

Available operations:

__`python human_readable_vv.py <path-to-vv-files> -w <feature>`__

> Lists all words for a particular feature, ordered by the number of subjects that produced the feature for that word.

__`python human_readable_vv.py <path-to-vv-files> -f <word>`__

> Lists all features for a particular word, ordered by the number of subjects that produced the feature for that word.

__`python human_readable_vv.py <path-to-vv-files> --savewords <path>`__

> Saves a list of words and features in the specified file (which should not exist).  
> Format of the file will be:  
> `word: feature1, feature2, ...`  
> on each line.

__`python human_readable_vv.py <path-to-vv-files> --savefeatures <path>`__

> Saves a list of features and words in the specified file (which should not exist).  
> Format of the file will be:  
> `feature: word1, word2, ...`  
> on each line.


## Requirements

- Requires python 3.6+.
- Requires numpy.


[1]: https://link.springer.com/article/10.3758/BRM.40.1.183 (Possibly behind a paywall.)
