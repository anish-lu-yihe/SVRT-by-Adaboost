Here I attempt to solve the Synthesis Visual Reasoning Test by Adaboost.

## Synthesis Visual Reasoning Test (SVRT)
The SVRT was firstly published in [Comparing machines and humans on a visual categorization test](https://www.pnas.org/content/108/43/17621.short). It has been shown to be a task, relatively easy for human subjects, but challenging for machine agents.

The original code for the SVRT generator can be found [here](https://www.idiap.ch/~fleuret/svrt/). My colleague [Scott](https://github.com/scottclowe) has been developing a more complicated version of the generator (which is in his private repository; please contact him if you are interested). Based on one of his new versions, I have made some modifications (which I would probably import to my own repository at some time point in the future).

## Adaboost
Adaboost (short for Adaptive Boosting) is an ensemble method developed by Y. Freund, and R. Schapire (JCSS, 1997). The idea is to fit a sequence of weak classifiers and to change their weights contributing to the entire classifier iteratively, so that the entire classifier modifies its predication according to what it mistakes gradually.

### Requirements
- Python 2.7
- [scikit-learn](https://scikit-learn.org/dev/index.html#) 0.21.3 (and all its dependence)

### Branches
- **master**:
Following the literature, Adaboost is used here as a baseline. For fair comparison with program synthesis, the inputs are parsings and the outputs are class indices. A parsing from pySVRT/parsed_classic is encoded in the following form:

> Shape(x-coordinate, y-coordinate, shape identity, scale), Shape(x-coordinate, y-coordinate, shape identity, scale)[, ...]\
[borders(shape index, shape index)]\
[...]\
[contains(shape index, shape index)]\
[...]

where those within the brackets [] can be omitted dependent on the image. To encode the bordering and containing data, a bordering and a containing matrices are formed: if borders(shape a, shape b), then the (a,b) and (b,a) entries of the bordering matrix are set to be 100; if contains(shape a, shape b), then the (a,b) entry of the containing matrix equals 100 while the (b,a) entry equals -100. Each row of the two matrices is then inserted to the individual shape information encoded by the 4 parameters. An image parsing vector is obtained by simply flattening these individual shape parameters into a real vector, whose dimension is always (4 + #shape * 2) * #shape.

- [parsing](https://github.com/anish-lu-yihe/SVRT-by-Adaboost/tree/parsing):
I attempt to vary the parsing dimension to avoid redundant zeros in inputs. However, Adaboost cannot take inputs (parsings of different images in one problem) of varying input dimesions, which caused the code to fail in problem \#4, \#7, \#8, \#11, \#12, \#13, \#19, \#21 and \#23. For other problems, the results are identical to what obtained by the code in master branch (with redundant zeros for containing and bordering information in inputs). Therefore, the code in this parsing branch is totally useless.

### Usage
1. Generate SVRT problems (not included in this project).
2. Run main.py.

## Results
I ran Adaboost with 10, 100 and 1000 stumps on 1k, 2k and 9k traning examples, and tested the models with 1k unseen examples.

| \#train    | 1000     |          |          |          | 2000     |          |          | 9000     |          |          |          |
|------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| \#stumps   | 10       | 100      | 1000     | 10000    | 10       | 100      | 1000     | 10       | 100      | 1000     | 10000    |
| 1          | 48\.80%  | 50\.10%  | 50\.30%  | 51\.20%  | 50\.80%  | 49\.30%  | 49\.30%  | 50\.40%  | 48\.20%  | 48\.30%  | 50\.20%  |
| 2          | 64\.30%  | 66\.20%  | 62\.40%  | 59\.90%  | 63\.10%  | 63\.30%  | 61\.50%  | 62\.40%  | 63\.40%  | 62\.80%  | 59\.60%  |
| 3          | 50\.70%  | 50\.10%  | 51\.20%  | 51\.80%  | 50\.10%  | 49\.50%  | 52\.40%  | 52\.80%  | 53\.00%  | 51\.40%  | 52\.40%  |
| 4          | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% |
| 5          | 65\.00%  | 61\.40%  | 60\.30%  | 57\.10%  | 65\.50%  | 62\.90%  | 62\.10%  | 65\.90%  | 66\.20%  | 65\.50%  | 64\.00%  |
| 6          | 63\.90%  | 62\.30%  | 61\.70%  | 60\.40%  | 65\.10%  | 64\.50%  | 61\.60%  | 65\.50%  | 67\.20%  | 64\.90%  | 63\.60%  |
| 7          | 59\.10%  | 52\.70%  | 51\.60%  | 51\.90%  | 59\.10%  | 56\.70%  | 55\.60%  | 59\.30%  | 59\.20%  | 58\.70%  | 55\.40%  |
| 8          | 90\.80%  | 90\.50%  | 88\.00%  | 84\.30%  | 90\.80%  | 90\.80%  | 89\.50%  | 90\.80%  | 90\.80%  | 90\.80%  | 90\.60%  |
| 9          | 53\.20%  | 50\.60%  | 49\.00%  | 48\.50%  | 52\.80%  | 52\.90%  | 50\.80%  | 56\.60%  | 56\.00%  | 57\.10%  | 56\.20%  |
| 10         | 61\.20%  | 59\.60%  | 59\.70%  | 58\.40%  | 65\.40%  | 61\.10%  | 60\.50%  | 60\.40%  | 63\.10%  | 61\.50%  | 60\.90%  |
| 11         | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% |
| 12         | 52\.30%  | 51\.40%  | 50\.10%  | 50\.70%  | 53\.70%  | 54\.90%  | 53\.90%  | 51\.10%  | 51\.60%  | 53\.20%  | 53\.90%  |
| 13         | 52\.30%  | 50\.00%  | 50\.50%  | 49\.10%  | 50\.00%  | 49\.70%  | 49\.40%  | 50\.70%  | 49\.80%  | 49\.40%  | 49\.30%  |
| 14         | 49\.40%  | 48\.90%  | 49\.90%  | 51\.20%  | 51\.90%  | 49\.00%  | 49\.70%  | 50\.50%  | 51\.10%  | 52\.20%  | 50\.80%  |
| 15         | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% |
| 16         | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% |
| 17         | 56\.10%  | 54\.00%  | 53\.90%  | 53\.10%  | 57\.30%  | 54\.20%  | 54\.40%  | 58\.50%  | 58\.00%  | 58\.20%  | 56\.10%  |
| 18         | 58\.40%  | 62\.40%  | 60\.30%  | 59\.60%  | 59\.90%  | 61\.80%  | 60\.90%  | 60\.00%  | 64\.10%  | 63\.60%  | 61\.90%  |
| 19         | 48\.90%  | 49\.10%  | 48\.10%  | 47\.60%  | 46\.60%  | 48\.00%  | 48\.90%  | 48\.90%  | 49\.90%  | 51\.60%  | 51\.00%  |
| 20         | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% |
| 21         | 50\.30%  | 49\.80%  | 49\.60%  | 47\.40%  | 50\.40%  | 48\.70%  | 49\.80%  | 50\.00%  | 49\.10%  | 47\.00%  | 47\.90%  |
| 22         | 49\.70%  | 50\.80%  | 52\.50%  | 51\.30%  | 50\.30%  | 51\.10%  | 51\.40%  | 49\.10%  | 50\.50%  | 48\.00%  | 48\.10%  |
| 23         | 67\.40%  | 64\.20%  | 58\.20%  | 54\.70%  | 73\.60%  | 68\.70%  | 62\.10%  | 78\.20%  | 77\.20%  | 71\.60%  | 65\.70%  |

It is very clear that Adaboost can perform classification pretty well in problem 4, 8, 11, 15, 16 and 20, but poorly in other problems.
