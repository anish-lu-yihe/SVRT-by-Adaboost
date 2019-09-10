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
- [master](https://github.com/anish-lu-yihe/SVRT-by-Adaboost):
Following the literature, Adaboost is used here as a baseline. For fair comparison with program synthesis, the inputs are parsings and the outputs are class indices. A parsing from pySVRT/parsed_classic is encoded in the following form:

> Shape(x-coordinate, y-coordinate, shape identity, scale), Shape(x-coordinate, y-coordinate, shape identity, scale)[, ...]\
[borders(shape index, shape index)]\
[...]\
[contains(shape index, shape index)]\
[...]

where those within the brackets [] can be omitted dependent on the image. To encode the bordering and containing data, a bordering and a containing matrices are formed: if borders(shape a, shape b), then the (a,b) and (b,a) entries of the bordering matrix are set to be 100; if contains(shape a, shape b), then the (a,b) entry of the containing matrix equals 100 while the (b,a) entry equals -100. Each row of the two matrices is then inserted to the individual shape information encoded by the 4 parameters. An image parsing vector is obtained by simply flattening these individual shape parameters into a real vector, whose dimension is always (4 + #shape * 2) * #shape.

- [parsing](https://github.com/anish-lu-yihe/SVRT-by-Adaboost/tree/parsing):
I attempt to vary the parsing dimension to avoid redundant zeros in inputs. However, Adaboost cannot take inputs (parsings of different images in one problem) of varying input dimesions, which caused the code to fail in problem \#4, \#7, \#8, \#11, \#12, \#13, \#19, \#21 and \#23. For other problems, the results are identical to what obtained by the code in master branch (with redundant zeros for containing and bordering information in inputs). Therefore, the code in this parsing branch is totally useless.

- **small-sample**
Here I want to see if Adaboost can learn from relatively few examples.

### Usage
1. Generate SVRT problems (not included in this project).
2. Run main.py.

## Results by parsing
I ran Adaboost with 10, 100 and 1000 stumps on 1k, 2k and 9k training examples, and tested the models with 1k unseen examples.

|           | Sasquatch| train 10 | test 40  |           | New pars | train 10 | test 40  |          | Sasquatch| train 40 | test 10  |          | New pars | train 10 | test 40  |          | New pars | train 1000 | test 1000  |          |
|-----------|----------|----------|----------|-----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| stumps    | 10       | 100      | 1000     | 10000     | 10       | 100      | 1000     | 10000    | 10       | 100      | 1000     | 10000    | 10       | 100      | 1000     | 10000    | 10       | 100      | 1000     | 10000    |
| 1         | 58\.75%  | 60\.00%  | 58\.75%  | 57\.50%   | 43\.75%  | 48\.75%  | 45\.00%  | 43\.75%  | 45\.00%  | 20\.00%  | 25\.00%  | 35\.00%  | 70\.00%  | 50\.00%  | 45\.00%  | 40\.00%  | 48\.80%  | 50\.10%  | 50\.30%  | 51\.20%  |
| 2         | 52\.50%  | 46\.25%  | 51\.25%  | 51\.25%   | 50\.00%  | 53\.75%  | 53\.75%  | 57\.50%  | 55\.00%  | 50\.00%  | 60\.00%  | 60\.00%  | 70\.00%  | 60\.00%  | 45\.00%  | 50\.00%  | 64\.30%  | 66\.20%  | 62\.40%  | 59\.90%  |
| 3         | 85\.00%  | 70\.00%  | 68\.75%  | 68\.75%   | 51\.25%  | 50\.00%  | 51\.25%  | 50\.00%  | 85\.00%  | 75\.00%  | 85\.00%  | 80\.00%  | 60\.00%  | 55\.00%  | 55\.00%  | 55\.00%  | 50\.70%  | 50\.10%  | 51\.20%  | 51\.80%  |
| 4         | 100\.00% | 100\.00% | 100\.00% | 100\.00%  | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% |
| 5         | 55\.00%  | 53\.75%  | 51\.25%  | 51\.25%   | 62\.50%  | 58\.75%  | 58\.75%  | 58\.75%  | 45\.00%  | 45\.00%  | 55\.00%  | 55\.00%  | 80\.00%  | 80\.00%  | 70\.00%  | 70\.00%  | 65\.00%  | 61\.40%  | 60\.30%  | 57\.10%  |
| 6         | 57\.50%  | 63\.75%  | 65\.00%  | 65\.00%   | 52\.50%  | 53\.75%  | 57\.50%  | 57\.50%  | 65\.00%  | 50\.00%  | 50\.00%  | 50\.00%  | 60\.00%  | 55\.00%  | 55\.00%  | 40\.00%  | 63\.90%  | 62\.30%  | 61\.70%  | 60\.40%  |
| 7         | 52\.50%  | 48\.75%  | 47\.50%  | 47\.50%   | 48\.75%  | 55\.00%  | 53\.75%  | 53\.75%  | 55\.00%  | 35\.00%  | 25\.00%  | 35\.00%  | 50\.00%  | 60\.00%  | 50\.00%  | 55\.00%  | 59\.10%  | 52\.70%  | 51\.60%  | 51\.90%  |
| 8         | 88\.75%  | 78\.75%  | 80\.00%  | 80\.00%   | 81\.25%  | 78\.75%  | 78\.75%  | 78\.75%  | 100\.00% | 95\.00%  | 95\.00%  | 90\.00%  | 100\.00% | 85\.00%  | 90\.00%  | 90\.00%  | 90\.80%  | 90\.50%  | 88\.00%  | 84\.30%  |
| 9         | 52\.50%  | 56\.25%  | 50\.00%  | 51\.25%   | 50\.00%  | 48\.75%  | 50\.00%  | 55\.00%  | 55\.00%  | 65\.00%  | 60\.00%  | 60\.00%  | 50\.00%  | 45\.00%  | 45\.00%  | 45\.00%  | 53\.20%  | 50\.60%  | 49\.00%  | 48\.50%  |
| 10        | 60\.00%  | 58\.75%  | 56\.25%  | 53\.75%   | 57\.50%  | 52\.50%  | 55\.00%  | 56\.25%  | 85\.00%  | 75\.00%  | 75\.00%  | 75\.00%  | 60\.00%  | 45\.00%  | 55\.00%  | 65\.00%  | 61\.20%  | 59\.60%  | 59\.70%  | 58\.40%  |
| 11        | 100\.00% | 100\.00% | 100\.00% | 100\.00%  | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% |
| 12        | 47\.50%  | 46\.25%  | 48\.75%  | 48\.75%   | 51\.25%  | 48\.75%  | 51\.25%  | 52\.50%  | 45\.00%  | 65\.00%  | 65\.00%  | 70\.00%  | 45\.00%  | 45\.00%  | 40\.00%  | 40\.00%  | 52\.30%  | 51\.40%  | 50\.10%  | 50\.70%  |
| 13        | 53\.75%  | 51\.25%  | 55\.00%  | 57\.50%   | 52\.50%  | 58\.75%  | 55\.00%  | 55\.00%  | 60\.00%  | 65\.00%  | 60\.00%  | 55\.00%  | 35\.00%  | 50\.00%  | 50\.00%  | 50\.00%  | 52\.30%  | 50\.00%  | 50\.50%  | 49\.10%  |
| 14        | 67\.50%  | 65\.00%  | 67\.50%  | 66\.25%   | 58\.75%  | 58\.75%  | 62\.50%  | 62\.50%  | 70\.00%  | 80\.00%  | 85\.00%  | 85\.00%  | 65\.00%  | 45\.00%  | 50\.00%  | 40\.00%  | 49\.40%  | 48\.90%  | 49\.90%  | 51\.20%  |
| 15        | 87\.50%  | 85\.00%  | 82\.50%  | 76\.25%   | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 90\.00%  | 90\.00%  | 90\.00%  | 90\.00%  | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% |
| 16        | 95\.00%  | 95\.00%  | 95\.00%  | 95\.00%   | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 95\.00%  | 95\.00%  | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% |
| 17        | 45\.00%  | 47\.50%  | 50\.00%  | 48\.75%   | 57\.50%  | 56\.25%  | 48\.75%  | 48\.75%  | 50\.00%  | 50\.00%  | 45\.00%  | 45\.00%  | 40\.00%  | 45\.00%  | 45\.00%  | 45\.00%  | 56\.10%  | 54\.00%  | 53\.90%  | 53\.10%  |
| 18        | 67\.50%  | 62\.50%  | 65\.00%  | 65\.00%   | 67\.50%  | 65\.00%  | 66\.25%  | 66\.25%  | 75\.00%  | 70\.00%  | 60\.00%  | 70\.00%  | 70\.00%  | 60\.00%  | 70\.00%  | 70\.00%  | 58\.40%  | 62\.40%  | 60\.30%  | 59\.60%  |
| 19        | 55\.00%  | 53\.75%  | 55\.00%  | 60\.00%   | 47\.50%  | 53\.75%  | 52\.50%  | 51\.25%  | 65\.00%  | 75\.00%  | 65\.00%  | 60\.00%  | 60\.00%  | 55\.00%  | 70\.00%  | 70\.00%  | 48\.90%  | 49\.10%  | 48\.10%  | 47\.60%  |
| 20        | 50\.00%  | 48\.75%  | 46\.25%  | 50\.00%   | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 55\.00%  | 65\.00%  | 75\.00%  | 65\.00%  | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% | 100\.00% |
| 21        | 52\.50%  | 47\.50%  | 50\.00%  | 52\.50%   | 47\.50%  | 41\.25%  | 38\.75%  | 38\.75%  | 50\.00%  | 30\.00%  | 50\.00%  | 50\.00%  | 65\.00%  | 35\.00%  | 45\.00%  | 40\.00%  | 50\.30%  | 49\.80%  | 49\.60%  | 47\.40%  |
| 22        | 48\.75%  | 47\.50%  | 46\.25%  | 41\.25%   | 52\.50%  | 48\.75%  | 50\.00%  | 53\.75%  | 70\.00%  | 70\.00%  | 50\.00%  | 55\.00%  | 45\.00%  | 40\.00%  | 40\.00%  | 35\.00%  | 49\.70%  | 50\.80%  | 52\.50%  | 51\.30%  |
| 23        | 50\.00%  | 61\.25%  | 57\.50%  | 58\.75%   | 56\.25%  | 53\.75%  | 48\.75%  | 50\.00%  | 85\.00%  | 75\.00%  | 75\.00%  | 75\.00%  | 60\.00%  | 50\.00%  | 50\.00%  | 55\.00%  | 67\.40%  | 64\.20%  | 58\.20%  | 54\.70%  |


It is very clear that Adaboost can perform classification pretty well in problem 4, 8, 11, 15, 16 and 20, but poorly in other problems.
