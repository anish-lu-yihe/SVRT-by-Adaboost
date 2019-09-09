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
> Shape(x-coordinate, y-coordinate, shape identity, scale), Shape(x-coordinate, y-coordinate, shape identity, scale)[, ...]

> [borders(shape index, shape index)]

> [...]

> [contains(shape index, shape index)]

> [...]

where those within the brackets [] can be omitted dependent on the image. To encode the bordering and containing data, a bordering and a containing matrices are formed: if borders(shape a, shape b), then the (a,b) and (b,a) entries of the bordering matrix are set to be 100; if contains(shape a, shape b), then the (a,b) entry of the containing matrix equals 100 while the (b,a) entry equals -100. Each row of the two matrices is then inserted to the individual shape information encoded by the 4 parameters. An image parsing vector is obtained by simply flattening these individual shape parameters into a real vector, whose dimension is always (4 + #shape * 2) * #shape.

- **parsing**:
Under development. I attempt to vary the parsing dimension to avoid to redundant zeros.

### Usage
1. Generate SVRT problems (not included in this project).
2. Run main.py.

## Results by parsing
I ran Adaboost on this varying-dimensional parsings with 10, 100, 1000 and 10000 stumps on 9k examples, and tested them with 1k unseen examples. The code actually failed to work for 9 out of 23 problems due to the varying dimension, which caused the input dimension into Adaboost not consistent. For other problems, the results are identical to what obtained by non-varying-dimensional parsings.
