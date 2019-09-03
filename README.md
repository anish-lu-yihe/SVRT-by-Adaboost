Here I attempt to solve the Synthesis Visual Reasoning Test by Adaboost.

## Synthesis Visual Reasoning Test (SVRT)
The SVRT was firstly published in [Comparing machines and humans on a visual categorization test](https://www.pnas.org/content/108/43/17621.short). It has been shown to be a task, relatively easy for human subjects, but challenging for machine agents.

The original code for the SVRT generator can be found [here](https://www.idiap.ch/~fleuret/svrt/). My colleague [Scott](https://github.com/scottclowe) has been developing a more complicated version of the generator (which is in his private repository; please contact him if you are interested). Based on one of his new versions, I have made some modifications (which I would probably import to my own repository at some time point in the future).

## Adaboost


### Requirements
- Python 2.7
- [scikit-learn](https://scikit-learn.org/dev/index.html#) 0.21.3 (and all its dependence)

### Branches
- [master](https://github.com/anish-lu-yihe/SVRT-by-RN):
Currently this branch contains the [Adaboost example](https://scikit-learn.org/dev/modules/ensemble.html#adaboost) for provided by scikit-learn.

- **parsing**:
Following the literature, Adaboost is used here as a baseline. For fair comparison with program synthesis, the inputs are parsings and the outputs are class indices. A parsing from pySVRT/parsed_classic is encoded in the following form:
> Shape(x-coordinate, y-coordinate, shape identity, scale), Shape(x-coordinate, y-coordinate, shape identity, scale)[, ...]
> [borders(shape index, shape index), ...]
> [contains(shape index, shape index), ...]

where those within the brackets [] can be omitted dependent on the image. The parsing is transformed into a real vector by simply taking all the numerical values one by one.

### Usage
1. Generate SVRT problems (not included in this project).
2. Run main.py.

## Results by parsing
My machine is running the classification with 10 stumps.
