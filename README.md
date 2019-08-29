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

- [parsing]
Under development.
