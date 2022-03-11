# RandLAPACK

RandLAPACK will provide high-performance randomized algorithms for linear algebra problems such as least squares, ridge regression, low-rank approximation, and certain matrix factorizations.
RandLAPACK will surely undergo several major revisions before its API stabilizes.
Therefore we insist that no one use it as a dependency in a larger project for the time being.

## Related libraries

RandLAPACK depends on [RandBLAS](https://github.com/BallisticLA/proto_rblas), which we are also
developing. 

We've implemented most of RandLAPACK's planned algorithms in Matlab ([MARLA](https://github.com/BallisticLA/marla)) and Python ([PARLA](https://github.com/BallisticLA/parla))
PARLA takes an approach where *algorithms are objects.*
An algorithm needs to be instantiated with its tuning parameters and implementations 
of appropriate subroutines in order to be used.
RandLAPACK's main API will take a similar approach.


## Notes for collaborators

Refer to ``INSTALL.md`` for directions on how to install RandLAPACK's dependencies,
install RandLAPACK itself, and use RandLAPACK in other projects.

We'll need to conduct experiments for proofs-of-concept and benchmarking while developing RandLAPACK. Those experiments should be kept under version control. If you want to make such an experiment, create a branch like
```
git checkout -b experiments/riley-svdidea-220311
```
The branch name should always have the prefix "experiments/[your name]". The example name above includes keywords on the nature of the branch and date in YYMMDD format. If you want to share that example with others then you can push the branch to the BallisticLA/RandLAPACK repository.

If you get to the point of a clean example which you would like to refer to in the past, we recommend that you use [git tags](https://en.wikibooks.org/wiki/Git/Advanced#Tags) for important commits on your experiment branch.
