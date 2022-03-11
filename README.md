# RandLAPACK

RandLAPACK will provide high-performance randomized algorithms for linear algebra problems such as least squares, ridge regression, low-rank approximation, and certain matrix factorizations.
RandLAPACK will surely undergo several major revisions before its API stabilizes.
Therefore we insist that no one use it as a dependency in a larger project for the time being.

Refer to ``INSTALL.md`` for directions on how to install RandLAPACK's dependencies,
install RandLAPACK itself, and use RandLAPACK in other projects.

## Related libraries

RandLAPACK depends on [RandBLAS](https://github.com/BallisticLA/proto_rblas), which we are also
developing. 

We've implemented most of RandLAPACK's planned algorithms in Matlab ([MARLA](https://github.com/BallisticLA/marla)) and Python ([PARLA](https://github.com/BallisticLA/parla))
PARLA takes an approach where *algorithms are objects.*
An algorithm needs to be instantiated with its tuning parameters and implementations 
of appropriate subroutines in order to be used.
RandLAPACK's main API will take a similar approach.


## Setting up this repo on your machine

Run ``git clone https://github.com/BallisticLA/proto_RandLAPACK --recursive``.
That will make sure you get the ``proto_RandBLAS`` repository downloaded as well.
