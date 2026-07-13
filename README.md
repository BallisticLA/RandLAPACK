# RandLAPACK

RandLAPACK provides high-performance randomized algorithms for linear algebra problems such as least squares, (kernel) ridge regression, low-rank approximation, and matrix factorizations.
RandLAPACK's API is not yet stable. We're interested in changing that, but commitment is scary.

Please swing by [**our Discord server**](https://discord.gg/R4qj8Er9YW) if you have questions about RandLAPACK or would like to get involved in its development.

## Related libraries

RandLAPACK depends on [RandBLAS](https://github.com/BallisticLA/RandBLAS), which we are also
developing. 

Before starting on RandLAPACK we implemented several high-level RandNLA algorithms in Matlab ([MARLA](https://github.com/BallisticLA/marla)) and Python ([PARLA](https://github.com/BallisticLA/parla)).
In the latter library we took an approach where *algorithms are objects.*
An algorithm needs to be instantiated with its tuning parameters and subroutines in order to be used.
RandLAPACK currently emphasizes that "algorithms as objects" approach.

## Notes for collaborators

Refer to ``INSTALL.md`` for directions on how to install RandLAPACK's dependencies,
install RandLAPACK itself, and use RandLAPACK in other projects.
