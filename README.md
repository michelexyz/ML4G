# RCIT
This is an R package implementing the Randomized Conditional Independence Test (RCIT) and the Randomized conditional Correlation Test (RCoT).

The package depends on the MASS and momentchi2 packages on CRAN, so please install these first.

# Installation
> library(devtools)

> install_github("ericstrobl/RCIT")

> library(RCIT)

Loading required package: momentchi2

Loading required package: MASS

> RCIT(rnorm(1000),rnorm(1000),rnorm(1000))

