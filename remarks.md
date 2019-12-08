# Heisenberg

Only one big sweep
Unit cell = 2  

* D = 14
    * SVD without rotate does not converge (nothing converges)
    * SVD with rotate converges
    * 1-site without rotate converges
    * 1-site with rotate converges (sometimes, when it doesn't its energy doesnt converge either)
* D = 15
    * SVD without rotate does not converge (nothing converges)
    * SVD with rotate converges
    * 1-site without rotate converges
    * 1-site with rotate does not converge (sometimes)
* D = 16
    * SVD without rotate does not converge (only mixed tf converges)
    * SVD with rotate converges
    * 1-site without rotate converges (faster than without)
    * 1-site with rotate converges
