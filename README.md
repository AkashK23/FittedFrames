# FittedFrames
adding fitted frames to s-reps

fit2D.py:

This file fits a 2D s-rep to a flat 2D object.
Input: set of discrete points on the boundary of the 2D objects
Ouput: skeletal points and boundary points of the fitted 2D s-rep

curvedSrep.py

This file maps the skeletal points of the inputted ellipsoids skeleton onto the XY plane.
Then a 2D s-rep is fit to the translated/rotated set of points
Then, the 2D s-rep is mapped back to the space of their original locations
input: skeletal points of ellipsoid s-rep
output: s-rep of the ellipsoid's s-rep

finalFitted.py

This file fits the fitted frames to the best fitted ellipsoid's skeleton.
Then based on the points of the ellipsoid and the initial target object, it interpolates the fitted frames of the s-rep of the target object.
Input: mesh of target object, mesh of ellipsoid, s-rep of target object, s-rep of ellipsoid
output: fitted frames for the target object ellipsoid
