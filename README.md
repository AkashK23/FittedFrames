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
Input: mesh of target object, mesh of ellipsoid, s-rep of ellipsoid
output: fitted frames for the target object ellipsoid

Overview:

Currently finalFitted.py takes in ellipsoid mesh, s-rep of ellipsoid, and target object mesh.
First the skeleton of the ellipsoid s-rep is found and then curvedSrep script is run. 
curvedSrep maps this ellipsoid skeleton into the XY plan and then calls fit2d.
fit2d then fits a 2D s-rep to the 2D skeleton and returns the skeletal points and boundary points of the 2D s-rep
curvedSrep takes those skeletal points and maps them back onto the original plane of the ellipsoid's skeleton. Then returns those points to finalFitted.
finalFitted then takes the 2D s-rep and uses that to get the theta and tao1 directions to fit the fitted frames on its the ellipsoids skeleton.
The 3D spokes of the ellipsoid's skeleton are used to obtain the tao2 direction to get the fitted frames at the boundary (or along the spokes as well if parameter changes)
Then the fitted frames for the ellipsoid are mapped to the target object using the target object mesh.

Changes to be made:

Incorporate the Mean curvature flow so that you aren't using a single diffeomorphism deformation between ellipsoid and target object.
