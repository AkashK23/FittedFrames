import vtk
import math
import numpy as np
import sys
import os
# from curvedSrep import curvedSrep as cs
sys.path.append(os.path.abspath("."))
from curvedSrep import curvedSrep
import matplotlib.pyplot as plt
from numpy import random
from scipy.spatial import distance

plt.figure()
ax = plt.subplot(111, projection='3d')

# s-rep of ellipse
reader = vtk.vtkPolyDataReader()
reader.SetFileName('sreps/data/reSrep.vtk')
reader.Update()
bot_srep = reader.GetOutput()

#s-rep of target object
reader2 = vtk.vtkPolyDataReader()
reader2.SetFileName('sreps/control/top_srep_twist1.vtk')
reader2.Update()
top_srep = reader2.GetOutput()

# get all the skeletal and boundary points of the ellipsoid s-rep
source_pts = vtk.vtkPoints()
for i in range(bot_srep.GetNumberOfCells()):
    base_pt_id = i * 2
    bdry_pt_id = i * 2 + 1
    s_pt = bot_srep.GetPoint(base_pt_id)
    b_pt = bot_srep.GetPoint(bdry_pt_id)
    # ax.scatter(s_pt[0], s_pt[1], s_pt[2], color='k')
    source_pts.InsertNextPoint([s_pt[0], s_pt[1], s_pt[2]])
source_pts.Modified()

# getting 2D s-rep of the ellipsoid skeleton
rXs, rYs, rZs, rSamPts = curvedSrep(source_pts)
numSamp = len(rSamPts[0])
print(numSamp)

iXs = []
iYs = []
iZs = []
iSamPts = []
srepPts = vtk.vtkPoints()
pPts = vtk.vtkPoints()

# viewing the 2D skeleton
ptsOnSkel = []
for i in range(0, len(rXs)):
    intPt = (rXs[i], rYs[i], rZs[i])
    ax.scatter(rXs[i], rYs[i], rZs[i], color='b')
    iXs.append(intPt[0])
    iYs.append(intPt[1])
    iZs.append(intPt[2])

    if i == 0:
        iSpoke = []
        srepPts.InsertNextPoint([iXs[i], iYs[i], iZs[i]])
        ptsOnSkel.append([iXs[i], iYs[i], iZs[i]])
        for j in range(0, numSamp):
            intPt = rSamPts[i][j]
            iSpoke.append([intPt[0], intPt[1], intPt[2]])
            srepPts.InsertNextPoint([intPt[0], intPt[1], intPt[2]])
            ptsOnSkel.append([intPt[0], intPt[1], intPt[2]])
            ax.scatter(intPt[0], intPt[1], intPt[2], color='b')
            if j == 0:
                ax.plot([iXs[i], iSpoke[0][0]], [iYs[i], iSpoke[0][1]], [iZs[i], iSpoke[0][2]], 'r')
            else:
                ax.plot([iSpoke[j-1][0], iSpoke[j][0]], [iSpoke[j-1][1], iSpoke[j][1]], [iSpoke[j-1][2], iSpoke[j][2]], 'r')
        iSamPts.append(iSpoke)
    elif i == len(rXs)-1:
        iSpoke = []
        srepPts.InsertNextPoint([iXs[i], iYs[i], iZs[i]])
        ptsOnSkel.append([iXs[i], iYs[i], iZs[i]])
        for j in range(0, numSamp):
            intPt = rSamPts[-1][j]
            iSpoke.append([intPt[0], intPt[1], intPt[2]])
            srepPts.InsertNextPoint([intPt[0], intPt[1], intPt[2]])
            ptsOnSkel.append([intPt[0], intPt[1], intPt[2]])
            ax.scatter(intPt[0], intPt[1], intPt[2], color='b')
            # if j == 0:
            #     ax.plot([iXs[i], iSpoke[0][0]], [iYs[i], iSpoke[0][1]], [iZs[i], iSpoke[0][2]], 'r')
            # else:
            #     ax.plot([iSpoke[j-1][0], iSpoke[j][0]], [iSpoke[j-1][1], iSpoke[j][1]], [iSpoke[j-1][2], iSpoke[j][2]], 'r')
        iSamPts.append(iSpoke)
    else:
        iSpoke = []
        srepPts.InsertNextPoint([iXs[i], iYs[i], iZs[i]])
        ptsOnSkel.append([iXs[i], iYs[i], iZs[i]])
        for j in range(0, numSamp):
            intPt = rSamPts[2*i-1][j]
            iSpoke.append([intPt[0], intPt[1], intPt[2]])
            srepPts.InsertNextPoint([intPt[0], intPt[1], intPt[2]])
            ptsOnSkel.append([intPt[0], intPt[1], intPt[2]])
            ax.scatter(intPt[0], intPt[1], intPt[2], color='b')
            if j == 0:
                ax.plot([iXs[i], iSpoke[0][0]], [iYs[i], iSpoke[0][1]], [iZs[i], iSpoke[0][2]], 'r')
            else:
                ax.plot([iSpoke[j-1][0], iSpoke[j][0]], [iSpoke[j-1][1], iSpoke[j][1]], [iSpoke[j-1][2], iSpoke[j][2]], 'r')
        iSamPts.append(iSpoke)

        iSpoke = []
        srepPts.InsertNextPoint([iXs[i], iYs[i], iZs[i]])
        ptsOnSkel.append([iXs[i], iYs[i], iZs[i]])
        for j in range(0, numSamp):
            intPt = rSamPts[2*i][j]
            iSpoke.append([intPt[0], intPt[1], intPt[2]])
            srepPts.InsertNextPoint([intPt[0], intPt[1], intPt[2]])
            ptsOnSkel.append([intPt[0], intPt[1], intPt[2]])
            ax.scatter(intPt[0], intPt[1], intPt[2], color='b')
            if j == 0:
                ax.plot([iXs[i], iSpoke[0][0]], [iYs[i], iSpoke[0][1]], [iZs[i], iSpoke[0][2]], 'r')
            else:
                ax.plot([iSpoke[j-1][0], iSpoke[j][0]], [iSpoke[j-1][1], iSpoke[j][1]], [iSpoke[j-1][2], iSpoke[j][2]], 'r')
        iSamPts.append(iSpoke)
srepSet = vtk.vtkPolyData()
srepSet.SetPoints(srepPts)
srepSet.Modified()
ax.plot(iXs, iYs, iZs, 'r')

print(len(rXs))
print(len(ptsOnSkel))

# using thin plate splines to be able to interpolate 3D spokes on the ellipsoid s-rep
source_pts2 = vtk.vtkPoints()
target_pts2 = vtk.vtkPoints()
for i in range(math.floor((bot_srep.GetNumberOfCells() -24) / 2)):
    base_pt_id = i * 2
    bdry_pt_id = i * 2 + 1
    s_pt = bot_srep.GetPoint(base_pt_id)
    b_pt = bot_srep.GetPoint(bdry_pt_id)
    source_pts2.InsertNextPoint(s_pt)
    target_pts2.InsertNextPoint(b_pt)
tps = vtk.vtkThinPlateSplineTransform()
tps.SetSourceLandmarks(source_pts2)
tps.SetTargetLandmarks(target_pts2)
tps.SetBasisToR()
tps.Modified()

# ellipsoid mesh
reader3 = vtk.vtkPolyDataReader()
reader3.SetFileName('sreps/data/recenter.vtk')
reader3.Update()
bot_mesh = reader3.GetOutput()
# mesh of target object
reader4 = vtk.vtkPolyDataReader()
reader4.SetFileName('sreps/control/FinTopMesh1.vtk')
reader4.Update()
top_mesh = reader4.GetOutput()

source_pts3 = vtk.vtkPoints()
target_pts3 = vtk.vtkPoints()

ellPts = bot_mesh.GetNumberOfPoints()

# getting points on ellipsoid and target object mesh
for i in range(0, ellPts):
    pt = [0] * 3
    bot_mesh.GetPoint(i, pt)
    # print(pt)
    source_pts3.InsertNextPoint(pt)

    top_mesh.GetPoint(i, pt)
    # print(pt)
    target_pts3.InsertNextPoint(pt)
    # if i == 5:
    #     break
source_pts3.Modified()
target_pts3.Modified()

### Interpolate deformation with thin-plate-spline
tps2 = vtk.vtkThinPlateSplineTransform()
tps2.SetSourceLandmarks(source_pts3)
tps2.SetTargetLandmarks(target_pts3)
tps2.SetBasisToR()
tps2.Modified()

# method to find closest points on 2D skeleton
def findClosetPt(pt, pts):
    # test = pt[2]
    dist = (pt[0]-pts[0][0])**2+(pt[1]-pts[0][1])**2+(pt[2]-pts[0][2])**2
    ind0 = 0
    # print(pts)
    for i in range(0, len(pts)):
        val = (pt[0]-pts[i][0])**2+(pt[1]-pts[i][1])**2+(pt[2]-pts[i][2])**2
        if val < dist:
            dist = val
            ind0 = i
    
    nextPt = (ind0+10) % len(pts)
    prevPt = ind0-10
    if ind0 - 5 < 0:
        prevPt = ind0+5
    elif ind0 - 10 < 0:
        prevPt = ind0-5
    if ind0 + 5 > len(pts):
        nextPt = ind0 - 10
        prevPt = ind0 - 5
    elif ind0 + 10 > len(pts):
        nextPt = ind0 + 5
    distComp1 = (pt[0]-pts[nextPt][0])**2+(pt[1]-pts[nextPt][1])**2+(pt[2]-pts[nextPt][2])**2
    distComp2 = (pt[0]-pts[prevPt][0])**2+(pt[1]-pts[prevPt][1])**2+(pt[2]-pts[prevPt][2])**2

    if distComp1 < distComp2:
        ind1 = nextPt
    else:
        ind1 = prevPt

    if pts[ind0][0] == pts[ind1][0] and pts[ind0][1] == pts[ind1][1] and pts[ind0][2] == pts[ind1][2]:
        if ind1 == nextPt:
            ind1 = prevPt
        else:
            ind1 = nextPt

    # print([ind0, ind1])
    return dist, [ind0, ind1]

fitFrames = vtk.vtkPolyData()
fitFrames_ends = vtk.vtkPoints()
fitFrames_lines = vtk.vtkCellArray()

# finding closest 2D s-rep points on 2D s-rep
for i in range(0, math.floor((bot_srep.GetNumberOfCells() -24) / 2)):
    base_pt_id = i * 2
    bdry_pt_id = i * 2 + 1
    s_pt = bot_srep.GetPoint(base_pt_id)
    b_pt = bot_srep.GetPoint(bdry_pt_id)
    base_pt = np.array(s_pt)
    bdry_pt = np.array(b_pt)
    radius = np.linalg.norm(bdry_pt - base_pt)
    direction = (bdry_pt - base_pt) / radius

    numSamp = 5
    eps = 0.5
    
    dist, ind = findClosetPt([s_pt[0], s_pt[1], s_pt[2]], ptsOnSkel)
    # ax.scatter(s_pt[0], s_pt[1], s_pt[2], color='r')

    spk0 = math.floor(ind[0] / numSamp)
    spk1 = math.floor(ind[1] / numSamp)

    # if spk0 == len(rXs) -1:
    #     ax.scatter(ptsOnSkel[ind[0]][0], ptsOnSkel[ind[0]][1], ptsOnSkel[ind[0]][2], color='r')
    #     ax.scatter(ptsOnSkel[ind[1]][0], ptsOnSkel[ind[1]][1], ptsOnSkel[ind[1]][2], color='r')
    

    if spk0 % 2 == 1:
        if spk1 < spk0:
            uDir = [ptsOnSkel[ind[1]][0] - ptsOnSkel[ind[0]][0], ptsOnSkel[ind[1]][1] - ptsOnSkel[ind[0]][1], ptsOnSkel[ind[1]][2] - ptsOnSkel[ind[0]][2]]
        else:
            uDir = [ptsOnSkel[ind[0]][0] - ptsOnSkel[ind[1]][0], ptsOnSkel[ind[0]][1] - ptsOnSkel[ind[1]][1], ptsOnSkel[ind[0]][2] - ptsOnSkel[ind[1]][2]]
    else:
        if spk1 > spk0:
            uDir = [ptsOnSkel[ind[1]][0] - ptsOnSkel[ind[0]][0], ptsOnSkel[ind[1]][1] - ptsOnSkel[ind[0]][1], ptsOnSkel[ind[1]][2] - ptsOnSkel[ind[0]][2]]
        else:
            uDir = [ptsOnSkel[ind[0]][0] - ptsOnSkel[ind[1]][0], ptsOnSkel[ind[0]][1] - ptsOnSkel[ind[1]][1], ptsOnSkel[ind[0]][2] - ptsOnSkel[ind[1]][2]]
    if spk0 == 0:
        if spk1 % 2 == 1:
            uDir = [ptsOnSkel[ind[0]][0] - ptsOnSkel[ind[1]][0], ptsOnSkel[ind[0]][1] - ptsOnSkel[ind[1]][1], ptsOnSkel[ind[0]][2] - ptsOnSkel[ind[1]][2]]
        else:
            uDir = [ptsOnSkel[ind[1]][0] - ptsOnSkel[ind[0]][0], ptsOnSkel[ind[1]][1] - ptsOnSkel[ind[0]][1], ptsOnSkel[ind[1]][2] - ptsOnSkel[ind[0]][2]]           
    if ind[0] >= len(ptsOnSkel) - numSamp:
        if spk1 % 2 == 1:
            uDir = [ptsOnSkel[ind[1]][0] - ptsOnSkel[ind[0]][0], ptsOnSkel[ind[1]][1] - ptsOnSkel[ind[0]][1], ptsOnSkel[ind[1]][2] - ptsOnSkel[ind[0]][2]]
        else:
            uDir = [ptsOnSkel[ind[0]][0] - ptsOnSkel[ind[1]][0], ptsOnSkel[ind[0]][1] - ptsOnSkel[ind[1]][1], ptsOnSkel[ind[0]][2] - ptsOnSkel[ind[1]][2]]


    # uDir = [ptsOnSkel[ind[1]][0] - ptsOnSkel[ind[0]][0], ptsOnSkel[ind[1]][1] - ptsOnSkel[ind[0]][1], ptsOnSkel[ind[1]][2] - ptsOnSkel[ind[0]][2]]
    # print(ptsOnSkel[125])
    # print(ptsOnSkel[130])
    length = math.sqrt(uDir[0]**2 + uDir[1]**2 + uDir[2]**2)
    uDir[0] = uDir[0]*eps / length
    uDir[1] = uDir[1]*eps / length
    uDir[2] = uDir[2]*eps / length


    if ind[0] % 5 != 4 and ind[0] % 5 != 0:
        tao = [ptsOnSkel[ind[0]+1][0] - ptsOnSkel[ind[0]-1][0], ptsOnSkel[ind[0]+1][1] - ptsOnSkel[ind[0]-1][1], ptsOnSkel[ind[0]+1][2] - ptsOnSkel[ind[0]-1][2]]
    if ind[0] % 5 == 0:
        tao = [ptsOnSkel[ind[0]+1][0] - ptsOnSkel[ind[0]][0], ptsOnSkel[ind[0]+1][1] - ptsOnSkel[ind[0]][1], ptsOnSkel[ind[0]+1][2] - ptsOnSkel[ind[0]][2]]
    if ind[0] % 5 == 4:
        tao = [ptsOnSkel[ind[0]][0] - ptsOnSkel[ind[0]-1][0], ptsOnSkel[ind[0]][1] - ptsOnSkel[ind[0]-1][1], ptsOnSkel[ind[0]][2] - ptsOnSkel[ind[0]-1][2]]
    
    if ind[1] % 5 != 4 and ind[1] % 5 != 0:
        tao1 = [ptsOnSkel[ind[1]+1][0] - ptsOnSkel[ind[1]-1][0], ptsOnSkel[ind[1]+1][1] - ptsOnSkel[ind[1]-1][1], ptsOnSkel[ind[1]+1][2] - ptsOnSkel[ind[1]-1][2]]
    if ind[1] % 5 == 0:
        tao1 = [ptsOnSkel[ind[1]+1][0] - ptsOnSkel[ind[1]][0], ptsOnSkel[ind[1]+1][1] - ptsOnSkel[ind[1]][1], ptsOnSkel[ind[1]+1][2] - ptsOnSkel[ind[1]][2]]
    if ind[1] % 5 == 4:
        tao1 = [ptsOnSkel[ind[1]][0] - ptsOnSkel[ind[1]-1][0], ptsOnSkel[ind[1]][1] - ptsOnSkel[ind[1]-1][1], ptsOnSkel[ind[1]][2] - ptsOnSkel[ind[1]-1][2]]
    
    dist1 = (s_pt[0]-ptsOnSkel[ind[0]][0])**2+(s_pt[1]-ptsOnSkel[ind[0]][1])**2+(s_pt[2]-ptsOnSkel[ind[0]][2])**2
    dist2 = (s_pt[0]-ptsOnSkel[ind[1]][0])**2+(s_pt[1]-ptsOnSkel[ind[1]][1])**2+(s_pt[2]-ptsOnSkel[ind[1]][2])**2
    ratio1 = dist1 / (dist1+dist2)
    ratio2 = dist2 / (dist1+dist2)
    finTao = [(ratio1*tao[0]+ratio2*tao1[0])/2, (ratio1*tao[1]+ratio2*tao1[1])/2, (ratio1*tao[2]+ratio2*tao1[2])/2]
    lenT = math.sqrt(finTao[0]**2 + finTao[1]**2 + finTao[2]**2)
    finTao[0] = finTao[0]*eps / lenT
    finTao[1] = finTao[1]*eps / lenT
    finTao[2] = finTao[2]*eps / lenT

    tU = [s_pt[0]+finTao[0], s_pt[1]+finTao[1], s_pt[2]+finTao[2]]
    tD = [s_pt[0]-finTao[0], s_pt[1]-finTao[1], s_pt[2]-finTao[2]]
    uR =  [s_pt[0]+uDir[0], s_pt[1]+uDir[1], s_pt[2]+uDir[2]]
    uL =  [s_pt[0]-uDir[0], s_pt[1]-uDir[1], s_pt[2]-uDir[2]]

    spokeTU = tps.TransformPoint(tU)
    vecTU = [spokeTU[0] - tU[0], spokeTU[1] - tU[1], spokeTU[2] - tU[2]]
    lengthTU = math.sqrt(vecTU[0]**2 + vecTU[1]**2 + vecTU[2]**2)
    vecTU = [vecTU[0]/lengthTU, vecTU[1]/lengthTU, vecTU[2]/lengthTU]

    spokeTD = tps.TransformPoint(tD)
    vecTD = [spokeTD[0] - tD[0], spokeTD[1] - tD[1], spokeTD[2] - tD[2]]
    lengthTD = math.sqrt(vecTD[0]**2 + vecTD[1]**2 + vecTD[2]**2)
    vecTD = [vecTD[0]/lengthTD, vecTD[1]/lengthTD, vecTD[2]/lengthTD]

    spoke = tps.TransformPoint(s_pt)
    vec = [spoke[0] - s_pt[0], spoke[1] - s_pt[1], spoke[2] - s_pt[2]]
    length = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    vec = [vec[0]/length, vec[1]/length, vec[2]/length]

    spokeUR = tps.TransformPoint(uR)
    vecUR = [spokeUR[0] - uR[0], spokeUR[1] - uR[1], spokeUR[2] - uR[2]]
    lengthUR = math.sqrt(vecUR[0]**2 + vecUR[1]**2 + vecUR[2]**2)
    vecUR = [vecUR[0]/lengthUR, vecUR[1]/lengthUR, vecUR[2]/lengthUR]

    spokeUL = tps.TransformPoint(uL)
    vecUL = [spokeUL[0] - uL[0], spokeUL[1] - uL[1], spokeUL[2] - uL[2]]
    lengthUL = math.sqrt(vecUL[0]**2 + vecUL[1]**2 + vecUL[2]**2)
    vecUL = [vecUL[0]/lengthUL, vecUL[1]/lengthUL, vecUL[2]/lengthUL]

    spokeSam = 5

    for j in range(0, spokeSam):
        unit = length/(spokeSam-1)
        spokePt = [s_pt[0] + vec[0]*j*unit, s_pt[1] + vec[1]*j*unit, s_pt[2] + vec[2]*j*unit]

        unit = lengthTU/(spokeSam-1)
        tUpt = [tU[0] + vecTU[0]*j*unit, tU[1] + vecTU[1]*j*unit, tU[2] + vecTU[2]*j*unit]

        unit = lengthTD/(spokeSam-1)
        tDpt = [tD[0] + vecTD[0]*j*unit, tD[1] + vecTD[1]*j*unit, tD[2] + vecTD[2]*j*unit]

        unit = lengthUR/(spokeSam-1)
        uRpt = [uR[0] + vecUR[0]*j*unit, uR[1] + vecUR[1]*j*unit, uR[2] + vecUR[2]*j*unit]

        unit = lengthUL/(spokeSam-1)
        uLpt = [uL[0] + vecUL[0]*j*unit, uL[1] + vecUL[1]*j*unit, uL[2] + vecUL[2]*j*unit]

        # if spk0 == len(rXs) -1:
        #     ax.scatter(spokePt[0], spokePt[1], spokePt[2], color='k')

        spokeT = tps2.TransformPoint(spokePt)
        tUtrans = tps2.TransformPoint(tUpt)
        tDtrans = tps2.TransformPoint(tDpt)
        uRtrans = tps2.TransformPoint(uRpt)
        uLtrans = tps2.TransformPoint(uLpt)

        # ax.scatter(tDpt[0], tDpt[1], tDpt[2], color='k')
        # ax.scatter(tUpt[0], tUpt[1], tUpt[2], color='k')
        # ax.scatter(uRpt[0], uRpt[1], uRpt[2], color='k')
        # ax.scatter(uLpt[0], uLpt[1], uLpt[2], color='k')
        # ax.scatter(tUtrans[0], tUtrans[1], tUtrans[2], color='k')
        # ax.scatter(tDtrans[0], tDtrans[1], tDtrans[2], color='r')
        # if j == 4:
        #     ax.scatter(uRtrans[0], uRtrans[1], uRtrans[2], color='k')
        #     ax.scatter(uLtrans[0], uLtrans[1], uLtrans[2], color='r')

        # ax.scatter(tU[0]+ tU[0]*4*unit, tU[1]+ vecTU[1]*4*unit, tU[2]+ vecTU[2]*4*unit, color='k')
        # ax.scatter(ptsOnSkel[ind[0]][0], ptsOnSkel[ind[0]][1], ptsOnSkel[ind[0]][2], color='r')
        # ax.scatter(ptsOnSkel[ind[0]][0], ptsOnSkel[ind[0]][1], ptsOnSkel[ind[0]][2], color='r')
        # ax.scatter(ptsOnSkel[ind[0]][0], ptsOnSkel[ind[0]][1], ptsOnSkel[ind[0]][2], color='r')

        uVec = [uRtrans[0] - uLtrans[0], uRtrans[1] - uLtrans[1], uRtrans[2] - uLtrans[2]]
        uDist = np.linalg.norm(uVec)
        uVec = [uVec[0]/uDist, uVec[1]/uDist, uVec[2]/uDist]

        tVec = [tUtrans[0] - tDtrans[0], tUtrans[1] - tDtrans[1], tUtrans[2] - tDtrans[2]]
        tDist = np.linalg.norm(tVec)
        tVec = [tVec[0]/tDist, tVec[1]/tDist, tVec[2]/tDist]

        norm1 = np.cross(uVec, tVec)
        norm2 = np.cross(tVec, uVec)
        

        finU = [spokeT[0] + uVec[0], spokeT[1] + uVec[1], spokeT[2] + uVec[2]]
        
        finN1 = [spokeT[0] + norm1[0], spokeT[1] + norm1[1], spokeT[2] + norm1[2]]
        finN2 = [spokeT[0] + norm2[0], spokeT[1] + norm2[1], spokeT[2] + norm2[2]]

        nextPt = [s_pt[0] + vec[0]*(j+1)*unit, s_pt[1] + vec[1]*(j+1)*unit, s_pt[2] + vec[2]*(j+1)*unit]
        if j == 4:
            nextPt = [s_pt[0] + vec[0]*(j-1)*unit, s_pt[1] + vec[1]*(j-1)*unit, s_pt[2] + vec[2]*(j-1)*unit]
   
        normToPt1 = [nextPt[0] - finN1[0], nextPt[1] - finN1[1], nextPt[2] - finN1[2]]
        normDist1 = np.linalg.norm(normToPt1)
        normToPt2 = [nextPt[0] - finN2[0], nextPt[1] - finN2[1], nextPt[2] - finN2[2]]
        normDist2 = np.linalg.norm(normToPt2)

        boo1 = False
        if normDist1 < normDist2:
            finN = finN1
            vec3 = np.cross(uVec, norm1)
            boo1 = True
        else:
            finN = finN2
            vec3 = np.cross(uVec, norm2)

        if j == 4:
            if boo1:
                finN = finN2
                vec3 = np.cross(uVec, norm2)
            else:
                finN = finN1
                vec3 = np.cross(uVec, norm1)

        
        finT1 = [spokeT[0] + vec3[0], spokeT[1] + vec3[1], spokeT[2] + vec3[2]]
        normToPt1 = [tUtrans[0] - finT1[0], tUtrans[1] - finT1[1], tUtrans[2] - finT1[2]]
        normDist1 = np.linalg.norm(normToPt1)

        finT2 = [spokeT[0] - vec3[0], spokeT[1] - vec3[1], spokeT[2] - vec3[2]]
        normToPt2 = [tUtrans[0] - finT2[0], tUtrans[1] - finT2[1], tUtrans[2] - finT2[2]]
        normDist2 = np.linalg.norm(normToPt2)

        boo1 = False
        if normDist1 < normDist2:
            finT = finT1
        else:
            finT = finT2
            vec3 = np.cross(uVec, norm2)

        # if j == 4:
        #     if boo1:
        #         finN = finN2
        #         vec3 = np.cross(uVec, norm2)
        #     else:
        #         finN = finN1
        #         vec3 = np.cross(uVec, norm1)





        # if spk0 != 0 and spk0 != len(rXs) -1:
        ax.plot([spokeT[0], finU[0]], [spokeT[1], finU[1]], [spokeT[2], finU[2]], 'k')
        ax.plot([spokeT[0], finT[0]], [spokeT[1], finT[1]], [spokeT[2], finT[2]], 'k')
        ax.plot([spokeT[0], finN[0]], [spokeT[1], finN[1]], [spokeT[2], finN[2]], 'k')

        id0 = fitFrames_ends.InsertNextPoint(tuple(spokeT))
        id1 = fitFrames_ends.InsertNextPoint(tuple(finU))
        spoke_line = vtk.vtkLine()
        spoke_line.GetPointIds().SetId(0, id0)
        spoke_line.GetPointIds().SetId(1, id1)
        fitFrames_lines.InsertNextCell(spoke_line)

        id0 = fitFrames_ends.InsertNextPoint(tuple(spokeT))
        id1 = fitFrames_ends.InsertNextPoint(tuple(finT))
        spoke_line = vtk.vtkLine()
        spoke_line.GetPointIds().SetId(0, id0)
        spoke_line.GetPointIds().SetId(1, id1)
        fitFrames_lines.InsertNextCell(spoke_line)

        id0 = fitFrames_ends.InsertNextPoint(tuple(spokeT))
        id1 = fitFrames_ends.InsertNextPoint(tuple(finN))
        spoke_line = vtk.vtkLine()
        spoke_line.GetPointIds().SetId(0, id0)
        spoke_line.GetPointIds().SetId(1, id1)
        fitFrames_lines.InsertNextCell(spoke_line)

    # ax.plot([tU[0], spokeTU[0]], [tU[1], spokeTU[1]], [tU[2], spokeTU[2]], 'b')
    # ax.plot([tD[0], spokeTD[0]], [tD[1], spokeTD[1]], [tD[2], spokeTD[2]], 'b')
    # ax.plot([s_pt[0], spoke[0]], [s_pt[1], spoke[1]], [s_pt[2], spoke[2]], 'r')
    # ax.plot([uR[0], spokeUR[0]], [uR[1], spokeUR[1]], [uR[2], spokeUR[2]], 'b')
    # ax.plot([uL[0], spokeUL[0]], [uL[1], spokeUL[1]], [uL[2], spokeUL[2]], 'b')

    # ax.plot([s_pt[0], s_pt[0]+finTao[0]], [s_pt[1], s_pt[1]+finTao[1]], [s_pt[2], s_pt[2]+finTao[2]], 'k')
    # ax.plot([s_pt[0], s_pt[0]-finTao[0]], [s_pt[1], s_pt[1]-finTao[1]], [s_pt[2], s_pt[2]-finTao[2]], 'k')
    # ax.plot([s_pt[0], s_pt[0]+uDir[0]], [s_pt[1], s_pt[1]+uDir[1]], [s_pt[2], s_pt[2]+uDir[2]], 'k')
    # ax.plot([s_pt[0], s_pt[0]-uDir[0]], [s_pt[1], s_pt[1]-uDir[1]], [s_pt[2], s_pt[2]-uDir[2]], 'k')
    # print(finTao)
    # print(uDir)
    # print(str(i) + " index")
    # if i == 100:
    #     break

    # get U and T from the closest points on 2d skel
    # use TPH to interpolate 3D spoke # 2 in U sides and 2 in T sides
    # sample points on the actual spokes and the interpolated spokes
    # use TPH to find where those sampled points would be in actual 3d object
    #   get 3d mesh, ellipsoid points as source and target object points as target
    # tph the neighboring points for the frames for the target object


plt.show()

fitFrames.SetPoints(fitFrames_ends)
fitFrames.SetLines(fitFrames_lines)
fitFrames.Modified()

writer2 = vtk.vtkPolyDataWriter()
writer2.SetInputData(fitFrames)
writer2.SetFileName('data/frames.vtk')
writer2.Write()
