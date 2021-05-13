import vtk
import math
import numpy as np
import curvedSrep as cs
import matplotlib.pyplot as plt
from numpy import random
from scipy.spatial import distance


plt.figure()
ax = plt.subplot(111, projection='3d')

reader = vtk.vtkPolyDataReader()
reader.SetFileName('data/normEll.vtk')
reader.Update()
bot_srep = reader.GetOutput()

reader2 = vtk.vtkPolyDataReader()
reader2.SetFileName('control/top_srep_twist0.vtk')
reader2.Update()
top_srep = reader.GetOutput()

source_pts = vtk.vtkPoints()
for i in range(bot_srep.GetNumberOfCells()):
    base_pt_id = i * 2
    bdry_pt_id = i * 2 + 1
    s_pt = bot_srep.GetPoint(base_pt_id)
    b_pt = bot_srep.GetPoint(bdry_pt_id)
    # ax.scatter(s_pt[0], s_pt[1], s_pt[2], color='k')
    source_pts.InsertNextPoint([s_pt[0], s_pt[1], s_pt[2]])
source_pts.Modified()

rXs, rYs, rZs, rSamPts = cs.curvedSrep(source_pts)
numSamp = len(rSamPts[0])

iXs = []
iYs = []
iZs = []
iSamPts = []
srepPts = vtk.vtkPoints()
pPts = vtk.vtkPoints()

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
            if j == 0:
                ax.plot([iXs[i], iSpoke[0][0]], [iYs[i], iSpoke[0][1]], [iZs[i], iSpoke[0][2]], 'r')
            else:
                ax.plot([iSpoke[j-1][0], iSpoke[j][0]], [iSpoke[j-1][1], iSpoke[j][1]], [iSpoke[j-1][2], iSpoke[j][2]], 'r')
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
plt.show()

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

def findClosetPt(pt, pts):
    dist = 100
    ind = -1
    for i in range(0, len(pts)):
        val = (pt[0]-pts[i][0])**2+(pt[1]-pts[i][1])**2+(pt[2]-pts[i][2])**2
        if val < dist:
            dist = val
            ind = i
    return dist, i

for i in range(0, (bot_srep.GetNumberOfCells() -24) / 2):
    base_pt_id = i * 2
    bdry_pt_id = i * 2 + 1
    s_pt = bot_srep.GetPoint(base_pt_id)
    b_pt = bot_srep.GetPoint(bdry_pt_id)
    base_pt = np.array(s_pt)
    bdry_pt = np.array(b_pt)
    radius = np.linalg.norm(bdry_pt - base_pt)
    direction = (bdry_pt - base_pt) / radius

    numSamp = 5
    eps = 0.05
    ind, dist = findClosetPt([s_pt[0], s_pt[1], s_pt[2]], ptsOnSkel)

fitFrames = vtk.vtkPolyData()
fitFrames_ends = vtk.vtkPoints()
fitFrames_lines = vtk.vtkCellArray()

for i in range(0, (bot_srep.GetNumberOfCells() -24) / 2):
    base_pt_id = i * 2
    bdry_pt_id = i * 2 + 1
    s_pt = bot_srep.GetPoint(base_pt_id)
    b_pt = bot_srep.GetPoint(bdry_pt_id)
    base_pt = np.array(s_pt)
    bdry_pt = np.array(b_pt)
    radius = np.linalg.norm(bdry_pt - base_pt)
    direction = (bdry_pt - base_pt) / radius

    numSamp = 5
    eps = 0.05
    
    lSpt = list(s_pt)
    lSpt[0] = lSpt[0] - eps
    lSpt = tuple(lSpt)
    leftS = tps.TransformPoint(lSpt)

    base_pt = np.array(lSpt)
    bdry_pt = np.array(leftS)
    radiusL = np.linalg.norm(bdry_pt - base_pt)
    directionL = (bdry_pt - base_pt) / radiusL

    rSpt = list(s_pt)
    rSpt[0] = rSpt[0] + eps
    rSpt = tuple(rSpt)
    rightS = tps.TransformPoint(rSpt)

    base_pt = np.array(rSpt)
    bdry_pt = np.array(rightS)
    radiusR = np.linalg.norm(bdry_pt - base_pt)
    directionR = (bdry_pt - base_pt) / radiusR

    uSpt = list(s_pt)
    uSpt[1] = uSpt[1] + eps
    uSpt = tuple(uSpt)
    upS = tps.TransformPoint(uSpt)

    base_pt = np.array(uSpt)
    bdry_pt = np.array(upS)
    radiusU = np.linalg.norm(bdry_pt - base_pt)
    directionU = (bdry_pt - base_pt) / radiusU

    dSpt = list(s_pt)
    dSpt[1] = dSpt[1] - eps
    dSpt = tuple(dSpt)
    downS = tps.TransformPoint(dSpt)

    base_pt = np.array(dSpt)
    bdry_pt = np.array(downS)
    radiusD = np.linalg.norm(bdry_pt - base_pt)
    directionD = (bdry_pt - base_pt) / radiusD

    for j in range(0, numSamp):
        ptC = list(s_pt)
        dir = list(direction)
        dist = radius*j / (numSamp-1)
        if j < numSamp-1:
            nPt = [0,0,0]
            dist2 = radius*(j+1) / (numSamp-1)
            nPt[0] = ptC[0] + dir[0]*dist2
            nPt[1] = ptC[1] + dir[1]*dist2
            nPt[2] = ptC[2] + dir[2]*dist2


        ptC[0] = ptC[0] + dir[0]*dist
        ptC[1] = ptC[1] + dir[1]*dist
        ptC[2] = ptC[2] + dir[2]*dist

        # if j < numSamp-1:
        #     cSrep = tps3.TransformPoint(tuple(ptC))
        #     id0 = spoke_ends.InsertNextPoint(cSrep)
        #     cSrep2 = tps3.TransformPoint(tuple(nPt))
        #     id1 = spoke_ends.InsertNextPoint(cSrep2)
        #     spoke_seg = vtk.vtkLine()
        #     spoke_seg.GetPointIds().SetId(0, id0)
        #     spoke_seg.GetPointIds().SetId(1, id1)
        #     srepLines.InsertNextCell(spoke_seg)


        ptL = list(lSpt)
        dirL = list(directionL)
        distL = radiusL*j / (numSamp-1)
        ptL[0] = ptL[0] + dirL[0]*distL
        ptL[1] = ptL[1] + dirL[1]*distL
        ptL[2] = ptL[2] + dirL[2]*distL

        ptR = list(rSpt)
        dirR = list(directionR)
        distR = radiusR*j / (numSamp-1)
        ptR[0] = ptR[0] + dirR[0]*distR
        ptR[1] = ptR[1] + dirR[1]*distR
        ptR[2] = ptR[2] + dirR[2]*distR

        ptU = list(uSpt)
        dirU = list(directionU)
        distU = radiusU*j / (numSamp-1)
        ptU[0] = ptU[0] + dirU[0]*distU
        ptU[1] = ptU[1] + dirU[1]*distU
        ptU[2] = ptU[2] + dirU[2]*distU

        ptD = list(dSpt)
        dirD = list(directionD)
        distD = radiusD*j / (numSamp-1)
        ptD[0] = ptD[0] + dirD[0]*distD
        ptD[1] = ptD[1] + dirD[1]*distD
        ptD[2] = ptD[2] + dirD[2]*distD

        uVec = [ptR[0] - ptL[0], ptR[1] - ptL[1], ptR[2] - ptL[2]]
        uDist = np.linalg.norm([ptR[0] - ptL[0], ptR[1] - ptL[1], ptR[2] - ptL[2]])
        uVec = [uVec[0]/uDist, uVec[1]/uDist, uVec[2]/uDist]

        tVec = [ptU[0] - ptD[0], ptU[1] - ptD[1], ptU[2] - ptD[2]]
        tDist = np.linalg.norm([ptU[0] - ptD[0], ptU[1] - ptD[1], ptU[2] - ptD[2]])
        tVec = [tVec[0]/tDist, tVec[1]/tDist, tVec[2]/tDist]

        norm = np.cross(uVec, tVec)
        if i > math.floor((bot_srep.GetNumberOfCells() -24) / 2):
            norm = np.cross(tVec, uVec)

        uFrame = [ptC[0]+ uVec[0], ptC[1]+ uVec[1], ptC[2]+ uVec[2]]
        tFrame = [ptC[0]+ tVec[0], ptC[1]+ tVec[1], ptC[2]+ tVec[2]]
        nFrame = [ptC[0]- norm[0], ptC[1]- norm[1], ptC[2]- norm[2]]

        id0 = fitFrames_ends.InsertNextPoint(tuple(ptC))
        id1 = fitFrames_ends.InsertNextPoint(tuple(uFrame))
        spoke_line = vtk.vtkLine()
        spoke_line.GetPointIds().SetId(0, id0)
        spoke_line.GetPointIds().SetId(1, id1)
        fitFrames_lines.InsertNextCell(spoke_line)

        id0 = fitFrames_ends.InsertNextPoint(tuple(ptC))
        id1 = fitFrames_ends.InsertNextPoint(tuple(tFrame))
        spoke_line = vtk.vtkLine()
        spoke_line.GetPointIds().SetId(0, id0)
        spoke_line.GetPointIds().SetId(1, id1)
        fitFrames_lines.InsertNextCell(spoke_line)

        id0 = fitFrames_ends.InsertNextPoint(tuple(ptC))
        id1 = fitFrames_ends.InsertNextPoint(tuple(nFrame))
        spoke_line = vtk.vtkLine()
        spoke_line.GetPointIds().SetId(0, id0)
        spoke_line.GetPointIds().SetId(1, id1)
        fitFrames_lines.InsertNextCell(spoke_line)

fitFrames.SetPoints(fitFrames_ends)
fitFrames.SetLines(fitFrames_lines)
fitFrames.Modified()

writer2 = vtk.vtkPolyDataWriter()
writer2.SetInputData(fitFrames)
writer2.SetFileName('data/frames.vtk')
writer2.Write()

# finSrep = vtk.vtkPolyData()
# finSrep.SetPoints(spoke_ends)
# finSrep.SetLines(srepLines)
# finSrep.Modified()







