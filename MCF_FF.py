import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from glob import glob
import vtk
import scipy
import os
import re

def flow(mesh_file, iter_num, show_flow=False):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(mesh_file)
    reader.Update()

    mesh = reader.GetOutput()
    tol = 0.05
    q = 1.0
    dt = 0.001
    orig_mesh = mesh
    prev_mesh = mesh
    thin_plate_spline_list = []
    for i in range(iter_num + 1):
        deformed_surface_writer = vtk.vtkPolyDataWriter()
        deformed_surface_writer.SetFileName('data/flow/' + str(i) + '.vtk')
        deformed_surface_writer.SetInputData(mesh)
        deformed_surface_writer.Update()

        taubin_smooth = vtk.vtkWindowedSincPolyDataFilter()
        taubin_smooth.SetInputData(mesh)
        taubin_smooth.SetNumberOfIterations(20)
        taubin_smooth.BoundarySmoothingOff()
        taubin_smooth.FeatureEdgeSmoothingOff()
        taubin_smooth.SetPassBand(0.01)
        taubin_smooth.NonManifoldSmoothingOn()
        taubin_smooth.NormalizeCoordinatesOn()

        taubin_smooth.Update()
        mesh = taubin_smooth.GetOutput()

        normal_generator = vtk.vtkPolyDataNormals()
        normal_generator.SetInputData(mesh)
        normal_generator.SplittingOff()
        normal_generator.ComputePointNormalsOn()
        normal_generator.ComputeCellNormalsOff()
        normal_generator.Update()
        mesh = normal_generator.GetOutput()

        curvatures = vtk.vtkCurvatures()
        curvatures.SetCurvatureTypeToMean()
        curvatures.SetInputData(mesh)
        curvatures.Update()

        mean_curvatures = curvatures.GetOutput().GetPointData().GetArray("Mean_Curvature")
        normals = normal_generator.GetOutput().GetPointData().GetNormals()

        mesh_pts = mesh.GetPoints()
        for j in range(mesh.GetNumberOfPoints()):
            current_point = mesh.GetPoint(j)
            current_normal = np.array(normals.GetTuple3(j))
            current_mean_curvature = mean_curvatures.GetValue(j)

            pt = np.array(mesh_pts.GetPoint(j))
            pt -= dt * current_mean_curvature * current_normal
            mesh_pts.SetPoint(j, pt)
        mesh_pts.Modified()
        mesh.SetPoints(mesh_pts)
        mesh.Modified()
#        if i % 100 == 0 and show_flow:
            # plotter = pyvista.Plotter()
            # plotter.add_mesh(mesh)
            # plotter.show()

        tps_deform = get_thin_plate_spline_deform(prev_mesh, mesh)

        prev_mesh = mesh
        thin_plate_spline_list.append(tps_deform)
    # new_mesh = apply_tps_on_spokes_poly(mesh, thin_plate_spline_list)
    # plotter = pyvista.Plotter()
    # plotter.add_mesh(new_mesh, color='red', opacity=0.3)
    # plotter.add_mesh(orig_mesh, color='blue', opacity=0.3)
    # plotter.show()
    return mesh, thin_plate_spline_list

def get_thin_plate_spline_deform(input_target_mesh, input_source_mesh):
    ## sparsen boundary points to speed up computation
    # clean_ratio = 0.1
    # clean_data_polydata = vtk.vtkCleanPolyData()
    # clean_data_polydata.SetInputData(input_target_mesh)
    # clean_data_polydata.SetTolerance(clean_ratio)
    # clean_data_polydata.Update()
    # target_mesh = clean_data_polydata.GetOutput()
    
    # source_clean_data_polydata = vtk.vtkCleanPolyData()
    # source_clean_data_polydata.SetTolerance(clean_ratio)
    # source_clean_data_polydata.SetInputData(input_source_mesh)
    # source_clean_data_polydata.Update()
    # source_mesh = source_clean_data_polydata.GetOutput()

    target_mesh = input_target_mesh
    source_mesh = input_source_mesh
#    compute_distance_between_poly(target_mesh, source_mesh)
    source_pts = vtk.vtkPoints()
    target_pts = vtk.vtkPoints()
    for i in range(target_mesh.GetNumberOfPoints()):
        source_pts.InsertNextPoint(source_mesh.GetPoint(i))
        target_pts.InsertNextPoint(target_mesh.GetPoint(i))
    tps = vtk.vtkThinPlateSplineTransform()
    tps.SetSourceLandmarks(source_pts)
    tps.SetTargetLandmarks(target_pts)
    tps.SetBasisToR()
    tps.Modified()
    return tps
flow('data/107524.vtk', 25)