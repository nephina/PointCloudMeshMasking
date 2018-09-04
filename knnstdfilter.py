from sklearn.neighbors import NearestNeighbors
import numpy as np
from stl import mesh
import csv
import pycaster
from pycaster.pycaster import rayCaster

pv3d_file_path = 'Combined.pv3d'
stl_file_path = 'PhantomMaskforAugust27thData.stl'
stdev_threshold_multiplier = 2
#mad_threshold_multiplier = 50
masking_nearest_neighbor_num = 250 #how many triangles to run through the ray_triangle_search_width filter This matters a lot for particles that lie near geometry boundaries that are parallel to the x direction, which is the direction that the ray is cast. There are a lot of possible triangles, but the ray only passes through one of them, if you set this number too low, it may not find the correct triangle
ray_triangle_search_width = 0.2 #mm #how far any triangle centroid can be from the testing ray
stat_filtering_nearest_neighbor_num = 50 #how many particles to 

def Ray_Triangle_Intersection(p0, p1, triangle):
    v0, v1, v2 = triangle
    u = v1 - v0
    v = v2 - v0
    normal = np.cross(u, v)
    b = np.inner(normal, p1 - p0)
    a = np.inner(normal, v0 - p0)
    if (b == 0.0):
        if a != 0.0:
            return 0
        else:
            rI = 0.0
    else:
        rI = a / b
    if rI < 0.0:
        return 0
    w = p0 + rI * (p1 - p0) - v0
    denom = np.inner(u, v) * np.inner(u, v) - \
        np.inner(u, u) * np.inner(v, v)
    si = (np.inner(u, v) * np.inner(w, v) - \
        np.inner(v, v) * np.inner(w, u)) / denom
    if (si < 0.0) | (si > 1.0):
        return 0
    ti = (np.inner(u, v) * np.inner(w, u) - \
        np.inner(u, u) * np.inner(w, v)) / denom
    if (ti < 0.0) | (si + ti > 1.0):
        return 0
    if (rI == 0.0):
        return 2
    return 1

def Geometry_Mask_pycaster(data,stl_file_path): #good for absolute particle masking, slow
	points = data[:,0:3]
	xbounds = np.max(points[0])+(0.1*np.ptp(points[0])),np.min(points[0])-(0.1*np.ptp(points[0]))
	data_shape = np.shape(data)
	print('\nFinding points inside geometry...')
	mask_indices = np.zeros(data_shape[0])
	for point in range(data_shape[0]):
		ray_segment = np.hstack((points[point,:],xbounds[0], points[point,1:3])) #define the line segment as a ray heading upwards in the z direction from any given point, terminating at the top of the bounding box around the pointcloud
		intersection_number = 0
		caster = rayCaster.fromSTL(stl_path,scale=1)
		intersection_points = caster.castRay(ray_segment[0:3],ray_segment[3:6])
		intersection_shape = np.shape(intersection_points)
		if intersection_shape[0] % 2 != 0:
			mask_indices[point] = 1
			print('point #',point,'is inside\n')
	masked_data = data[(mask_indices != 0),:]
	return masked_data


def Geometry_Mask_knn_optimized(data,stl_file_path,masking_nearest_neighbor_num,ray_triangle_search_width): #does its best, but will miss some outside particles, and discard some inside particles, is pretty fast
	points = data[:,0:3]
	xbounds = np.max(points[0])+(0.1*np.ptp(points[0])),np.min(points[0])-(0.1*np.ptp(points[0]))
	data_shape = np.shape(data)
	print('\nLoading STL mask geometry...')
	masking_mesh = mesh.Mesh.from_file(stl_file_path)
	stl_shape = np.shape(masking_mesh)
	centroids = np.transpose([((masking_mesh[:,0]+masking_mesh[:,3]+masking_mesh[:,6])/3),((masking_mesh[:,1]+masking_mesh[:,4]+masking_mesh[:,7])/3),((masking_mesh[:,2]+masking_mesh[:,5]+masking_mesh[:,8])/3)]) #use the centroids of each triangle to find the closest faces
	print('\nTraining KNN on mask geometry vertex points...')
	mesh_neighbors = NearestNeighbors(n_neighbors=masking_nearest_neighbor_num,algorithm='auto').fit(centroids[:,1:3]) #use only the first vertex, and only the y and z coordinates of that vertex. That way referencing the triangle later is easy, it uses the same coordinates as the stl matrix
	vertexdistance,vertexindex = mesh_neighbors.kneighbors(points[:,1:3])
	print('\nFinding points inside geometry...')
	mask_indices = np.zeros(data_shape[0])
	triangle_matrix_shape = np.shape(masking_mesh[vertexindex[0,:]])
	for point in range(data_shape[0]):
		ray_segment = np.hstack((points[point,:],xbounds[0], points[point,1:3])) #define the line segment as a ray heading upwards in the z direction from any given point, terminating at the top of the bounding box around the pointcloud
		triangles_for_intersect_testing = masking_mesh[vertexindex[point,:]]
		intersection_number = 0
		for triangles in range(triangle_matrix_shape[0]):
			if vertexdistance[point,triangles] < ray_triangle_search_width:
				if (Ray_Triangle_Intersection(ray_segment[0:3],ray_segment[3:6],np.reshape(triangles_for_intersect_testing[triangles,:],[3,3])) > 0):
					intersection_number+=1
		if intersection_number % 2 != 0:
			mask_indices[point] = 1
			print('point #',point,'is inside')
	masked_data = data[(mask_indices != 0),:]
	return masked_data


def Mode_Filtering(data,stat_filtering_nearest_neighbor_num,stdev_threshold_multiplier):
	points = data[:,0:3]
	data_shape = np.shape(data)
	print('\nTraining KNN on Masked Pointcloud...')
	neighbors = NearestNeighbors(n_neighbors=stat_filtering_nearest_neighbor_num, algorithm='auto').fit(points)
	distances,indices = neighbors.kneighbors(points)
	print('\nStatistical Analysis...')
	velocity_std_dev = stdev_threshold_multiplier*np.std(data[indices,3:6],axis=1)
	velocity_mode = np.empty((data_shape[0],3))
	sorted_point_velocities=np.empty(stat_filtering_nearest_neighbor_num)
	Differences = np.empty(stat_filtering_nearest_neighbor_num-1)
	for point in range(data_shape[0]):
		for dimension in range(3):
			sorted_point_velocities = sorted(data[indices[point],3+dimension])
			Differences =[sorted_point_velocities[i+1]-sorted_point_velocities[i] for i in range(stat_filtering_nearest_neighbor_num) if i+1 < stat_filtering_nearest_neighbor_num]
			velocity_mode[point,dimension] = (sorted_point_velocities[np.argmin(Differences)]+sorted_point_velocities[np.argmin(Differences)+1])/2
	print('\nTesting Points...')
	passed,failed = 0,0
	pass_index = np.empty(data_shape[0],dtype=int)
	for i in range(int(data_shape[0])):
		if data[i,3] > (velocity_mode[i,0]-velocity_std_dev[i,0]) and data[i,3] < (velocity_mode[i,0]+velocity_std_dev[i,0]) and data[i,4] > (velocity_mode[i,1]-velocity_std_dev[i,1]) and data[i,4] < (velocity_mode[i,1]+velocity_std_dev[i,1]) and data[i,5] > (velocity_mode[i,2]-velocity_std_dev[i,2]) and data[i,5] < (velocity_mode[i,2]+velocity_std_dev[i,2]):
			pass_index[i] = i
			passed+=1
		else:
			pass_index[i] = 0
			failed+=1
	pass_indexed = pass_index[pass_index!=0]
	filtered_data=data[pass_indexed,:]
	return filtered_data

#def Read_PV3D(pv3d_file_path)


print('Loading pv3d data...')
data = np.genfromtxt(pv3d_file_path, dtype=np.float64, delimiter=',',skip_header=1)

masked_data = Geometry_Mask_knn_optimized(data,stl_file_path,masking_nearest_neighbor_num,ray_triangle_search_width)
filtered_masked_data = masked_data
#filtered_masked_data = Mode_Filtering(masked_data,stat_filtering_nearest_neighbor_num,stdev_threshold_multiplier)
filtered_masked_data_shape = np.shape(filtered_masked_data)

with open('Standard Deviation Filtered(200,0.5).ply', mode='w') as Output:
	Output.write('ply\nformat ascii 1.0\nelement vertex '+str(filtered_masked_data_shape[0]-1)+'\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nend_header\n')
	for i in range(1,filtered_masked_data_shape[0]):
		Output.write('\n'+str(filtered_masked_data[i,0])+' '+str(filtered_masked_data[i,1])+' '+str(filtered_masked_data[i,2])+' '+str(filtered_masked_data[i,3])+' '+str(filtered_masked_data[i,4])+' '+str(filtered_masked_data[i,5])+'\n')