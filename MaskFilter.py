from sklearn.neighbors import NearestNeighbors
import numpy as np
from stl import mesh
import pycaster
from pycaster.pycaster import rayCaster
import os,fnmatch,csv

pv3d_file_path = 'Combined.pv3d'
pv3d_folder_path = './PV3D Files/'
stl_file_path = 'PhantomMaskforAugust27thData.stl'
stdev_threshold_multiplier = 2
masking_nearest_neighbor_num = 250 #how many triangles to run through the ray_triangle_search_width filter This matters a lot for particles that lie near geometry boundaries that are parallel to the x direction, which is the direction that the ray is cast. There are a lot of possible triangles, but the ray only passes through one of them, if you set this number too low, it may not find the correct triangle
ray_triangle_search_width = 0.5 #mm #how far any triangle centroid can be from the testing ray
stat_filtering_nearest_neighbor_num = 100 #how many neighbors to use to calculate the statistical model for any given point
sparse_filling_nearest_neighbor_num = 2 #keep this as low as possible, otherwise there will be far too many added points (2 is mathematically optimal, may enforce it later)
min_distance_parameter = 0.01 #how far away to allow nearest particles to be before adding a point in between them
voxel_size = 0.5 #mm


def Geometry_Mask_pycaster(data,stl_file_path): #good for absolute particle masking, slow (uses vtk's ray intersection algorithm)
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

def Geometry_Mask_knn_optimized(data,stl_file_path,masking_nearest_neighbor_num,ray_triangle_search_width): #does its best, but will miss some outside particles, and discard some inside particles, is pretty fast
	points = data[:,0:3]
	xbounds = np.max(points[0])+(0.1*np.ptp(points[0])),np.min(points[0])-(0.1*np.ptp(points[0])) #set the bounds for the x dimension, this will be used as the extents of the raycasting segments
	data_shape = np.shape(data) #defines the shape of the input data
	print('\nLoading STL mask geometry...')
	masking_mesh = mesh.Mesh.from_file(stl_file_path) #loads the stl file as a series of triangles defined by three points (shape is N triangles by 9 total coordinate values)
	stl_shape = np.shape(masking_mesh)
	centroids = np.transpose([((masking_mesh[:,0]+masking_mesh[:,3]+masking_mesh[:,6])/3),((masking_mesh[:,1]+masking_mesh[:,4]+masking_mesh[:,7])/3),((masking_mesh[:,2]+masking_mesh[:,5]+masking_mesh[:,8])/3)]) #use the centroids of each triangle to find the closest faces to any given ray, rather than the vertices, which start at random locations on the triangles
	print('\nTraining KNN on mask geometry vertex points...')
	mesh_neighbors = NearestNeighbors(n_neighbors=masking_nearest_neighbor_num,algorithm='auto',n_jobs=-1).fit(centroids[:,1:3]) #run a knn algorithm on all the triangle centroids, but only on their y and z coordinates, so we can find the nearest triangles to a given line, rather than a given point
	vertexdistance,vertexindex = mesh_neighbors.kneighbors(points[:,1:3]) #find the distances and indices of the point data with relation to the triangle faces of the mesh (again, in y and z plane)
	print('\nFinding points inside geometry...')
	mask_indices = np.zeros(data_shape[0])
	triangle_matrix_shape = np.shape(masking_mesh[vertexindex[0,:]])
	for point in range(data_shape[0]):
		ray_segment = np.hstack((points[point,:],xbounds[0], points[point,1:3])) #define the line segment for a given point as a ray heading in the positive x direction from any given point, terminating at the bounding box defined before
		triangles_for_intersect_testing = masking_mesh[vertexindex[point,:]] #find the n nearest triangles to this line
		intersection_number = 0
		for triangles in range(triangle_matrix_shape[0]):
			if vertexdistance[point,triangles] < ray_triangle_search_width: #if any triangle centroid is farther away from the line than a given parameter, discard it. This cuts down on processing steps. Make it too small and skew triangles will be excluded, but they might end up being the ones that the ray passes through, so be careful
				if (Ray_Triangle_Intersection(ray_segment[0:3],ray_segment[3:6],np.reshape(triangles_for_intersect_testing[triangles,:],[3,3])) > 0): #count the number of times that a ray intersects the triangles of the mesh. This step would take forever, however with the knn algorithm and distance filter we have filtered out all the non-relevant triangles
					intersection_number+=1
		if intersection_number % 2 != 0: #modulus 2, if the number of intersections is odd, that means that the point lies inside of the geometry
			mask_indices[point] = 1
			print('point #',point,'is inside')
	masked_data = data[(mask_indices != 0),:] #pull all points that passed the test into a new matrix
	return masked_data


def Mode_Filtering(data,stat_filtering_nearest_neighbor_num,stdev_threshold_multiplier):
	points = data[:,0:3]
	data_shape = np.shape(data)
	print('\nTraining KNN on Masked Pointcloud...')
	neighbors = NearestNeighbors(n_neighbors=stat_filtering_nearest_neighbor_num, algorithm='auto').fit(points) #run a knn on all the points in the pointcloud
	distances,indices = neighbors.kneighbors(points) #use the knn results back on the same pointcloud, generating a group of nearest neighboring points for every point in the pointcloud
	print('\nStatistical Analysis...')
	velocity_std_dev = stdev_threshold_multiplier*np.std(data[indices,3:6],axis=1) #find the standard deviation of the velocity data for individual x,y,z components
	velocity_mode = np.empty((data_shape[0],3))
	sorted_point_velocities=np.empty(stat_filtering_nearest_neighbor_num)
	Differences = np.empty(stat_filtering_nearest_neighbor_num-1)
	for point in range(data_shape[0]):
		for dimension in range(3):
			sorted_point_velocities = sorted(data[indices[point],3+dimension]) # sort all the velocities of all the nearest neighbors to a given point in each dimension
			Differences =[sorted_point_velocities[i+1]-sorted_point_velocities[i] for i in range(stat_filtering_nearest_neighbor_num) if i+1 < stat_filtering_nearest_neighbor_num] #find the difference between each consecutive velocity value
			velocity_mode[point,dimension] = (sorted_point_velocities[np.argmin(Differences)]+sorted_point_velocities[np.argmin(Differences)+1])/2 #assume that the mode is pretty close to the two velocities with the smallest difference between them (i.e. highest density data)
	print('\nTesting Points...')
	passed,failed = 0,0
	pass_index = np.empty(data_shape[0],dtype=int)
	for i in range(int(data_shape[0])):
		if data[i,3] > (velocity_mode[i,0]-velocity_std_dev[i,0]) and data[i,3] < (velocity_mode[i,0]+velocity_std_dev[i,0]) and data[i,4] > (velocity_mode[i,1]-velocity_std_dev[i,1]) and data[i,4] < (velocity_mode[i,1]+velocity_std_dev[i,1]) and data[i,5] > (velocity_mode[i,2]-velocity_std_dev[i,2]) and data[i,5] < (velocity_mode[i,2]+velocity_std_dev[i,2]): # the data that passes must lie close enough to the mode in every dimension that it is within n standard deviation, n defined by user
			pass_index[i] = i
			passed+=1
		else:
			pass_index[i] = 0
			failed+=1
	pass_indexed = pass_index[pass_index!=0]
	filtered_data=data[pass_indexed,:] #use the passindex values to generate a new matrix with only data that passed the mode-filtering bounds
	return filtered_data

def Median_Filtering(data,stat_filtering_nearest_neighbor_num,stdev_threshold_multiplier):
	points = data[:,0:3]
	data_shape = np.shape(data)
	print('\nTraining KNN on Masked Pointcloud...')
	neighbors = NearestNeighbors(n_neighbors=stat_filtering_nearest_neighbor_num, algorithm='auto',n_jobs=-1).fit(points) #run a knn on all the points in the pointcloud
	distances,indices = neighbors.kneighbors(points) #use the knn results back on the same pointcloud, generating a group of nearest neighboring points for every point in the pointcloud
	print('\nStatistical Analysis...')
	velocity_std_dev = stdev_threshold_multiplier*np.std(data[indices,3:6],axis=1) #find the standard deviation of the velocity data for individual x,y,z components
	velocity_median = np.median(data[indices,3:6],axis=1) #find the median of the velocity data for individual x,y,z components
	print('\nTesting Points...')
	passed,failed = 0,0
	pass_index = np.empty(data_shape[0],dtype=int)
	for i in range(int(data_shape[0])):
		if data[i,3] > (velocity_median[i,0]-velocity_std_dev[i,0]) and data[i,3] < (velocity_median[i,0]+velocity_std_dev[i,0]) and data[i,4] > (velocity_median[i,1]-velocity_std_dev[i,1]) and data[i,4] < (velocity_median[i,1]+velocity_std_dev[i,1]) and data[i,5] > (velocity_median[i,2]-velocity_std_dev[i,2]) and data[i,5] < (velocity_median[i,2]+velocity_std_dev[i,2]): # the data that passes must lie close enough to the median in every dimension that it is within n standard deviation, n defined by user
			pass_index[i] = i
			passed+=1
		else:
			pass_index[i] = 0
			failed+=1
	pass_indexed = pass_index[pass_index!=0]
	filtered_data=data[pass_indexed,:] #use the passindex values to generate a new matrix with only data that passed the mode-filtering bounds
	return filtered_data

def Fill_Sparse_Areas(data,sparse_filling_nearest_neighbor_num,min_distance_parameter):
	points = data[:,0:3]
	data_shape = np.shape(data)
	print('\nTraining KNN for Sparsity Filling...')
	neighbors = NearestNeighbors(n_neighbors=sparse_filling_nearest_neighbor_num, algorithm='auto',n_jobs=-1).fit(points)
	distances,indices = neighbors.kneighbors(points)
	print('\nFilling Sparse Areas')
	addable_number = np.shape(indices[indices>min_distance_parameter])
	added_data = np.empty([data_shape[0]*sparse_filling_nearest_neighbor_num,10])
	for point in range(data_shape[0]):
		for index in range(sparse_filling_nearest_neighbor_num):
			if indices[point,index] > min_distance_parameter:
				added_data[(point+index)] = ((points[point,0]+points[indices[point,index]])/2),((points[point,1]+points[indices[point,index]])/2),((points[point,2]+points[indices[point,index]])/2),((data[point,3]+data[indices[point,index],3])/2),((data[point,4]+data[indices[point,index],4])/2),((data[point,5]+data[indices[point,index],5])/2),0,0,0,0
				print('added point between',point,'and',indices[point,index])
	total_data = np.append(data,added_data[np.sum(added_data,axis=1)!=0],axis=1)
	return total_data

def Generate_Structured_Data(data,voxel_size):
	data_bounds = [np.min(data[:,0]),np.max(data[:,0]),np.min(data[:,1]),np.max(data[:,1]),np.min(data[:,2]),np.max(data[:,2])]
	x_kernel_bounds = np.arange(data_bounds[0],data_bounds[1]+voxel_size,voxel_size)
	y_kernel_bounds = np.arange(data_bounds[2],data_bounds[3]+voxel_size,voxel_size)
	z_kernel_bounds = np.arange(data_bounds[4],data_bounds[5]+voxel_size,voxel_size)
	grid_shape = np.shape(np.meshgrid(x_kernel_bounds,y_kernel_bounds,z_kernel_bounds))
	grid = np.vstack(np.meshgrid(x_kernel_bounds,y_kernel_bounds,z_kernel_bounds)).reshape(3,-1).T
	print(np.shape(grid),len(grid),np.shape(range(len(grid))))
	neighbors = NearestNeighbors(n_neighbors=50, algorithm='auto',n_jobs=-1).fit(data[:,0:3]) #run a knn on all the points in the pointcloud
	distances,indices = neighbors.kneighbors(grid)
	velocity_grid = np.empty(np.shape(grid))
	for gridpoint in range(len(grid)):
		kernel_data = data[indices[gridpoint,distances[gridpoint,:]<(voxel_size*(1.8/2))],3:6]
		if len(kernel_data) >= 1:
			velocity_grid[gridpoint,:] = np.mean(kernel_data,axis=0)
		else:
			velocity_grid[gridpoint,:] = np.array([0,0,0])
		print(gridpoint)
	structured_grid = np.append(grid,velocity_grid,axis=1)
	return structured_grid, grid_shape
def Read_PV3D_Files(pv3d_folder_path):
	file_names = fnmatch.filter(sorted(os.listdir(pv3d_folder_path)),'*pv3d')
	print('reading '+file_names[0])
	data = np.genfromtxt(pv3d_folder_path+file_names[0], dtype=np.float64, delimiter=',',skip_header=1)
	for file in range(1,len(file_names)):
		print('reading '+file_names[file])
		data = np.append(data,np.genfromtxt(pv3d_file_path, dtype=np.float64, delimiter=',',skip_header=1),axis=0)
	return data
def Write_PV3D_File(data,pv3d_file_name):
	print('Writing',pv3d_file_name)
	data_shape = np.shape(data)
	with open(pv3d_file_name+'.pv3d',mode='w') as output:
		output.write('Title="'+pv3d_file_name+'" VARIABLES="X","Y","Z","U","V","W","CHC","idParticleMatchA","idParticleMatchB",DATASETAUXDATA DataType="P",DATASETAUXDATA Dimension="3",DATASETAUXDATA HasVelocity="Y",DATASETAUXDATA ExtraDataNumber="2",ZONE T="T1",I='+str(data_shape[0])+',F=POINT,\n')
		for i in range(data_shape[0]):
			output.write('\n'+str(data[i,0])+', '+str(data[i,1])+', '+str(data[i,2])+', '+str(data[i,3])+', '+str(data[i,4])+', '+str(data[i,5])+', '+str(data[i,6])+', '+str(data[i,7])+', '+str(data[i,8])+',')

def Write_PLY_File(data,ply_file_name):
	print('Writing',ply_file_name)
	data_shape = np.shape(data)
	with open(ply_file_name, mode='w') as output:
		output.write('ply\nformat ascii 1.0\nelement vertex '+str(data_shape[0])+'\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nend_header\n')
		for i in range(data_shape[0]):
			output.write('\n'+str(data[i,0])+' '+str(data[i,1])+' '+str(data[i,2])+' '+str(data[i,3])+' '+str(data[i,4])+' '+str(data[i,5])+'\n')

print('Loading pv3d data...')
data = np.genfromtxt(pv3d_file_path, dtype=np.float64, delimiter=',',skip_header=1)

#data = Geometry_Mask_knn_optimized(data,stl_file_path,masking_nearest_neighbor_num,ray_triangle_search_width)
#Write_PV3D_File(data,'MaskedData')
data = Median_Filtering(data,stat_filtering_nearest_neighbor_num,stdev_threshold_multiplier)
Write_PV3D_File(data,'FilteredData')
#data = Fill_Sparse_Areas(data,sparse_filling_nearest_neighbor_num,min_distance_parameter)
#Write_PV3D_File(data,'SparseFilledData')
#data,grid_shape = Generate_Structured_Data(data,voxel_size)
#print(grid_shape[1:] )
#Write_PLY_File(data,'StructuredData')