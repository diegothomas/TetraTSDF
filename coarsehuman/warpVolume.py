import numpy as np
import cv2


def global_rigid_transformation(pose, J, kintree_table):

    results = {}

    pose = pose.reshape((-1,3))
    
    id_to_col = {kintree_table[1,i] : i for i in range(kintree_table.shape[1])}

    parent = {i : id_to_col[kintree_table[0,i]] for i in range(1, kintree_table.shape[1])}

    
    rodrigues = lambda x : cv2.Rodrigues(x)[0]
    rodriguesJB = lambda x : cv2.Rodrigues(x)[1]
    with_zeros = lambda x : np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))
    pack = lambda x : np.hstack([np.zeros((4, 3)), x.reshape((4,1))])
    

    results[0] = with_zeros(np.hstack((rodrigues(pose[0,:]), J[0,:].reshape((3,1)))))        

    for i in range(1, kintree_table.shape[1]):
        results[i] = results[parent[i]].dot(with_zeros(np.hstack((rodrigues(pose[i,:]),(J[i] - J[parent[i]]).reshape(3,1)))))

    Rt_Global = np.dstack([results[i] for i in sorted(results.keys())])
    Rt_A = np.dstack([results[i] - (pack(results[i].dot(np.concatenate( ( (J[i,:]), (0,) ) )))) for i in range(len(results))])
    
    return Rt_A, Rt_Global



def warpVolume(V, J, J_shapedir, pose, betas, kintree_table, weights):

    # All vertices on the Coarse mesh has 24 weights corresponding to each joints.
    # These vertices are transformed by weighted average of the transformation of each joints

    # Translate all vertices depend on betas
    J_displacements = np.sum(J_shapedir * betas, axis=2)
    # J_displacements = np.sum(J_shapedir * np.array([5,5,0,0,0,0,0,0,0,0], dtype=np.float32), axis=2) #example

    V_beta = V + np.dot(weights, J_displacements)
    J_beta = J + J_displacements

    (Rt_A, Rt_Global)= global_rigid_transformation(pose, J_beta, kintree_table)

    # Deform all vertices and joints
    T = Rt_A.dot(weights.T)
    V_beta_4dim = np.vstack((V_beta.T, np.ones((1, V_beta.shape[0]))))
    J_beta_4dim = np.vstack((J_beta.T, np.ones((1, J_beta.shape[0]))))
    V_posebeta = (T[:,0] * V_beta_4dim[0] + T[:,1] * V_beta_4dim[1] + T[:,2] * V_beta_4dim[2] + T[:,3] * V_beta_4dim[3]).T[:,0:3]
    J_posebeta = (Rt_A[:,0] * J_beta_4dim[0] + Rt_A[:,1] * J_beta_4dim[1] + Rt_A[:,2] * J_beta_4dim[2] + Rt_A[:,3] * J_beta_4dim[3]).T[:,0:3]

    return V_posebeta, J_posebeta

def unwarpVolume(V_posed, J, pose, kintree_table, weights):

    (Rt_A, Rt_Global)= global_rigid_transformation(pose, J, kintree_table)
    
    Rt_A_inv = np.dstack([np.linalg.inv(Rt_A[:,:,i]) for i in range(24)])

    # Deform all vertices and joints
    T = Rt_A_inv.dot(weights.T)
    V_posed_4dim = np.vstack((V_posed.T, np.ones((1, V_posed.shape[0]))))
    V_unposed = (T[:,0] * V_posed_4dim[0] + T[:,1] * V_posed_4dim[1] + T[:,2] * V_posed_4dim[2] + T[:,3] * V_posed_4dim[3]).T[:,0:3]

    return V_unposed
