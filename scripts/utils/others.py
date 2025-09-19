import numpy as np
from geometry_msgs.msg import Quaternion
from tf.transformations import quaternion_from_matrix
from scipy.spatial.transform import Rotation

def R_to_quat(R: np.ndarray) -> Quaternion:
    """
    Convert a 3x3 rotation matrix to a geometry_msgs/Quaternion (x,y,z,w).
    """
    assert R.shape == (3, 3)
    # Build a 4x4 homogeneous matrix for tf
    M = np.eye(4)
    M[:3, :3] = R
    x, y, z, w = quaternion_from_matrix(M)  # returns (x,y,z,w)
    q = Quaternion()
    q.x, q.y, q.z, q.w = float(x), float(y), float(z), float(w)
    return q

def cylinder_inertia(m, r_outer, h, r_inner=0.0):
    # Check validity
    if r_inner < 0 or r_inner >= r_outer:
        raise ValueError("0 ≤ r_inner < r_outer must hold.")
    
    # Solid cylinder special-case
    if r_inner == 0.0:  
        Izz = 0.5  * m * r_outer**2
        Ixx = Iyy = (1/12) * m * (3*r_outer**2 + h**2)
    else:
        # Hollow cylinder inertia: subtract inner solid from outer solid
        V_outer = np.pi * r_outer**2 * h
        V_inner = np.pi * r_inner**2 * h
        rho       = m / (V_outer - V_inner)       # uniform density
        
        m_outer = rho * V_outer
        m_inner = rho * V_inner
        
        Izz = 0.5 * (m_outer * r_outer**2 - m_inner * r_inner**2)
        Ixx = Iyy = (1/12) * (
            m_outer * (3*r_outer**2 + h**2) - 
            m_inner * (3*r_inner**2 + h**2)
        )
    return np.diag([Ixx, Iyy, Izz])

def body_angular_velocity(R_prev: np.ndarray,
                          R_curr: np.ndarray,
                          dt: float) -> np.ndarray:
    if dt <= 0.0:                   
        return np.zeros(3)

    # delta R = R_prev.T * R_curr  (SO(3))
    delta_R = R_prev.T @ R_curr
    rot_vec = Rotation.from_matrix(delta_R).as_rotvec()   # rad
    return rot_vec / dt      

def limit_tilt(body_z: np.ndarray,
               max_angle_rad: float,
               eps: float = 1e-6) -> np.ndarray:
    """
    Clamp the angle between body_z and world_z (0,0,1) to max_angle_rad.
    Returns a *unit* vector.
    """
    world_z = np.array([0., 0., 1.])
    body_z = body_z / np.linalg.norm(body_z)              # ensure unit
    dot = np.clip(np.dot(body_z, world_z), -1.0, 1.0)
    angle = np.arccos(dot)

    if angle > max_angle_rad:
        rejection = body_z - dot * world_z               # component ⟂ world_z
        if np.dot(rejection, rejection) < eps:           # parallel case
            rejection = np.array([1., 0., 0.])
        rejection /= np.linalg.norm(rejection)
        body_z = (np.cos(max_angle_rad) * world_z +
                  np.sin(max_angle_rad) * rejection)

    return body_z / np.linalg.norm(body_z)

def px4_body_z(acc_sp: np.ndarray,
               gravity: float = 9.81,
               decouple: bool = False,
               max_tilt_deg: float = 45.0) -> np.ndarray:
    """
    Re-create PX4 _accelerationControl body_z vector.
    Unit vector of the body Z-axis expressed in world (NED) frame.
    """
    g = gravity
    z_spec = -g + (0.0 if decouple else acc_sp[2])
    vec = np.array([-acc_sp[0], -acc_sp[1], -z_spec])

    if np.allclose(vec, 0):
        vec[2] = 1.0  # safety

    body_z = vec / np.linalg.norm(vec)

    body_z = limit_tilt(body_z, np.deg2rad(max_tilt_deg))
    return body_z

def quat_mul(q1, q2):
    # xyzw * xyzw
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,   # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,   # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,   # z
        w1*w2 - x1*x2 - y1*y2 - z1*z2    # w
    ], dtype=float)

def quat_conj(q):
    x,y,z,w = q
    return np.array([-x, -y, -z, w], dtype=float)

def ned_quat_to_enu(q_ned_xyzw):
    # Fixed change-of-basis: NED -> ENU
    q_T = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.0, 0.0], dtype=float)  # xyzw
    q_enu = quat_mul(quat_mul(q_T, q_ned_xyzw), quat_conj(q_T))
    # normalize (numerical safety)
    return q_enu / np.linalg.norm(q_enu)