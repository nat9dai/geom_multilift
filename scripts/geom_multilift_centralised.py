#!/usr/bin/env python

import rospy
from tf.transformations import quaternion_from_matrix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_msgs.msg import Int32, Float64
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion
from mavros_msgs.msg import AttitudeTarget, PositionTarget

from scipy.spatial.transform import Rotation

import numpy as np
import pathlib, atexit
from datetime import datetime
from utils.get_data_new import *
from utils.matrix_utils import *

# import components
from components.stop_trajectory import StopTrajPlanner
from components.second_order_butterworth import SecondOrderButterworth
from components.acc_kalman_filter import AccKalmanFilter

NUM_DRONES = 6  # default number of drones
CTRL_HZ = 100.0  # default control frequency [Hz]
DATA_HZ = 100.0

# states
INIT   = 0
ARMING = 1
TAKEOFF= 2
GOTO   = 3
HOVER  = 4
TRAJ   = 5
END_TRAJ   = 6
LAND   = 7

log_dir = pathlib.Path.home() / "lift_log"
log_dir.mkdir(exist_ok=True)
log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = log_dir / f"pd_errors_{log_time}.npz"

EPS = 1e-9 

# wrapper for external calls
polyval_stop = lambda coeff, tau: (
    coeff.polyval(tau) if isinstance(coeff, StopTrajPlanner) else (None, None, None)
)

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

class GeometricControl:
    def __init__(self, num_drones: int = NUM_DRONES) -> None:
        """
        Subscribes to:
        /payload_odom
        /simulation/position_drone_i  (PoseStamped, i=1..N)
        {ns}/mavros/local_position/pose
        {ns}/mavros/local_position/velocity
        {ns}/mavros/imu/data
        {ns}/state/state_drone_i  (Int32, i=1..N)

        Publishes to:
        /ros_geom_state/payload_q
        /ros_geom_state/sim_time
        """
        self.num_drones = num_drones

        # simulation time
        self.sim_time = None

        #  payload state 
        self.payload_pos     = None        # position (3,)
        self.prev_payload_pos = None
        self.payload_q      = None         # ENU quaternion
        self.payload_R       = None        # rotation (3,3); from {B} to {I}
        self.prev_payload_R   = None
        self.payload_vel     = None        # linear velocity (3,)
        self.payload_ang_v   = None        # body-frame _Omega_ (3,)
        self.payload_lin_acc = None        # filtered linear acc (3,)

        # accel estimation helpers
        self.prev_payload_vel      = None
        self.prev_vel_time         = None
        self.kf                    = AccKalmanFilter(dt_init=1.0 / CTRL_HZ)
        # self.fof                   = FirstOrderLowPass(cutoff_hz=10.0)
        self.fof = SecondOrderButterworth(cutoff_hz=20.0)

        #  drone state (lists) 
        self.n              = self.num_drones
        self.drone_pos      = [None] * self.num_drones
        self.drone_R        = [None] * self.num_drones
        self.drone_vel      = [None] * self.num_drones
        self.drone_omega    = [None] * self.num_drones
        self.drone_lin_acc  = [None] * self.num_drones
        self.drone_state    = [None] * self.num_drones

        self.ready = False
        self.T_enu2ned = np.array([[0, 1, 0],
                     [1, 0, 0],
                     [0, 0, -1]])
        self.T_enu2flu = np.array([[0, -1, 0],
                     [1, 0, 0],
                     [0, 0, 1]])
        self.T_flu2frd = np.diag([1, -1, -1])
        self.T_trans = self.T_enu2flu @ self.T_flu2frd
        self.T_body = np.array([[0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]])
        self.T_xy = np.array([[0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]])

        # Publishers
        self.pub_att = [
            rospy.Publisher(
            ("" if i == 0 else f"/px4_{i}") + "/mavros/setpoint_raw/attitude",
            AttitudeTarget,
            queue_size=1
            )
            for i in range(self.num_drones)
        ]
        self.pub_pos = [
            rospy.Publisher(
                ("" if i == 0 else f"/px4_{i}") + "/mavros/setpoint_raw/local",
                PositionTarget,
                queue_size=1
            )
            for i in range(self.num_drones)
        ]
        self.pub_cmd = [
            rospy.Publisher(
                f"/state/command_drone_{i}",
                Int32,
                queue_size=1
            )
            for i in range(self.num_drones)
        ]

        # Delete all pubs below later
        # Pub
        self.pub_payload_q = rospy.Publisher('/ros_geom_state/payload_q', Quaternion, queue_size=1)
        self.pub_sim_time = rospy.Publisher('/ros_geom_state/sim_time', Float64, queue_size=1)
        self.pub_payload_odom = rospy.Publisher('/ros_geom_state/payload_odom', Odometry, queue_size=1)
        self.pub_payload_acc = rospy.Publisher('/ros_geom_state/payload_acc', Imu, queue_size=1)
        # Pub num_drones odoms
        self.pub_drone_odom = [rospy.Publisher(f'/ros_geom_state/drone_{i}/odom', Odometry, queue_size=1) for i in range(num_drones)]
        self.pub_drone_acc = [rospy.Publisher(f'/ros_geom_state/drone_{i}/acc', Imu, queue_size=1) for i in range(num_drones)]
        self.pub_drone_state = [rospy.Publisher(f'/ros_geom_state/drone_{i}/state', Int32, queue_size=1) for i in range(num_drones)]

        # Subscribers
        rospy.Subscriber('/payload_odom', Odometry, self._cb_payload_odom, queue_size=1)
        # Drone pose in sim-frame
        for i in range(1, num_drones + 1):
            topic = f"/simulation/position_drone_{i}"
            rospy.Subscriber(
                topic,
                PoseStamped,
                lambda msg, idx=i-1: self._cb_drone_pose(msg, idx),
                queue_size=1
            )
        # PX4 local-position & odometry per drone
        for idx in range(self.num_drones):
            lp_topic = '/mavros/local_position/pose' if idx == 0 \
                        else '/px4_{}/mavros/local_position/pose'.format(idx)
            
            lv_topic = '/mavros/local_position/velocity' if idx == 0 \
                        else '/px4_{}/mavros/local_position/velocity'.format(idx)

            rospy.Subscriber(
                lp_topic,
                PoseStamped,
                lambda msg, i=idx: self._cb_drone_local_pose(msg, i),
                queue_size=1
            )
            rospy.Subscriber(
                lv_topic,
                TwistStamped,
                lambda msg, i=idx: self._cb_drone_local_velocity(msg, i),
                queue_size=1
            )

            #acc_topic = '/mavros/local_position/accel' if idx == 0 \
            #             else '/px4_{}/mavros/local_position/accel'.format(idx)
            acc_topic = '/mavros/imu/data' if idx == 0 \
                         else '/px4_{}/mavros/imu/data'.format(idx)
            rospy.Subscriber(
                acc_topic,
                Imu,
                lambda msg, i=idx: self._cb_drone_acceleration(msg, i),
                queue_size=1
            )

            state_topic = '/state/state_drone_{}'.format(idx)
            rospy.Subscriber(
                state_topic,
                Int32,
                lambda msg, i=idx: self._cb_drone_state(msg, i),
                queue_size=1
            )

            setpoint_topic = '/mavros/setpoint_raw/target_local' if idx == 0 \
                                else '/px4_{}/mavros/setpoint_raw/target_local'.format(idx)
            rospy.Subscriber(
                setpoint_topic,
                PositionTarget,
                lambda msg, i=idx: self._cb_drone_local_position_setpoint(msg, i),
                queue_size=1
            )
        self.drone_acc_sp  = [None] * num_drones

        # create_timer
        # self.heartbeat = rospy.Timer(rospy.Duration(1.0), self._heartbeat)


        #--- trajectory ----------------------------------------------
        self.trajectory_ENU = DataLoader()  # get the offline trajectory
        

        #--- physical parameters (edit for real hardware) ------------
        self.payload_m = 1.50
        self.m_drones = 0.250      # drone mass    [kg]
        # self.max_thrust = 2.58 * self.m_drones * 9.81    # for 1kg drone
        self.max_thrust = 8.0 * self.m_drones * 9.81  # max thrust per drone [N] for 250g drone
        self.l    = 1.0         # cable length   [m]
        self.g    = np.array([0.0, 0.0, 9.81])

        for i in range(NUM_DRONES):
            angle = 2 * np.pi * i / NUM_DRONES
            y = self.payload_r * np.cos(angle)
            x = self.payload_r * np.sin(angle)
            self.rho.append(np.array([x, y, 0.0]))
            y_offset = self.offset_r * np.cos(angle)
            x_offset = self.offset_r * np.sin(angle)
            self.offset_pos.append(np.array([x_offset, y_offset, 0.0]))
        self.rho = np.array(self.rho)  # shape (num_drones, 3)
        self.offset_pos = np.array(self.offset_pos)  # shape (num_drones, 3)

        I3 = np.eye(3)
        self.P = np.zeros((6, 3*self.num_drones))           
        for i in range(self.n):
            self.P[0:3, 3*i:3*(i+1)] = I3
            self.P[3:6, 3*i:3*(i+1)] = hat(self.rho[i])

        # timer
        self.dt_nom = 1.0 / CTRL_HZ
        self.dt      = self.dt_nom  
        self.t_prev = None        
        self.t0     = None  
        self.sim_t_prev = None
        self.sim_t0 = None   
        self.traj_t0 = None          
        self.traj_duration = 6.0  
        self.x_start = None 
        self.t_wait_traj = 0.0
        # This is a ROS2 thing
        #self.create_timer(self.dt_nom, self._step)
        self.step = rospy.Timer(rospy.Duration(self.dt_nom), self._step)

        rospy.loginfo("GeomLiftCtrl started.")

        # desired values initialization
        self.x_id = np.zeros((self.n, 3))
        self.v_id = np.zeros((self.n, 3))
        self.mu_id = np.zeros((self.n, 3))  # desired force on each drone
        self.q_id = np.zeros((self.n, 3))    # desired cable direction
        self.omega_id = np.zeros((self.n, 3))  # desired angular velocity of each drone
        self.omega_id_dot = np.zeros((self.n, 3))  # desired angular acceleration of each drone
        self.b1d = np.array([1, 0, 0])  


        # first order lowpass
        # self.mu_id_dot_f = FirstOrderLowPass(cutoff_hz=8.0)
        # self.mu_id_ddot_f = FirstOrderLowPass(cutoff_hz=8.0)
        # self.Omega_0_dot_f = FirstOrderLowPass(cutoff_hz=10.0)
        # self.mu_f = [FirstOrderLowPass(cutoff_hz=20.0)   for _ in range(self.n)]
        # second order butterworth filter
        self.mu_id_dot_f = SecondOrderButterworth(cutoff_hz=10.0)
        self.mu_id_ddot_f = SecondOrderButterworth(cutoff_hz=10.0)
        self.Omega_0_dot_f = SecondOrderButterworth(cutoff_hz=18.0)
        self.mu_f = [SecondOrderButterworth(cutoff_hz=20.0)   for _ in range(self.n)]

        #--- control gains (PD only ) --------------------
        self.kq = 10.0
        self.kw = 3.20
        self.z_weight = 0.380    # weight for the geometric control z axis


        self.k_ddp = np.zeros((6,13))
        # self.alpha = 0.0       # DDP feedback gain 
        self.alpha = 0.10

        # self.slowdown = 1.25     # for test only, no slowdown
        self.slowdown = 1.0     # no slowdown

        self.thrust_bias = 0.0

        self.time_switch = 6.0

        self.x_id_final = np.zeros((self.n, 3))  # final desired position of each drone
        q_init = [1.0, 0.0, 0.0, 0.0]
    
        self.drone_q_prev = [None] * self.n  # previous drone quaternions

        z_column   = np.full((NUM_DRONES, 1), -0.80)  # (N, 1)
        self.dirs_final = np.hstack((self.rho[:, :2], z_column))   # (N, 3)

        for i in range(self.n):
            self.drone_q_prev[i] = q_init  # previous drone quaternions
            # self.x_id_final[i] = self.T_enu2ned @ self.trajectory_ENU.payload_x[-1, :] + self.rho[i] + self.l * self.dirs_final[i] / np.linalg.norm(self.dirs_final[i]) - self.offset_pos[i]
            self.x_id_final[i] = self.T_enu2ned @ self.trajectory_ENU.payload_x[-1, :] + self.rho[i] + self.l * self.T_enu2ned @ self.trajectory_ENU.cable_direction[i, -1, :] - self.offset_pos[i]  # final desired position of each drone
        print(f"Final desired position of each drone: {self.x_id_final}")
        # flag
        self.traj_ready = False
        self.traj_done = False
        self.traj_done_bit = np.zeros(self.n, dtype=bool)
        
        self.t_log = None

        self.log = {
            't' : [],       
            'ex': [],       
            'ev': [],      
            'eR': [],       
            'eO': [],       
            'eq': [],       
            'ew': [],       
            "q_ref": [], "q_act": [],  
            "w_ref": [], "w_act": [],  
        }

        atexit.register(self._save_log)


        rospy.loginfo('ROSGeomState running with {} drones.'.format(self.num_drones))

    #  Callbacks
    def _cb_drone_local_position_setpoint(self, msg: PositionTarget, idx: int) -> None:
        # ENU -> NED
        self.drone_acc_sp[idx] = np.array([msg.acceleration_or_force.y, msg.acceleration_or_force.x, -msg.acceleration_or_force.z], float)

    def _cb_payload_odom(self, msg: Odometry) -> None:
        """Handle /payload_odom, extract pose, velocity, angular velocity."""
        # now = self.get_clock().now().nanoseconds * 1e-9 # ROS2

        # Pose -> position & rotation matrix
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.payload_q = q

        # tf: ENU -> NED 
        self.payload_pos = np.array([p.y, p.x, -p.z], float)

        # FRD -> BODY -> ENU -> NED
        self.payload_R   = self.T_enu2ned @ Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix() @ self.T_trans
        self.sim_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        now = self.sim_time
        self.pub_sim_time.publish(self.sim_time)

        # use the sim time for kalman update 
        # dt = 1/CTRL_HZ if self.prev_vel_time is None else max(1e-4, now - self.prev_vel_time)
        if self.prev_vel_time is None:
            dt = 1.0 / CTRL_HZ
        elif now != self.prev_vel_time and self.prev_vel_time is not None:
            dt = max(1e-5, now - self.prev_vel_time)
            self.prev_vel_time = now
        

        # finite-difference acceleration + Kalman smoother 
        self.kf.step(self.payload_pos, dt)
        self.payload_vel     = self.kf.vel.copy()
        self.payload_lin_acc = self.kf.acc.copy()

        # get the angular velocity of the payload
        if self.prev_payload_R is not None and self.prev_vel_time is not None:
            payload_ang_v = body_angular_velocity(self.prev_payload_R, self.payload_R, dt)
            self.payload_ang_v = self.fof(payload_ang_v, dt)
        else:
            self.payload_ang_v = np.zeros(3)

        self.prev_vel_time = now
        self.prev_payload_pos = self.payload_pos
        self.prev_payload_R = self.payload_R
        self._check_ready()

    def _cb_drone_pose(self, msg: PoseStamped, idx: int) -> None:
        p = msg.pose.position
        # ENU -> NED
        self.drone_pos[idx] = np.array([p.y, p.x, -p.z], float)
        self._check_ready()

    def _cb_drone_local_pose(self, msg: PoseStamped, idx: int) -> None:
        # ENU -> NED
        q = msg.pose.orientation
        R_enu = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        self.drone_R[idx] = self.T_enu2ned @ R_enu @ self.T_enu2ned.T
        self._check_ready()

    def _cb_drone_local_velocity(self, msg: TwistStamped, idx: int) -> None:
        # ENU -> NED
        self.drone_vel[idx] = np.array([msg.twist.linear.y, msg.twist.linear.x, -msg.twist.linear.z], float)
        self.drone_omega[idx] = np.array([msg.twist.angular.y, msg.twist.angular.x, -msg.twist.angular.z], float)
        self._check_ready()

    def _cb_drone_acceleration(self, msg: Imu, idx: int) -> None:
        # ENU -> NED
        self.drone_lin_acc[idx] = np.array([ msg.linear_acceleration.y,  msg.linear_acceleration.x, -msg.linear_acceleration.z], float)
        self._check_ready()

    def _cb_drone_state(self, msg: Int32, idx: int) -> None:
        state = msg.data
        self.drone_state[idx] = state
        self._check_ready()

    def _check_ready(self) -> None:
        """Mark node 'ready' once every critical field has been set."""
        payload_ready = all(val is not None for val in (
            self.payload_pos, self.payload_R, self.payload_vel,
            self.payload_ang_v, self.payload_lin_acc
        ))
        drones_ready = all(
            p is not None and v is not None and o is not None and a is not None
            for p, v, o, a in zip(
                self.drone_pos, self.drone_vel, self.drone_omega, self.drone_lin_acc
            )
        )
        self.ready = payload_ready and drones_ready

    def _heartbeat(self) -> None:
        """Periodic log of payload state once ready."""
        if not self.ready:
            return
        t = rospy.Time.now().to_sec()
        rospy.loginfo(
            f'{t:2.4f}s payload position {self.payload_pos} vel {self.payload_vel}'
        )

    # Public accessor
    # To be deleted after migrating all nodes to ROS2
    def get_state(self) -> dict | None:
        """Return deep-copied dict with payload & drone states, or None if not ready."""
        if not self.ready:
            return None
        
        payload = {
            'pos':     self.payload_pos.copy(),
            'R':       self.payload_R.copy(),
            'vel':     self.payload_vel.copy(),
            'lin_acc': self.payload_lin_acc.copy(),
            'omega':   self.payload_ang_v.copy(),
        }



        drones = []
        for i, (p, R, v, o, a) in enumerate(zip(
            self.drone_pos, self.drone_R,
            self.drone_vel, self.drone_omega,
            self.drone_lin_acc,
        )):
            drones.append({
                'pos':     p.copy(),
                'R':       R.copy(),
                'vel':     v.copy(),
                'omega':   o.copy(),
                'lin_acc': a.copy(),
                'state':   self.drone_state[i],   
            })
        return {'payload': payload, 'drones': drones}
    
    # Delete later
    def pub_snapshot(self) -> None:
        # Pub this instead of calling get_state [ROS1]
        if not self.ready:
            return None
        
        # Payload
        po = Odometry()
        po.pose.pose.position.x = self.payload_pos[0] # NED
        po.pose.pose.position.y = self.payload_pos[1]
        po.pose.pose.position.z = self.payload_pos[2]
        po.pose.pose.orientation = R_to_quat(self.payload_R)
        po.twist.twist.linear.x = self.payload_vel[0]  # NED
        po.twist.twist.linear.y = self.payload_vel[1]
        po.twist.twist.linear.z = self.payload_vel[2]
        po.twist.twist.angular.x = self.payload_ang_v[0]  # body frame
        po.twist.twist.angular.y = self.payload_ang_v[1]
        po.twist.twist.angular.z = self.payload_ang_v[2]
        p_acc = Imu()
        p_acc.linear_acceleration.x = self.payload_lin_acc[0]
        p_acc.linear_acceleration.y = self.payload_lin_acc[1]
        p_acc.linear_acceleration.z = self.payload_lin_acc[2]
        p_acc.angular_velocity = R_to_quat(self.payload_ang_v)
        self.pub_payload_odom.publish(po)
        self.pub_payload_acc.publish(p_acc)

        # Drones
        for i in range(self.num_drones):
            if self.drone_pos[i] is None or self.drone_R[i] is None:
                continue
            do = Odometry()
            do.pose.pose.position.x = self.drone_pos[i][0]  # NED
            do.pose.pose.position.y = self.drone_pos[i][1]
            do.pose.pose.position.z = self.drone_pos[i][2]
            do.pose.pose.orientation = R_to_quat(self.drone_R[i])
            do.twist.twist.linear.x = self.drone_vel[i][0]  # NED
            do.twist.twist.linear.y = self.drone_vel[i][1]
            do.twist.twist.linear.z = self.drone_vel[i][2]
            do.twist.twist.angular.x = self.drone_omega[i][0]  # body frame
            do.twist.twist.angular.y = self.drone_omega[i][1]
            do.twist.twist.angular.z = self.drone_omega[i][2]
            self.pub_drone_odom[i].publish(do)

            da = Imu()
            da.linear_acceleration.x = self.drone_lin_acc[i][0]
            da.linear_acceleration.y = self.drone_lin_acc[i][1]
            da.linear_acceleration.z = self.drone_lin_acc[i][2]
            da.angular_velocity = R_to_quat(self.drone_omega[i])
            self.pub_drone_acc[i].publish(da)

            self.pub_drone_state[i].publish(Int32(self.drone_state[i]))

    def _save_log(self):
        if not self.log['t']:           
            return
        np.savez(
            LOG_PATH,
            t  = np.asarray(self.log['t']),
            ex = np.asarray(self.log['ex']),
            ev = np.asarray(self.log['ev']),
            eR = np.asarray(self.log['eR']),
            eO = np.asarray(self.log['eO']),
            eq = np.asarray(self.log['eq']),
            ew = np.asarray(self.log['ew']),
            q_ref = np.asarray(self.log['q_ref']),
            q_act = np.asarray(self.log['q_act']),
            w_ref = np.asarray(self.log['w_ref']),
            w_act = np.asarray(self.log['w_act']),
        )
        rospy.loginfo(f"Log saved in: {LOG_PATH}")

    def _desired(self, t: float) -> None:
        idx = int(round(t / ((1/DATA_HZ) * self.slowdown)))
        idx = np.clip(idx, 0, self.trajectory_ENU.cable_direction.shape[1] - 2)
        idx_len = self.trajectory_ENU.cable_direction.shape[1] - 2
        state_idx =  idx 
        dirs_enu =  self.trajectory_ENU.cable_direction[:, state_idx, :]   # shape: (6, 3), xl - xq
        mu_enu = self.trajectory_ENU.cable_mu[:, state_idx]                  # shape: (6,) scalar
        omega_enu = self.trajectory_ENU.cable_omega[:, state_idx, :]
        omega_enu_dot = self.trajectory_ENU.cable_omega_dot[:, idx, :]

        # DDP feedback gain
        self.k_ddp = self.trajectory_ENU.Kb[idx]
        # desired attitude
        payload_q_d_ENU = self.trajectory_ENU.payload_q[state_idx, :]   # w, x, y, z
        payload_q_ENU = [self.payload_q.w, self.payload_q.x, self.payload_q.y, self.payload_q.z]    # x, y, z, w
        R_ENU_d = self.trajectory_ENU.quat_2_rot(payload_q_d_ENU)   # {B} 2 {I}
        # self.R_d = self.T_body @ R_ENU_d @ self.T_enu2ned
        self.R_d = self.T_body @ np.eye(3) @ self.T_enu2ned
        # self.Omega_0_d = self.T_body @ self.trajectory_ENU.payload_w[state_idx, :]  # (3,) under FRD frame
        self.Omega_0_d = np.zeros(3)
        Omega_0_d_ENU = self.trajectory_ENU.payload_w[state_idx, :]   # BODY frame
        self.Omega_0_d_hat = hat(self.Omega_0_d)  # (3, 3) skew-symmetric matrix
        self.R_d_dot = self.Omega_0_d_hat @ self.R_d

        self.x_d = self.T_enu2ned @ self.trajectory_ENU.payload_x[state_idx, :]  # NED frame
        self.v_d = self.T_enu2ned @ self.trajectory_ENU.payload_v[state_idx, :]  # NED frame

        # #DEBUG: test the kq and kw
        # self.x_d = np.array([0.0, 0.0, -0.0])
        # self.v_d = np.zeros(3)
        # self.k_ddp = np.zeros((6, 13))
        # self.R_d = np.eye(3)
        # if t >= 4.0 * self.slowdown:
        #     self.x_d = self.T_enu2ned @ self.trajectory_ENU.payload_x[99, :]
        #     self.v_d = np.array([0.0, 0.0, 0.0])  # desired velocity
        # self.a_d = self.T_enu2ned @ self.trajectory_ENU.payload_a[idx, :]  
        # self.j_d = np.array([0.0, 0.0, 0.0])  # desired jerk

        # error (for original geometric control)
        self.e_x = self.x_0 - self.x_d  # NED frame
        self.e_v = self.v_0 - self.v_d  # velocity error
        e_x_ENU = self.T_enu2ned @ self.e_x # ENU frame
        e_v_ENU = self.T_enu2ned @ self.e_v
        e_q_ENU = payload_q_ENU - payload_q_d_ENU   # NOTE: use the subtraction of quaternions directly
        self.e_R_0 = 0.5 * vee(self.R_d.T @ self.R_0 - self.R_0.T @ self.R_d )  # rotation error
        self.e_Omega_0 = self.Omega_0 - self.R_0.T @ self.R_d @ self.Omega_0_d  # angular velocity error
        e_Omega_0_ENU = self.T_body @ self.Omega_0 - Omega_0_d_ENU
        e_ddp_ENU = np.concatenate((e_x_ENU, e_v_ENU, e_q_ENU, e_Omega_0_ENU))
        
        FM_BODY = self.alpha * self.k_ddp @ e_ddp_ENU   # need to make sure that this should under the ENU frame
        F_BODY = FM_BODY[0:3]
        M_BODY = FM_BODY[3:6]
        F_FRD = self.T_body @ F_BODY
        M_FRD = self.T_body @ M_BODY
        FM_FRD = np.concatenate((F_FRD, M_FRD))
        P_pseudo = self.P.T @ np.linalg.inv(self.P @ self.P.T)  # This is under NED frame
        delta_mu_FRD = P_pseudo @ FM_FRD
        
        # desired 
        for i in range(self.n):
            self.q_id[i] = self.T_enu2ned @ dirs_enu[i]    # (6, 3) q is point down
            self.mu_id[i] = mu_enu[i] * self.q_id[i] + self.R_0 @ delta_mu_FRD[i*3:i*3+3]  
            self.q_id[i] = - self.mu_id[i] / np.linalg.norm(self.mu_id[i])
            self.omega_id_dot[i] = self.T_enu2ned @ omega_enu_dot[i, :]
        if not hasattr(self, "mu_id_prev"):          # first iteration
            self.mu_id_prev = self.mu_id.copy()
            self.mu_id_dot  = np.zeros_like(self.mu_id)
        else:
            mu_id_dot  = (self.mu_id - self.mu_id_prev) / self.sim_dt
            self.mu_id_dot = self.mu_id_dot_f(mu_id_dot, self.sim_dt)
            self.mu_id_prev = self.mu_id.copy()
        
        if not hasattr(self, "mu_id_dot_prev"):          # first iteration
            self.mu_id_dot_prev = self.mu_id_dot.copy()
            self.mu_id_ddot  = np.zeros_like(self.mu_id)
        else:
            mu_id_ddot  = (self.mu_id_dot - self.mu_id_dot_prev) / self.sim_dt
            self.mu_id_ddot = self.mu_id_ddot_f(mu_id_ddot, self.sim_dt)
            self.mu_id_dot_prev = self.mu_id_dot.copy()
        proj = np.eye(3)[None,:,:] - np.einsum('ij,ik->ijk', self.q_id, self.q_id)
        self.q_id_dot = -np.einsum('ijk,ik->ij', proj, self.mu_id_dot) / np.linalg.norm(self.mu_id, axis=1, keepdims=True)
        self.omega_id = np.cross(self.q_id, self.q_id_dot)
        P_dot = - ( np.einsum('ij,ik->ijk', self.q_id_dot, self.q_id) + np.einsum('ij,ik->ijk', self.q_id,     self.q_id_dot) )
        L     = np.linalg.norm(self.mu_id,      axis=1, keepdims=True)                   # (n,1)
        L_dot = np.einsum('ij,ij->i', self.mu_id, self.mu_id_dot)[:,None] / L            # (n,1)
        term1 = np.einsum('ijk,ik->ij', P_dot,    self.mu_id_dot)                      # P' * mu_dot
        term2 = np.einsum('ijk,ik->ij', proj,        self.mu_id_ddot)                     # P  * mu_ddot
        term3 = (proj @ self.mu_id_dot[:,:,None])[:,:,0] * (L_dot / L)                    # (P*mu_dot)*(L_dot/L)
        self.q_id_ddot = - (term1 + term2) / L + term3    
        self.omega_id_dot = np.cross(self.q_id_dot, self.q_id_ddot)

        for i in range(self.n):
            self.x_id[i] = self.x_d + self.R_d @ self.rho[i] - self.l * self.q_id[i] - self.offset_pos[i]
            self.v_id[i] = (self.v_d + self.R_d_dot @ self.rho[i] - self.l * self.q_id_dot[i])  

    def _step(self) -> None:
        #snap = self.state.get_state()
        if snap is None:
            return  # not ready yet
        
        # unuse
        #self.fsm_states = [drone["state"] for drone in snap["drones"]]
        
        t_now = rospy.Time.now().to_sec()
        self.check_state(self.sim_time)
        if not self.traj_ready:
            return
        if self.t_prev is None:            
            self.t_prev = t_now
            self.t0     = t_now             
            self.dt     = self.dt_nom
            self.x_start = snap["payload"]["pos"].copy()
        else:
            self.dt     = max(2e-2, t_now - self.t_prev)   
            self.t_prev = t_now

        t_rel = t_now - self.t0 
        
        if self.sim_t_prev is None:
            # init
            self.sim_t_prev = self.sim_time
            self.sim_dt     = self.dt_nom          # or 0
            self.x_start    = snap["payload"]["pos"].copy()
        elif self.sim_time - self.sim_t_prev > EPS:
            # update dt
            raw_dt   = self.sim_time - self.sim_t_prev
            self.sim_dt = max(4e-5, raw_dt)        
            self.sim_t_prev = self.sim_time
        sim_t_rel = self.sim_time - self.sim_t0

        #---- payload state ----
        self.x_0  = snap["payload"]["pos"]      # (3,)
        self.R_0  = snap["payload"]["R"]        # (3,3)
        self.v_0  = snap["payload"]["vel"]      # (3,)
        self.a_0 = snap["payload"]["lin_acc"]  # (3,)
        self.Omega_0  = snap["payload"]["omega"]      # (3,)
        self.Omega_0_hat = hat(self.Omega_0)  # (3,3)
        self.R_0_dot = self.R_0 @ self.Omega_0_hat
        if not hasattr(self, "omega_0_prev"):
            # first iteration → no history yet
            self.omega_0_prev = self.Omega_0.copy()
            self.omega_0_dot  = np.zeros_like(self.Omega_0)
        else:
            omega_0_dot  = (self.Omega_0 - self.omega_0_prev) / self.sim_dt
            self.omega_0_dot = self.Omega_0_dot_f(omega_0_dot, self.sim_dt)
            self.omega_0_prev = self.Omega_0.copy()
        

        #---- drone states ----
        drones = snap["drones"]
        mu = np.zeros((self.n, 3))
        a = np.zeros((self.n, 3))
        u_parallel = np.zeros((self.n, 3))
        u_vertical = np.zeros((self.n, 3))
        u = np.zeros((self.n, 3))
        b3 = np.zeros((self.n, 3))  # unit thrust vector
        self._desired(sim_t_rel)  # update desired values
        mu_id = self.mu_id
        q_d = self.q_id
        omega_id = self.omega_id
        omega_id_dot = self.omega_id_dot
        eq_step   = np.zeros((self.n, 3))
        ew_step   = np.zeros((self.n, 3))
        q_ref     = np.zeros((self.n, 3))
        q_act     = np.zeros((self.n, 3))
        w_ref     = np.zeros((self.n, 3))
        w_act     = np.zeros((self.n, 3))

        for i, drone in enumerate(drones):
            # get drone position
            x_i = drone['pos']                   # (3,)
            Omega_i = drone['omega']             # (3,)
            v_i = drone['vel']                   # (3,)
            R_i = drone['R']                     # (3,3)
            # compute raw cable vector and normalize to unit q
            vec =   - x_i + self.x_0 + self.R_0 @ self.rho[i]   # vector from attach point to drone
            q_i = vec / np.linalg.norm(vec)          # unit direction along cable
            q_i_dot = (-v_i + self.v_0 + self.R_0_dot @ self.rho[i]) / self.l
            q_i_hat = hat(q_i)  # (3,3) skew-symmetric matrix of q_i
            mu[i] = np.dot(mu_id[i], q_i) * q_i  # desired force on drone i along cable
            a[i] = (
                self.a_0 
                - self.g 
                + self.R_0 @ (self.Omega_0_hat @ (self.Omega_0_hat @ self.rho[i]))
                - self.R_0 @ hat(self.rho[i]) @ self.omega_0_dot
            )
            omega_i = np.cross(q_i, q_i_dot)
            omega_sq = np.dot(omega_i, omega_i) 
            # compute u parallel 
            u_parallel[i] = (
                mu[i]
                + self.m_drones * self.l * omega_sq * q_i
                + self.m_drones * (np.dot(q_i, a[i]) * q_i)
            )

            # compute u vertical 
            e_qi = np.cross(q_d[i], q_i)
            q_i_hat_sqr = q_i_hat @ q_i_hat  # (3,3) outer product of q_i
            e_omega_i = omega_i + q_i_hat_sqr @ omega_id[i]
            u_vertical[i] = self.m_drones * self.l * q_i_hat @ (
                - self.kq * e_qi 
                - self.kw * e_omega_i 
                - np.dot(q_i, omega_id[i]) * q_i_dot
                - q_i_hat_sqr @ omega_id_dot[i]
            ) - self.m_drones * q_i_hat_sqr @ a[i]

            u[i] = u_parallel[i] + u_vertical[i]
            if np.linalg.norm(u[i]) < 1e-3:
                continue 

            # compute thrust
            b3[i] = - u[i] / np.linalg.norm(u[i])  # unit vector along thrust
            body_z = px4_body_z(acc_sp = self.drone_acc_sp[i])
            fused_z = ((1 - self.z_weight) * body_z + self.z_weight * b3[i]) / np.linalg.norm(((1 - self.z_weight) * body_z + self.z_weight * b3[i]))
            
            # thrust
            f_i =  - u[i] @ (R_i @ np.array([0, 0, 1]))
           
            A2 = np.cross(b3[i], fused_z)
            b2c = A2 / np.linalg.norm(A2)
            A1 = np.cross(b2c, fused_z)
            b1c = A1 / np.linalg.norm(A1)
            R_ic = np.column_stack((b1c, b2c, fused_z))
            ensure_SO3(R_ic)
            t_tmp = rospy.Time.now()
            ts_us = t_tmp.secs * 1_000_000 + t_tmp.nsecs // 1000
            att = VehicleAttitudeSetpoint()
            att.timestamp = ts_us
            q_new = rotation_matrix_to_quaternion(R_ic)
            q_new = np.asarray(q_new, dtype=np.float32).reshape(4,)  
            if self.drone_q_prev[i] is None:
                self.drone_q_prev[i] = q_new.copy()                                   
            
            if np.dot(q_new, self.drone_q_prev[i]) < 0.0:
                q_new = -q_new

            att.q_d = q_new.tolist()                                  # ROS msg, list
            self.drone_q_prev[i] = q_new
            norm = - f_i / self.max_thrust - self.thrust_bias
            norm = np.clip(norm , -1.0, -0.1)
            att.thrust_body = [0.0, 0.0, norm]
            if not self.traj_done: #and  sim_t_rel < 6.0:
                self.pub_att[i].publish(att)

            traj = TrajectorySetpoint()
            traj.timestamp = ts_us
            traj.position = self.x_id[i].astype(np.float32)
            traj.velocity = self.v_id[i].astype(np.float32)
            # traj.velocity = [np.nan, np.nan, np.nan]  # velocity is not used in PX4
            # traj.acceleration = [np.nan, np.nan, np.nan]
            # traj.yaw = 0.0
            # traj.yawspeed = np.nan
            self.pub_pos[i].publish(traj)

            e_qi      = np.cross(q_d[i], q_i)          # (3,)
            eq_step[i]   = e_qi                        
            ew_step[i]   = e_omega_i                  
            q_ref[i]     = q_d[i]                      
            q_act[i]     = q_i                         
            w_ref[i]     = omega_id[i]                 
            w_act[i]     = omega_i       
            
            
        self.log["t"].append(sim_t_rel)
        self.log["ex"].append(self.e_x.copy())
        self.log["ev"].append(self.e_v.copy())
        self.log["eR"].append(self.e_R_0.copy())
        self.log["eO"].append(self.e_Omega_0.copy())

        self.log["eq"].append(eq_step.copy())        
        self.log["ew"].append(ew_step.copy())
        self.log["q_ref"].append(q_ref.copy())
        self.log["q_act"].append(q_act.copy())
        self.log["w_ref"].append(w_ref.copy())
        self.log["w_act"].append(w_act.copy()) 
        # breakpoint()


def main() -> None:
    rospy.init_node('ros_geom_state')
    node = GeometricControl(num_drones=NUM_DRONES)
    rospy.Timer(rospy.Duration(0.02), lambda event: node.pub_snapshot())
    rospy.loginfo("Initialise ros_geom_state node!")
    rospy.spin()  # Keep the node running until shutdown

if __name__ == '__main__':
    main()