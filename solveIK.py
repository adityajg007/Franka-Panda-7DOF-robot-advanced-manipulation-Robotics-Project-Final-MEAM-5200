import numpy as np
from math import pi, sin, cos
from scipy.linalg import null_space  # returns an orthonormal basis!

from calculateFK import FK
from calcJacobian import calcJacobian
from IK_velocity import IK_velocity

from core.interfaces import ArmController


# arm = ArmController()


class IK:
    # JOINT LIMITS
    lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    center = lower + (upper - lower) / 2  # compute middle of range of motion of each joint

    fk = FK()

    # arm = ArmController()

    def __init__(self, linear_tol=1e-2, angular_tol=1e-3, max_steps=500, min_step_size=1e-5):
        """
        Constructs an optimization-based IK solver with given solver parameters.
        Default parameters are tuned to reasonable values.

        PARAMETERS:
        linear_tol - the maximum distance in meters between the target end
        effector origin and actual end effector origin for a solution to be
        considered successful
        angular_tol - the maximum angle of rotation in radians between the target
        end effector frame and actual end effector frame for a solution to be
        considered successful
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # solver parameters
        self.linear_tol = linear_tol
        self.angular_tol = angular_tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size

    ######################
    ## Helper Functions ##
    ######################

    @staticmethod
    def displacement_and_axis(target, current):
        """
        Helper function for the End Effector Task. Computes the displacement
        vector and axis of rotation from the current frame to the target frame

        This data can also be interpreted as an end effector velocity which will
        bring the end effector closer to the target position and orientation.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        current - 4x4 numpy array representing the "current" end effector orientation

        OUTPUTS:
        displacement - a 3-element numpy array containing the displacement from
        the current frame to the target frame, expressed in the world frame
        axis - a 3-element numpy array containing the axis of the rotation from
        the current frame to the end effector frame. The magnitude of this vector
        must be sin(angle), where angle is the angle of rotation around this axis
        """

        # displacement = np.zeros(3)
        # axis = np.zeros(3)

        # STUDENT CODE STARTS HERE

        displacement = target[0:3, -1] - current[0:3, -1]

        R = np.linalg.inv(current[0:3, 0:3]) @ target[0:3, 0:3]  # post-multiplying! R_tar = R_cur @ R
        S = (R - R.T) / 2  # S is the skew symmetric part of R

        a = np.array([S[2, 1], S[0, 2], S[1, 0]])  # a = [a1, a2, a3], axis of R
        axis = current[0:3, 0:3] @ a  # pre-multiply

        # END STUDENT CODE

        return displacement, axis

    @staticmethod
    def distance_and_angle(G, H):
        """
        Helper function which computes the distance and angle between any two
        transforms.

        This data can be used to decide whether two transforms can be
        considered equal within a certain linear and angular tolerance.

        Be careful! Using the axis output of displacement_and_axis to compute
        the angle will result in incorrect results when |angle| > pi/2

        INPUTS:
        G - a 4x4 numpy array representing some homogenous transformation
        H - a 4x4 numpy array representing some homogenous transformation

        OUTPUTS:
        distance - the distance in meters between the origins of G & H
        angle - the angle in radians between the orientations of G & H

        """

        # distance = 0
        # angle = 0

        # STUDENT CODE STARTS HERE

        distance = np.linalg.norm(G[0:3, -1] - H[0:3, -1])

        R = np.linalg.inv(G[0:3, 0:3]) @ H[0:3, 0:3]  # RG * R = RH

        mag = (np.trace(R) - 1) / 2
        if mag > 1:
            mag = 1
        elif mag < -1:
            mag = -1
        angle = np.arccos(mag)

        # END STUDENT CODE

        return distance, angle

    def is_valid_solution(self, q, target):
        """
        Given a candidate solution, determine if it achieves the primary task
        and also respects the joint limits.

        INPUTS
        q - the candidate solution, namely the joint angles
        target - 4x4 numpy array representing the desired transformation from
        end effector to world

        OUTPUTS:
        success - a Boolean which is True if and only if the candidate solution
        produces an end effector pose which is within the given linear and
        angular tolerances of the target pose, and also respects the joint
        limits.
        """

        # STUDENT CODE STARTS HERE

        success = True

        if np.any(q - self.upper) is True:
            print("q exceeds upper limits! ")
            success = False

        if np.any(self.lower - q) is True:
            print("q exceeds lower limits! ")
            success = False

        T0e = self.fk.forward(q)[1]  # forward() return joint_positions & T0e
        distance, angle = self.distance_and_angle(T0e, target)

        if distance > self.linear_tol:
            print("distance exceeds linear_tol! ")
            success = False

        if angle > self.angular_tol:
            print("angle exceeds angular_tol! ")
            success = False

        # END STUDENT CODE

        return success

    ####################
    ## Task Functions ##
    ####################

    @staticmethod
    def end_effector_task(q, target):
        """
        Primary task for IK solver. Computes a joint velocity which will reduce
        the error between the target end effector pose and the current end
        effector pose (corresponding to configuration q).

        INPUTS:
        q - the current joint configuration, a "best guess" so far for the final answer
        target - a 4x4 numpy array containing the desired end effector pose

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """

        # dq = np.zeros(7)

        # STUDENT CODE STARTS HERE

        current = FK().forward(q)[1]
        v, ω = IK().displacement_and_axis(target, current)  # displacement, axis
        dq = IK_velocity(q, v, ω)  # use np.lstsq() to solve

        # END STUDENT CODE

        return dq

    @staticmethod
    def joint_centering_task(q, rate=5e-1):
        """
        Secondary task for IK solver. Computes a joint velocity which will
        reduce the offset between each joint's angle and the center of its range
        of motion. This secondary task acts as a "soft constraint" which
        encourages the solver to choose solutions within the allowed range of
        motion for the joints.

        INPUTS:
        q - the joint angles
        rate - a tunable parameter dictating how quickly to try to center the
        joints. Turning this parameter improves convergence behavior for the
        primary task, but also requires more solver iterations.

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """

        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # normalize the offsets of all joints to range from -1 to 1 within the allowed range
        offset = 2 * (q - IK.center) / (IK.upper - IK.lower)
        dq = rate * -offset  # proportional term (implied quadratic cost)

        return dq

    ###############################
    ## Inverse Kinematics Solver ##
    ###############################

    def inverse(self, target, seed):
        """
        Uses gradient descent to solve the full inverse kinematics of the Panda robot.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        seed - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], which
        is the "initial guess" from which to proceed with optimization

        OUTPUTS:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], giving the
        solution if success is True or the closest guess if success is False.
        success - True if the IK algorithm successfully found a configuration
        which achieves the target within the given tolerance. Otherwise False
        rollout - a list containing the guess for q at each iteration of the algorithm
        """

        q = seed
        rollout = []
        cnt = 0  # counter

        # get joint limits
        lower_limit = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        upper_limit = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

        flag = False

        while True:

            rollout.append(q)

            # Primary Task - Achieve End Effector Pose
            dq_ik = self.end_effector_task(q, target)

            # Secondary Task - Center Joints
            dq_center = self.joint_centering_task(q)

            # STUDENT CODE STARTS HERE

            # 1 Task Prioritization

            J = calcJacobian(q)

            # method 1

            # # n > 6
            # J_add = J.T @ np.linalg.inv(J @ J.T)  # J+ = J^T * (J * J^T)^(-1)
            # dq = dq_ik + (np.identity(7) - J_add @ J) @ dq_center  # q̇' = J+ * ξ + (I - J+ * J) * b

            # method 2

            n = null_space(J).flatten()
            # print(n)

            project_dq_center = np.dot(dq_center, n) * (n / np.square(np.linalg.norm(n)))
            # print(np.linalg.norm(n))

            dq = 0.5 * (dq_ik + 2 * project_dq_center)

            # 2 Termination Conditions

            cnt += 1

            # cnt > max_steps or ||dq|| <= min_step_size
            if cnt > self.max_steps or np.linalg.norm(dq) <= self.min_step_size:
                break  # exit the while loop if conditions are met!

            # END STUDENT CODE

            q = q + dq

            # neutral position revision
            if flag is False:
                for i in range(7):
                    # while lower_limit[i] > q[i]:
                    #     print("q[i] is too small! - {}".format(q[i]))
                    #     q[i] = q[i] + 2 * pi
                    # while upper_limit[i] < q[i]:
                    #     print("q[i] is too large! - {}".format(q[i]))
                    #     q[i] = q[i] - 2 * pi
                    if lower_limit[i] > q[i] or upper_limit[i] < q[i]:
                        flag = True
                        q = np.array([0, 0, 0, -pi / 2, 0, pi / 2, pi / 4])
                        cnt = 0
                        print("Failed seed has changed to neutral pos seed, try it again!")
                        break

        success = self.is_valid_solution(q, target)
        return q, success, rollout


################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True, precision=5)

    ik = IK()

    # matches figure in the handout
    seed = np.array([0, 0, 0, -pi / 2, 0, pi / 2, pi / 4])

    target = np.array([
        [0, -1, 0, 0.3],
        [-1, 0, 0, 0],
        [0, 0, -1, .5],
        [0, 0, 0, 1],
    ])

    q, success, rollout = ik.inverse(target, seed)

    for i, q in enumerate(rollout):
        joints, pose = ik.fk.forward(q)  # joint_positions & T0e
        d, ang = IK.distance_and_angle(target, pose)
        print('iteration:', i, ' q =', q, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d, ang=ang))

    print("Success: ", success)
    print("Solution: ", q)
    print("Iterations:", len(rollout))