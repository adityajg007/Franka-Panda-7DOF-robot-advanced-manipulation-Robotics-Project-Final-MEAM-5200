import numpy as np
from math import pi
from math import sin
from math import cos


class FK:

    def __init__(self):
        # You may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout

        # joint position under intermediate frames
        P0j = (0, 0, 0.141, 1)
        P1j = (0, 0, 0, 1)
        P2j = (0, 0, 0.195, 1)
        P3j = (0, 0, 0, 1)
        P4j = (0, 0, 0.125, 1)
        P5j = (0, 0, -0.015, 1)
        P6j = (0, 0, 0.051, 1)
        P7j = (0, 0, 0, 1)
        self.P0j_P7j = [P0j, P1j, P2j, P3j, P4j, P5j, P6j, P7j]

        # dh parameters
        self.dh_a = (0, 0, 0.0825, -0.0825, 0, 0.088, 0)
        self.dh_alpha = (-pi / 2, pi / 2, pi / 2, -pi / 2, pi / 2, pi / 2, 0)
        self.dh_d = (0.333, 0, 0.316, 0, 0.384, 0, 0.21)
        self.dh_theta_offset = (0, 0, 0, 0, 0, 0, -pi / 4)

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here

        # D-H homogenous transformation matrices of each frame
        A1_A7 = self.compute_Ai(q)

        # T0i transformation matrices - frame i transfer to frame 0
        T01_T07 = self.compute_T0i(A1_A7)

        # joint position under 0 frame
        P0_P7 = self.compute_joint_positions(T01_T07)

        jointPositions = np.row_stack(P0_P7)
        T0e = T01_T07[-1]

        # Your code ends here
        return jointPositions, T0e

    # Helpers:

    @staticmethod
    def dh_matrix(a, alpha, d, theta):
        """
        Transformation matrix step by step
        """
        Ai = np.array([
            [cos(theta), -1 * sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
            [sin(theta), cos(theta) * cos(alpha), -1 * cos(theta) * sin(alpha), a * sin(theta)],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1]
        ])

        return Ai

    @staticmethod
    def compute_T0i(A1_A7):
        T01_T07 = [0, 0, 0, 0, 0, 0, 0]
        T01_T07[0] = A1_A7[0]
        for i in range(1, 7):
            T01_T07[i] = np.matmul(T01_T07[i - 1], A1_A7[i])

        return T01_T07

    def compute_joint_positions(self, T01_T07):
        P0_P7 = [0, 0, 0, 0, 0, 0, 0, 0]
        P0_P7[0] = (self.P0j_P7j[0])[0:3]  # calculation of P0_P7 doesn't need T matrix
        for i in range(7):
            P0_P7[i + 1] = (np.matmul(T01_T07[i], self.P0j_P7j[i + 1]))[0:3]

        return np.row_stack(P0_P7)

    # End helpers

    # This code is for Lab 2, you can ignore it for Lab 1

    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]
        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame
        """

        T01_T07 = self.compute_T0i(self.compute_Ai(q))
        z1_z6 = np.row_stack([((T01_T07[i])[0:3, 2]).flatten() for i in range(6)])
        z0_z6 = np.row_stack([np.array((0, 0, 1)), z1_z6])
        return z0_z6

    # Helper for Lab 2

    def get_origins(self, q):
        return self.compute_joint_positions(self.compute_T0i(self.compute_Ai(q)))

    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """

        # STUDENT CODE HERE: This is a function needed by lab 2

        # D-H homogenous transformation matrices of each frame
        A1_A7 = [
            self.dh_matrix(self.dh_a[i],
                           self.dh_alpha[i],
                           self.dh_d[i],
                           q[i] + self.dh_theta_offset[i])
            for i in range(7)]

        return A1_A7


if __name__ == "__main__":
    """
    main function, for testing
    """
    fk = FK()

    # matches figure in the handout or use the zero position
    # q = np.array([0, 0, 0, -pi / 2, 0, pi / 2, pi / 4])
    q = np.array([0, 0, 0, 0, 0, 0, 0])
    print(q)

    joint_positions, T0e = fk.forward(q)

    print("Joint Positions:\n", joint_positions.round(4))
    print("End Effector Pose:\n", T0e.round(4))
