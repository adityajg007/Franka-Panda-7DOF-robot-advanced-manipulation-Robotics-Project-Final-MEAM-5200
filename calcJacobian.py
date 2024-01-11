import numpy as np

if __name__ == '__main__':
    from calculateFK import FK
else:
    from calculateFK import FK

fk = FK()


def calcJacobian(q):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q: 3 x 7 configuration vector (of joint angles) [q0,q1,q2,q3,q4,q5,q6]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    # STUDENT CODE GOES HERE

    # Origins under frame 0. O0 is the first joint, O7 is the end-effector
    O0_O7 = fk.get_origins(q)
    O7_minus_O0_O6 = -O0_O7[0:-1] + O0_O7[-1]  # O7 - O0, ..., O7 - 06

    # Axes of rotation under frame 0. z0 is for the first joint, z6 is for the last joint
    z0_z6 = fk.get_axis_of_rotation(q)

    J = np.empty((6, 7))

    # linear velocity Jacobian Jv
    J[0:3] = np.cross(z0_z6, O7_minus_O0_O6, axisa=1, axisb=1, axisc=0)

    # angular velocity Jacobian Jw
    J[3:6] = z0_z6.T

    return J


if __name__ == '__main__':
    """
    main function
    """

    q = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, np.pi / 4])

    print(np.round(calcJacobian(q), 4))
