import numpy as np

if __name__ == '__main__':
    from calcJacobian import calcJacobian
else:
    from calcJacobian import calcJacobian


def IK_velocity(q_in, v_in, ω_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    NaN, then that velocity can be anything
    :param ω_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v and omega
         are infeasible, then dq should minimize the least squares error. If v
         and omega have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    # STUDENT CODE GOES HERE

    J = calcJacobian(q_in)
    ξ = np.append(v_in, ω_in)

    # Keep only the rows of J that correspond to non-NaN entries in ξ
    non_NaN = np.logical_not(np.isnan(ξ))
    J_filtered = J[non_NaN]
    ξ_filtered = ξ[non_NaN]

    q̇ = np.linalg.lstsq(J_filtered, ξ_filtered, rcond=None)[0]

    return q̇


if __name__ == "__main__":
    """
    main function
    """
    import itertools

    # Basic edge-case sanity testing

    q = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, np.pi / 4])

    # Edge case: all NaN
    # Make sure that J_filtered doesn't decay to 0-dimensional
    # Should output an array of 7 zeros
    v = np.array([np.nan, np.nan, np.nan])
    ω = np.array([np.nan, np.nan, np.nan])
    print(np.round(IK_velocity(q, v, ω), 3))

    # Edge case: only one non-NaN
    # Make sure that J_filtered doesn't decay to 1-dimensional
    v = np.array([1, np.nan, np.nan])
    ω = np.array([np.nan, np.nan, np.nan])
    print(np.round(IK_velocity(q, v, ω), 3))

    # Edge case: row/col vectors
    # Make sure this doesn't interfere with append()
    v = np.array([1, 1, 1])
    ω = np.array([1, 1, 1])
    v.shape = (1, 3)
    v.shape = (1, 3)
    print(np.round(IK_velocity(q, v, ω), 3))
    v.shape = (3, 1)
    v.shape = (3, 1)
    print(np.round(IK_velocity(q, v, ω), 3))
    v.shape = (3, 1)
    v.shape = (3, 1)
    print(np.round(IK_velocity(q, v, ω), 3))
    v.shape = (3,)
    v.shape = (3, 1)
    print(np.round(IK_velocity(q, v, ω), 3))
    v.shape = (1, 3)
    v.shape = (3,)
    print(np.round(IK_velocity(q, v, ω), 3))

    # Round-trip testing

    print("Round trip testing:")

    q1 = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, np.pi / 4])
    q2 = np.array([0, 0, 0, 0, 0, 0, 0])
    q3 = np.array([np.pi / 2, np.pi / 4, np.pi / 2, -np.pi / 4, np.pi / 2, np.pi / 2, np.pi / 4])
    q_s = (q1, q2, q3)

    v1 = np.array([0, 0, 0])
    v2 = np.array([1, np.nan, -1])
    v3 = np.array([0.3, -0.7, 0.5])
    v_s = (v1, v2, v3)

    ω1 = np.array([0, 0, 0])
    ω2 = np.array([1, np.nan, -1])
    ω3 = np.array([0.3, -0.7, 0.5])
    ω_s = (ω1, ω2, ω3)

    for q, v, ω in itertools.product(q_s, v_s, ω_s):
        ξ_expected = np.append(v, ω)
        nonNaN = np.logical_not(np.isnan(ξ_expected))
        ξ_expected = ξ_expected[nonNaN]

        q̇ = IK_velocity(q, v, ω)
        J = calcJacobian(q)
        ξ_actual = J @ q̇
        ξ_actual = ξ_actual[nonNaN]

        delta = np.linalg.norm(ξ_expected - ξ_actual)

        if delta >= 1e-5:
            print("=======================")
            print("q:", np.round(q, 4))
            print("v:", np.round(v, 4))
            print("ω:", np.round(ω, 4))
            print("Expected:", np.round(ξ_expected, 4))
            print("Actual:", np.round(ξ_actual, 4))
            print("delta: ", np.round(delta, 4))
