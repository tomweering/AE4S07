import math


class _Appendix13_8_eCalcs:
    def __init__(self, params):
        self.P = params.P
        self.H = params.H
        self.h = params.h
        self.t_1 = params.t_1
        self.t_2 = params.t_2
        self.ts_1 = params.ts_1
        self.ts_2 = params.ts_2
        self.A_1 = params.A_1
        self.A_2 = params.A_2
        self.H_1 = params.H_1
        self.h_1 = params.h_1
        self.p = params.p
        self.S = params.S
        self.S_y = params.S_y
        self.E_2 = params.E_2
        self.E_3 = params.E_3


    def beta_h(self):
        """

        :return: Beta for height h
        """
        return self.h / self.p

    def beta_H(self):
        """

        :return: Beta for height H
        """
        return self.H / self.p

    def table13_8_beta_h(self):
        """

        :return: beta_h or 1/beta_h, whichever is greater
        """
        return max(self.beta_h(), (1 / self.beta_h()))

    def table13_8_beta_H(self):
        """

        :return: beta_H or 1/beta_H, whichever is greater
        """
        return max(self.beta_H(), (1 / self.beta_H()))

    def __J(self, input_beta):
        """

        :param input_beta: the applicable beta parameter for the specified J factor
        :return: parameter J corresponding to the beta value
        """
        # beta or 1/beta, parameter J
        values = [
            (1.0, 4.9),
            (1.1, 4.3),
            (1.2, 3.9),
            (1.3, 3.6),
            (1.4, 3.3),
            (1.5, 3.1),
            (1.6, 2.9),
            (1.7, 2.8),
            (1.8, 2.6),
            (1.9, 2.5),
            (2.0, 2.4),
            (3.0, 2.1),
            (4.0, 2.0)
        ]
        higherInterpolIndex = 0
        for i in range(len(values)):
            if values[i][0] == input_beta:
                # We have what we want
                return values[i][1]

            elif input_beta > 4:
                # Limit of the table
                return 2

            elif values[i][0] < input_beta:
                # Keep looking...
                continue

            elif values[i][0] > input_beta:
                # we got it
                higherInterpolIndex = i

            else:
                raise ValueError("input <<%r>> to J table lookup is not in range" % input_beta)

        x0 = values[higherInterpolIndex - 1][0]
        y0 = values[higherInterpolIndex - 1][1]
        x1 = values[higherInterpolIndex][0]
        y1 = values[higherInterpolIndex][0]

        return y0 + (input_beta - x0) * ((y1 - y0) / (x1 - x0))

    def J_h(self):
        """

        :return: Plate parameter from Table 13-8(d) for height h
        """
        return self.__J(self.table13_8_beta_h())

    def J_H(self):
        """

        :return: Plate parameter from Table 13-8(d) for height H
        """
        return self.__J(self.table13_8_beta_H())

    def p_1(self):
        """

        :return: p_1 for side of length H; Appendix 13-8, equation 1a or 1b
        """
        factor = 1
        if self.H < self.p:
            factor = (1 / self.beta_H())

        return (self.t_1 * math.sqrt(self.S * self.J_H() / self.P)) * factor

    def p_2(self):
        """

        :return: p_2 for side of length H; Appendix 13-8, equation 1c or 1d
        """
        factor = 1
        if self.h < self.p:
            factor = (1 / self.beta_h())

        return (self.t_1 * math.sqrt(self.S * self.J_h() / self.P)) * factor

    def delta(self):
        """

        :return: Effective width coefficient per table 13-8(e). TODO: include tabular values
        """
        return math.sqrt(self.E_2 / self.E_3)

    def w_1(self):
        """

        :return: Stiffener effective width; Appendix 13-8, equation 2
        """
        return (self.t_1 * self.delta()) / math.sqrt(self.S_y)

    def w_2(self):
        """

        :return: Stiffener effective width; Appendix 13-8, equation 2
        """
        return (self.t_2 * self.delta()) / math.sqrt(self.S_y)

    def I_21(self):
        """
        Moment of inertia of combined reinforcing member and effective width of plate w and thickness t_2
        Includes stiffener contribution using the parallel axis theorem
        """
        # Moment of inertia of the plate's effective width
        I_plate = (1 / 12.0) * self.w_2() * self.t_2 ** 3

        # Stiffener moment of inertia + parallel axis term
        I_stiffener = (1 / 12.0) * self.ts_2 * self.H_1 ** 3
        A_stiffener = self.A_2
        #d_stiffener = 0.5 * (self.H_1 + self.t_2)
        d_stiffener = 0.5 * (self.t_2)

        return I_plate + I_stiffener + A_stiffener * d_stiffener ** 2

    def I_11(self):
        """
        Moment of inertia of combined reinforcing member and effective width of plate w and thickness t_1
        Includes stiffener contribution using the parallel axis theorem
        """
        # Moment of inertia of the plate's effective width
        I_plate = (1 / 12.0) * self.w_1() * self.t_1 ** 3

        # Stiffener moment of inertia + parallel axis term
        I_stiffener = (1 / 12.0) * self.ts_1 * self.h_1 ** 3
        A_stiffener = self.A_1
        #d_stiffener = 0.5 * (self.h_1 + self.t_1)
        d_stiffener = 0.5 * (self.t_1)

        return I_plate + I_stiffener + A_stiffener * d_stiffener ** 2

    def alpha1(self):
        """

        :return: rectangular vessel reinforcement parameter H_1 / h_1
        """
        return self.H_1 / self.h_1

    def k(self):
        """

        :return: reinforcement parameter (I_21 / I_11) * alpha1
        """
        return (self.I_21() / self.I_11()) * self.alpha1()

    def c_1(self):
        #  TODO: Generalize into general shapes rather than rectangles
        sign = 1
        if self.P <= 0:
            sign = -1
        return 0.5 * sign * (self.t_1 + self.ts_1)

    def c_2(self):
        #  TODO: Generalize into general shapes rather than rectangles
        sign = 1
        if self.P <= 0:
            sign = -1
        return 0.5 * sign * (self.t_2 + self.ts_2)

    def S_m_short(self):
        """

        :return: Membrane stress in short side members; Appendix 13-8, equation 3
        """
        return (self.P * self.h * self.p) / (2 * (self.A_1 + self.p * self.t_1))

    def S_m_long(self):
        """

        :return: Membrane stress in long side members; Appendix 13-8, equation 4
        """
        return (self.P * self.H * self.p) / (2 * (self.A_2 + self.p * self.t_2))

    def __commonGeomFactor(self):
        """

        :return: Factor common to equations 5, 6, 7, and 8
        """
        return (1 + self.k() * self.alpha1() ** 2) / (1 + self.k())

    def S_b_N(self):
        """

        :return: Bending stress in short side members at point N, midspan of short side; Appendix 13-8, equation 5
        """
        return (self.P * self.p * self.c_1() / (24.0 * self.I_11())) * \
               (-3.0 * (self.H ** 2) + 2.0 * (self.h ** 2) * self.__commonGeomFactor())

    def S_b_Q_short(self):
        """

        :return: Bending stress in short side members at point Q, corner; Appendix 13-8, equation 6
        """
        return ((self.P * (self.h ** 2) * self.p * self.c_1()) / (12 * self.I_11())) * self.__commonGeomFactor()

    def S_b_M(self):
        """

        :return: Bending stress in long side members at point M, midspan; Appendix 13-8, equation 7
        """
        return ((self.P * (self.h ** 2) * self.p * self.c_2()) / (24 * self.I_21())) * (-3.0 + 2.0 * self.__commonGeomFactor())

    def S_b_Q_long(self):
        """

        :return: Bending stress in long side members at point Q, corner; Appendix 13-8, equation 8
        """
        return ((self.P * (self.h ** 2) * self.p * self.c_2()) / (12.0 * self.I_21())) * self.__commonGeomFactor()

    def S_T_N(self):
        """

        :return: Total stress at point N from bending and membrane stress, midspan; Appendix 13-8, equation 9
        """
        return self.S_m_short() + self.S_b_N()

    def S_T_Q_short(self):
        """

        :return: Total stress at point Q from bending and membrane of the short side, corner; Appendix 13-8, equation 10
        """
        return self.S_m_short() + self.S_b_Q_short()

    def S_T_M(self):
        """

        :return: Total stress at point M from bending and membrane of the long side, midpsan; Appendix 13-8, equation 11
        """
        return self.S_m_long() + self.S_b_M()

    def S_T_Q_long(self):
        """

        :return: Total stress at point Q from bending and membrane of the long side, corner; Appendix 13-8, equation 12
        """
        return self.S_m_long() + self.S_b_Q_long()
    
""" if __name__ == "__main__":
    side_length = 70 / 1000
    stiffner_thickness = 10 / 1000
    stiffner_height = 10 / 1000
    total_height = 90 / 1000
    n_stiffeners = 5
    skin_thickness = 3 / 1000

    pitch = (total_height - stiffner_height) / (n_stiffeners - 1)
    print(f" Pitch = {pitch}")


    # Define input parameters
    params = {
        "P": 100000,       # Internal pressure in Pascals
        "H": side_length,          # Long side length in m
        "h": side_length,          # Short side length in m
        "t_1": skin_thickness,         # Plate thickness for long side in m
        "t_2": skin_thickness,         # Plate thickness for short side in m
        "ts_1": stiffner_thickness,         # Stiffener thickness for long side in m
        "ts_2": stiffner_thickness,         # Stiffener thickness for short side in m
        "A_1": side_length*stiffner_thickness,        # Cross-sectional area for long side stiffener in m^2
        "A_2": side_length*stiffner_thickness,        # Cross-sectional area for short side stiffener in m^2
        "H_1": stiffner_height,        # Effective height for long side stiffener in m
        "h_1": stiffner_height,        # Effective height for short side stiffener in m
        "p": pitch,          # Stiffener pitch in m
        "S": 250 * 10**6,  # Allowable stress in Pascals
        "S_y": 400 * 10**6, # Yield stress in Pascals
        "E_2": 200 * 10**9, # Young's modulus for plate in Pascals
        "E_3": 210 * 10**9  # Young's modulus for stiffener in Pascals
    }

    # Instantiate calculation object
    params_obj = type("Params", (object,), params)()
    calc = _Appendix13_8_eCalcs(params_obj)

    # Output results
    print("*** Input Parameters ***")
    for key, value in params.items():
        print(f"{key} = {value}")
    print("\n*** Stiffened Structure Calculations ***")
    print(f"Beta_h = {calc.beta_h()}")
    print(f"Beta_H = {calc.beta_H()}")
    print(f"J_h = {calc.J_h()}")
    print(f"J_H = {calc.J_H()}")
    print(f"p_1 = {calc.p_1()} mm")
    print(f"p_2 = {calc.p_2()} mm")
    print(f"Stiffener effective width w_1 = {calc.w_1()} m")
    print(f"Stiffener effective width w_2 = {calc.w_2()} m")

    print(f"Membrane stress (short side) = {calc.S_m_short()} Pa")
    print(f"Membrane stress (long side) = {calc.S_m_long()} Pa")
    print(f"Bending stress at N (short side) = {calc.S_b_N()} Pa")
    print(f"Bending stress at Q (short side) = {calc.S_b_Q_short()} Pa")
    print(f"Bending stress at M (long side) = {calc.S_b_M()} Pa")
    print(f"Bending stress at Q (long side) = {calc.S_b_Q_long()} Pa")
    print(f"Total stress at N (short side) = {calc.S_T_N()} Pa")
    print(f"Total stress at Q (short side) = {calc.S_T_Q_short()} Pa")
    print(f"Total stress at M (long side) = {calc.S_T_M()} Pa")
    print(f"Total stress at Q (long side) = {calc.S_T_Q_long()} Pa") """