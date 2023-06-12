import math
import numpy as np

from black_scholes import BlackScholes


def call_payoff(S: float, K: float) -> float:
    """
    Compute the payoff of a call option

    S: spot price
    K: strike price
    """
    return np.maximum(S - K, 0)


def put_payoff(S: float, K: float) -> float:
    """
    Compute the payoff of a put option

    S: spot price
    K: strike price
    """
    return np.maximum(K - S, 0)


class Strategy:
    def __init__(
        self,
        Kp: float,
        Kc: float,
        mp: int,
        mc: int,
        S0: float,
        r: float,
        T: float,
        sigma: float,
    ) -> None:
        """
        Kp: strike price of put option
        Kc: strike price of call option
        mp: number of put options
        mc: number of call options
        S0: spot price at initial time
        r: risk-free rate
        T: time to maturity
        sigma: volatility of underlying asset
        """
        self.Kp = Kp
        self.Kc = Kc
        self.mp = mp
        self.mc = mc
        self.S0 = S0
        self.r = r
        self.T = T
        self.sigma = sigma
        self.put = BlackScholes(S0, Kp, r, T, sigma)
        self.call = BlackScholes(S0, Kc, r, T, sigma)

    def expected_wealth(self, t: float, a: int, iterations: int = 1000) -> float:
        """
        t: current time (should be between 0 and T)
        iterations: number of iterations for Monte Carlo simulation
        """

        if 0 < t <= self.T:
            # Monte Carlo simulation of stock prices
            W_t = np.random.normal(0, np.sqrt(t), size=iterations)  # Brownian motion
            S_t = self.S0 * np.exp(
                (a - 0.5 * self.sigma**2) * t + self.sigma * W_t
            )  # stock prices

            # Calculate the wealth at time t for each iteration
            X_t = self.mp * put_payoff(S_t, self.Kp) + self.mc * call_payoff(
                S_t, self.Kc
            )

            # Calculate and return the average wealth
            return np.mean(X_t)

        elif t == 0:
            return self.mp * self.put.put_option_price(
                t=0
            ) + self.mc * self.call.call_option_price(t=0)
        else:
            raise ValueError("t should be between 0 and T")


if __name__ == "__main__":
    pass
