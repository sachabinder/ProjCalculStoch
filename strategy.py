import numpy as np

from scipy.stats import norm
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
        self.S0 = S0
        self.r = r
        self.T = T
        self.sigma = sigma
        self.put = BlackScholes(S0, Kp, r, T, sigma)
        self.call = BlackScholes(S0, Kc, r, T, sigma)
        self.mp = None
        self.mc = None
        self.set_put_call_proportion()

    def expected_wealth(self, t: float, a: int, iterations: int = 1000) -> float:
        """
        Compute the expected wealth at time t using Monte Carlo simulation
        by the strategy of buying mp put options and mc call options

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

    def expected_wealth_riskfree(self, t: float) -> float:
        """
        Compute the expected wealth at time t using the riskless way
        by the strategy of only the risk-free asset

        t: current time (should be between 0 and T)
        """
        initial_wealth = self.expected_wealth(t=0, a=None)

        if t > self.T or t < 0:
            raise ValueError("t should be between 0 and T")

        return initial_wealth * np.exp(self.r * t)

    def set_put_call_proportion(self) -> None:
        """
        Proportion compute the good proportion mc / mp
        """
        proportion = round((1 - norm.cdf(self.put.d1())) / norm.cdf(self.call.d1()), 4)
        self.mp = 1e4
        self.mc = int(proportion * self.mp)


if __name__ == "__main__":
    ####### Parameters #######
    KP = KC = 25  # strike price in dollars
    S0 = 30  # spot price in dollars
    T = 0.25  # time to maturity in years
    R = 0.05  # risk-free rate
    SIGMA = 0.45  # volatility of underlying asset

    A = 0.09  # drift of the stock price

    ####### Strategy #######
    strategy = Strategy(KP, KC, S0, R, T, SIGMA)

    ####### Results #######
    print("[*] Initial wealth: ", strategy.expected_wealth(t=0, a=None))
    print("[*] Safe strategy: ", strategy.expected_wealth_riskfree(t=T))
    print(
        "[*] Risk strategy: ", strategy.expected_wealth(t=T, a=A, iterations=int(1e8))
    )
