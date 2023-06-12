import math
from scipy.stats import norm


class BlackScholes:
    """
    Pricer for European call and put options using Black-Scholes model
    """

    def __init__(self, S0: float, K: float, r: float, T: float, sigma: float) -> None:
        """
        S0: spot price
        K: strike price
        r: risk-free rate
        T: time to maturity
        sigma: volatility of underlying asset
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.d1 = self.d1()
        self.d2 = self.d2()

    def d1(self, t: float = 0, St: float = None) -> float:
        """
        Compute the d1 term in Black-Scholes formula

        t: actual time
        St: spot price at time t
        """
        # Spot price at time t
        if t == 0:
            St = self.S0
        elif St is None:
            raise ValueError("St cannot be None if t is not 0")

        theta = self.T - t
        return (math.log(St / self.K) + (self.r + 0.5 * self.sigma**2) * theta) / (
            self.sigma * math.sqrt(theta)
        )

    def d2(self, t: float = 0, St: float = None) -> float:
        """
        Compute the d2 term in Black-Scholes formula

        t: actual time
        St: spot price at time t
        """
        # Spot price at time t
        if t == 0:
            St = self.S0
        elif St is None:
            raise ValueError("St cannot be None if t is not 0")

        theta = self.T - t
        return self.d1(t, St) - self.sigma * math.sqrt(theta)

    def call_option_price(self, t: float = 0, St: float = None) -> float:
        """
        Compute the price of a European call option at time t
        using Black-Scholes formula

        t: actual time
        St: spot price at time t
        """
        # Spot price at time t
        if t == 0:
            St = self.S0
        elif St is None:
            raise ValueError("St cannot be None if t is not 0")

        theta = self.T - t

        return St * norm.cdf(self.d1(t=t, St=St)) - self.K * math.exp(
            -self.r * theta
        ) * norm.cdf(self.d2(t=t, St=St))

    def put_option_price(self, t: float = 0, St: float = None) -> float:
        """
        Compute the price of a European put option at time t
        using Black-Scholes formula

        t: actual time
        St: spot price at time t
        """
        # Spot price at time t
        if t == 0:
            St = self.S0
        elif St is None:
            raise ValueError("St cannot be None if t is not 0")

        theta = self.T - t

        return self.K * math.exp(-self.r * theta) * norm.cdf(
            -self.d2(t=t, St=St)
        ) - St * norm.cdf(-self.d1(t=t, St=St))
