import numpy as np
import matplotlib.pyplot as plt

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


class Option:
    CALL = "call"
    PUT = "put"


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
        self.initial_wealth = self.expected_wealth(t=0, a=None)

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

        if t > self.T or t < 0:
            raise ValueError("t should be between 0 and T")

        return self.initial_wealth * np.exp(self.r * t)

    def set_put_call_proportion(self) -> None:
        """
        Proportion compute the good proportion mc / mp
        """
        proportion = round((1 - norm.cdf(self.put.d1())) / norm.cdf(self.call.d1()), 4)
        self.mp = 1e4
        self.mc = int(proportion * self.mp)


class HedgingStrategy:
    def __init__(
        self,
        black_scholes_model: BlackScholes,
        option_type: Option,
        mu: float,
        dt: float,
    ) -> None:
        """
        black_scholes_model: instance of BlackScholes model
        mu: drift of the stock price
        dt: time step for discretization
        iterations: number of iterations for Monte Carlo simulation
        """

        # Initialize model parameters
        self.model = black_scholes_model
        self.mu = mu
        self.dt = dt
        self.num_steps = int(self.model.T / dt)
        self.rf_rate = self.model.r
        self.sigma = self.model.sigma

        # Initialize option parameters
        if option_type == Option.CALL:
            self.payoff = call_payoff
            self.option_price = self.model.call_option_price
            self.option_delta = self.model.call_option_delta

        elif option_type == Option.PUT:
            self.payoff = put_payoff
            self.option_price = self.model.put_option_price
            self.option_delta = self.model.put_option_delta

        else:
            raise ValueError("option_type should be either 'call' or 'put'")

    def _initialize_paths(self, num_paths):
        # Initialize paths
        self.stock_prices = np.zeros((self.num_steps, num_paths))
        self.risk_free_asset = np.zeros_like(self.stock_prices)
        self.portfolio_value = np.zeros_like(self.stock_prices)
        self.deltas = np.zeros_like(self.stock_prices)

        # Initialize values at t=0
        self.stock_prices[0] = self.model.S0
        self.risk_free_asset[0] = 1
        self.portfolio_value[0] = self.option_price(t=0)

    def _update_paths(self, num_paths):
        for t in range(self.num_steps - 1):
            # Generate random paths for stock prices
            dW = np.random.normal(0, np.sqrt(self.dt), size=num_paths)
            self.stock_prices[t + 1] = self.stock_prices[t] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * dW
            )
            self.risk_free_asset[t + 1] = self.risk_free_asset[t] * np.exp(
                self.rf_rate * self.dt
            )
            self.deltas[t + 1] = self.option_delta(
                (t + 1) * self.dt, self.stock_prices[t + 1]
            )
            self.portfolio_value[t + 1] = self.deltas[t + 1] * self.stock_prices[
                t + 1
            ] + (
                np.exp(-self.rf_rate * (t + 1) * self.dt)
                * (
                    self.option_price((t + 1) * self.dt, self.stock_prices[t + 1])
                    - self.stock_prices[t + 1]
                )
                * self.risk_free_asset[t + 1]
            )

    def simulate(self, num_paths):
        """
        Simulate by Monte Carlo the hedging strategy
        num_paths: number of paths to simulate for Monte Carlo
        """
        self._initialize_paths(num_paths)
        self._update_paths(num_paths)

        payoff = np.maximum(self.stock_prices[-1] - self.model.K, 0)
        return np.mean((self.portfolio_value[-1] - payoff) ** 2)

    def plot_spot_price(self, num_paths):
        """
        Plot the evolution of the spot price and the option price
        num_paths: number of paths to simulate for Monte Carlo
        """
        self._initialize_paths(num_paths)
        self._update_paths(num_paths)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(self.stock_prices)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Spot Price")
        ax.legend()
        plt.show()

    def plot_option_price(self, num_paths):
        """
        Plot the evolution of the spot price and the option price
        num_paths: number of paths to simulate for Monte Carlo
        """
        self._initialize_paths(num_paths)
        self._update_paths(num_paths)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(self.portfolio_value)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Portfolio value")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    ####### Parameters #######
    KP = KC = 25  # strike price in dollars
    S0 = 30  # spot price in dollars
    T = 1  # time to maturity in years
    R = 0.05  # risk-free rate
    SIGMA = 0.45  # volatility of underlying asset

    ####### Edging for a call #######
    model = BlackScholes(S0, KC, R, T, SIGMA)
    strategy = HedgingStrategy(model, Option.CALL, 0.2, 1e-3)
    # mean_squared_error = strategy.simulate(int(1e5))
    # print("[*] Mean squared error: ", mean_squared_error)
    strategy.plot_option_price(20)
