import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from strategy import Strategy


def monte_carlo_convergence(
    iterations_numbers: list[int], plot_path: str = None, step_number: int = 100
) -> None:
    """
    Run the Monte Carlo simulation for different number of iterations
    to evaluate the convergence of the method.

    Plot a graph with the number of iterations on the x-axis and the
    average reward on the y-axis with error bars.

    :param iterations_numbers: list of number of iterations to run
    :param plot_path: path to save the plot
    """

    ####### Parameters #######
    KP = KC = 25  # strike price in dollars
    S0 = 30  # spot price in dollars
    T = 0.25  # time to maturity in years
    R = 0.05  # risk-free rate
    SIGMA = 0.45  # volatility of underlying asset

    A = R  # drift of the stock price

    ####### Strategy #######
    strategy = Strategy(KP, KC, S0, R, T, SIGMA)
    expected_value = strategy.expected_wealth_riskfree(t=T)  # because A = R

    ####### Monte Carlo simulation #######
    mean_rewards = []
    std_rewards = []
    print("Expected value:", expected_value)
    print(" ~ ~ Experiment to evaluate the convergence of the Monte Carlo method ~ ~")
    print("Number of iterations, Average reward, Standard deviation")
    for iterations_number in iterations_numbers:
        rewards = []
        for _ in tqdm(range(step_number)):
            rewards.append(
                strategy.expected_wealth(t=T, a=A, iterations=iterations_number)
            )
        print(
            f"{iterations_number}, {np.mean(rewards)}, {np.std(rewards)}"
        )  # log the results

        # Save the results
        mean_rewards.append(np.mean(rewards))
        std_rewards.append(np.std(rewards))

    ####### Plot #######
    if plot_path is not None:
        plt.xscale("log")
        plt.errorbar(
            iterations_numbers, mean_rewards, yerr=std_rewards, fmt="o", capsize=5
        )
        plt.axhline(y=expected_value, color="r", linestyle="--")
        plt.xlabel("Number of iterations")
        plt.ylabel("Average reward")
        plt.savefig(plot_path)
        plt.clf()


def results(
    a_values: list[float], plot_path: str = None, iteration_number: int = 100
) -> None:
    """
    Run the Monte Carlo simulation for different values of a
    to validate numerically the results of the paper.

    Plot a graph with the a value x-axis and the
    reward of risky method on the y-axis.

    :param a_values: list of a values to run
    """

    ####### Parameters #######
    KP = KC = 25  # strike price in dollars
    S0 = 30  # spot price in dollars
    T = 0.25  # time to maturity in years
    R = 0.05  # risk-free rate
    SIGMA = 0.45  # volatility of underlying asset

    ####### Strategy #######
    strategy = Strategy(KP, KC, S0, R, T, SIGMA)
    expected_value_risk_free = strategy.expected_wealth_riskfree(t=T)  # because A = R
    expected_value_riskies = []

    ####### Monte Carlo simulation #######
    print(" ~ ~ Experiment to evaluate the impact of a ~ ~")
    print("a,initial wealth,riskfree reward,resky reward")
    for a in a_values:
        expected_value_risky = strategy.expected_wealth(
            t=T, a=a, iterations=iteration_number
        )
        expected_value_riskies.append(expected_value_risky)

        # log the results
        print(
            f"{a}, {strategy.initial_wealth}, {expected_value_risk_free}, {expected_value_risky}"
        )

    ####### Plot #######
    if plot_path is not None:
        plt.plot(a_values, expected_value_riskies, "x", label="risky reward")
        plt.axhline(y=expected_value_risk_free, color="r", linestyle="--", label="risk-free reward")
        plt.xlabel("a")
        plt.ylabel("Strategy reward")
        plt.legend()
        plt.savefig(plot_path)
        plt.clf()


if __name__ == "__main__":
    # monte_carlo_convergence(
    #     iterations_numbers=[10**k for k in range(1, 8)],
    #     plot_path="results/convergence.pdf",
    #     step_number=20,
    # )
    results(
        a_values=np.linspace(-0.05, 0.2, 20),
        plot_path="results/a.pdf",
        iteration_number=int(1e8),
    )
