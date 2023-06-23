# Option Pricing and Hedging Strategies

This repository provides Python implementations for pricing European options and developing dynamic hedging strategies. These implementations are based on the [Black-Scholes model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model) and [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) simulations. 

## Overview

The `BlackScholes` class allows users to price European call and put options using the Black-Scholes-Merton formula. This includes calculating the option's `d1` and `d2` terms, which are integral to the pricing formula.

The `Strategy` class uses the Black-Scholes model and Monte Carlo simulations to evaluate different trading strategies involving European call and put options. The `HedgingStrategy` class, on the other hand, focuses on dynamic hedging strategies involving self-financing portfolios.

## Mathematical Background

The Black-Scholes model is a mathematical model for pricing options. It assumes that the market is efficient, there are no transaction costs or taxes, the risk-free interest rate is constant and known, the returns on the underlying asset are normally distributed and independent over time, and the volatility of the underlying asset is constant and known.

The option pricing formula used in the `BlackScholes` class relies on the concepts of `d1` and `d2`, which are derived from the Black-Scholes-Merton differential equation. These terms account for factors such as the underlying asset's price, the option's strike price, the risk-free interest rate, the time to expiry, and the asset's volatility.

The `Strategy` class simulates trading strategies involving options using Monte Carlo simulations. These simulations involve generating random price paths for the underlying asset based on its expected return and volatility. By averaging the payoffs of the options across these simulations, we can estimate the expected wealth from a given strategy.

The `HedgingStrategy` class uses the concept of 'Delta', the rate of change of the option price with respect to changes in the underlying asset's price, to dynamically adjust the portfolio's holdings of the risky asset to remain self-financing.

## Getting Started

Clone the repository and navigate to the downloaded folder:

```bash
git https://github.com/sachabinder/ProjCalculStoch.git 
cd ProjCalculStoch
```

This project uses Python 3.8.5 and the requirements can be installed by running:

```bash
pip install -r requirements.txt
```

## Contact

Please open an issue if you encounter any problems while using this repository.