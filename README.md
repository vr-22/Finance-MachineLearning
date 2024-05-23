JP Morgan Chase & Co. Quantitative Research Virtual Experience Program on Forage- January 2024 :

FRG1.py- Project for the PS in Task 1 :

Utilize monthly natural gas price data from October 31, 2020, to September 30, 2024, sourced from a market data provider.
Develop a tool to estimate and extrapolate natural gas purchase prices for any date within this timeframe, projecting one year into the future.
Conduct thorough analysis to identify price patterns and factors influencing variations, such as seasonal trends.
Exclude considerations for market holidays, weekends, or bank holidays in the analysis.

FRG2.py : Project For Tasks 2:

Develop a prototype pricing model for validation and testing, potentially for future automated quoting.
Create a function utilizing existing data to price gas storage contracts, considering client's flexibility in injection and withdrawal dates.
Account for all cash flows associated with the product.
Ensure the model accommodates manual oversight for exploring options with the client.

FRG3n.py : Project For Tasks 3:

Utilize tabular loan borrower data to develop a predictive model for default probability (PD) estimation.
Construct a function capable of estimating the expected loss on a loan based on borrower details, considering a recovery rate of 10%.
Explore various techniques, including regression, decision trees, and advanced methods, to train the model and conduct a comparative analysis.
Ensure the function takes loan properties as input and outputs the expected loss, facilitating risk management decisions.

FRG4.py : Project For Tasks 4:

Develop a rating map for FICO scores to credit ratings using quantization, aiming to optimize properties like mean squared error or log-likelihood.
Consider various approaches, including approximating all entries in a bucket to minimize squared error or maximizing log-likelihood considering bucket boundaries and default probabilities.
Utilize dynamic programming (although I found Unsupervised Learning oriented clustering techniques, especially Gaussian Mixture ,to be more useful)to address the quantization problem incrementally, possibly splitting it into subproblems based on FICO score ranges.
Explore resources for background on likelihood functions and dynamic programming to inform the quantization process.

FOREX-TradingStrategy.R : GARCH Forex Forecasting Strategy
