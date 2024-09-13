 Cryptocurrency Price Prediction using LSTM & Multi-Head Attention Networks:

 • Combined LSTM and Transformer Networks to predict crypto prices using a 5-year dataset, utilizing LSTM for
 temporal dependencies and Transformer for attention mechanisms
 • Used historical prices to forecast next-day prices, enhancing accuracy with the 8-Head Multi-Head Attention
 feature
 • Achieved a 0.001747 MSE on validation, showcasing the effectiveness of the integrated model

FOREX-TradingStrategy.R : GARCH Forex Forecasting Strategy

TCA.ipynb : Transaction Cost Analysis Project
Optimal execution with nonlinear transient market impact: The Research Paper that inspired the Project (TCA)
xnas-itch-20230703.tbbo.cv : The dataset used for developing the model and extracting the new features

Stochastic (GBM based Forecasting) Strategy:

StochasticDayTrading.py is the strategy that I implemented which centered around prdicting the next days AUD/USD Exchange rate by constrcuting a GBM process using previous 10 days of data.
Report by Ritwij : is the Report which dicusses in depth about the startegies performance.
dataset was scrapped from yahoo finance.

 Slippage Calculation & Transaction Cost Analysis (TCA)
 • Developed a multivariate regression model based on insights from the research paper ”Optimal Execution with
 Nonlinear Transient Market Impact 2014” .
 • Applied a log-linear decay model to analyze and predict transient market impacts, refining the precision of
 Slippage calculations .
 • Constructed and incorporated features like trade intensity, cumulative volume, and market resilience as
 predictive factors for Slippage Analysis
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
