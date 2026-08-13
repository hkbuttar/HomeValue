# Hedonic price model

The Step 8 model estimates OLS on log recorded sale price. Candidate structural
features are log building area, log land area, bedrooms, bathrooms, building
age, stories, garage spaces, and basement presence. Features without sufficient
training observations are omitted; retained numeric missing values use training
medians.

Controls include broad property type, assessor neighborhood (or census tract),
calendar-month effects, and a linear year trend. The trend can extrapolate into
the out-of-time test period, unlike an unseen future-year dummy. Rare and unseen
categories map to an explicit `__OTHER__` level learned from training data.

Inference uses HC3 heteroskedasticity-robust standard errors and confidence
intervals. Dollar predictions apply Duan's training-residual smearing factor
when converting predicted log prices back to levels. Results include both level
and log-scale metrics, coefficients, uncertainty, fitted schema, and plain-
language coefficient interpretations.

Coefficients describe conditional hedonic associations. They are not causal
effects without additional identification assumptions.

