# CTA accessibility premium analysis

Step 13 estimates four distance specifications on the same sales and future
holdout: linear miles, fixed distance bands, a cubic regression spline, and a
quantile-knot B-spline as a GAM-style smooth. The bands are 0–0.25, 0.25–0.50,
0.50–1.00, 1.00–2.00, and over 2 miles; the last is the regression reference.

Each model controls for the Step 8 structural features, property type, assessor
neighborhood, sale month, and market trend. OLS is fit to log price with HC3
robust uncertainty, while dollar predictions use training-residual smearing and
are evaluated strictly out of time.

Premium curves compare each distance with a three-mile reference and include
95% robust confidence intervals in the CSV. The best functional form is chosen
by future-period MAE, not in-sample fit. Descriptive flags summarize whether the
curve is consistent with a premium, immediately adjacent disamenity, or
diminishing benefit.

These are conditional hedonic associations. Transit placement, land use, and
housing prices are jointly related to neighborhood conditions, so the results
must not be described as causal effects.

