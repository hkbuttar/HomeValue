# Lakefront and downtown price gradients

This analysis estimates continuous linear, cubic regression-spline, and GAM-style
B-spline gradients for distance to Lake Michigan and downtown. All forms use the
same sales, structural features, property type, neighborhood controls, month
effects, market trend, and future holdout.

The lake model additionally controls for CTA and downtown distance. The downtown
model controls for CTA accessibility and lake distance, directly addressing
whether downtown proximity adds information after transit and neighborhood.
Numeric nuisance controls are imputed only with training medians.

Each curve compares price with the 90th-percentile training distance rather than
an arbitrary category. The lakefront report estimates where a positive near-lake
premium falls to half its near-shore magnitude. A robust joint Wald test reports
whether the selected downtown distance basis is distinguishable from zero.

Functional form is selected using future-period MAE. These gradients remain
conditional hedonic associations rather than causal amenity effects.

