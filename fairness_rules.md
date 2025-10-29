# Fairness & Gender Inclusivity Rules

- The model must not unfairly penalize women or underrepresent them in credit scoring.
- During training, class/sample weights are adjusted to ensure gender balance.
- Fairness metrics (demographic parity, equal opportunity) are computed and reported.
- If model performance for women is significantly lower, retrain or adjust until parity is improved.
- All fairness steps and metrics are logged for transparency.
