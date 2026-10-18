"""Project-wide configuration constants."""

RANDOM_STATE = 42

# Columns to drop immediately (IDs, constants, leakage)
DROP_COLS = [
    'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
    'Lat Long', 'Latitude', 'Longitude',
    'Churn Label', 'Churn Value', 'Churn Score', 'Churn Reason', 'CLTV',
]

# Features to drop for the final "slim" model
# (proven via permutation importance to have no signal)
SLIM_DROP = ['Gender', 'Streaming Movies', 'Multiple Lines']