import pandas as pd
import numpy as np
import os
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Paths
RESULTS_PATH = "./results"
os.makedirs(RESULTS_PATH, exist_ok=True)

CB_PATH = os.path.join(RESULTS_PATH, "content_pred_df.pkl")
SVD_PATH = os.path.join(RESULTS_PATH, "svd_pred_df.pkl")
NN_PATH = os.path.join(RESULTS_PATH, "nn_pred_df.pkl")

# Load prediction data
cbf_df = pd.read_pickle(CB_PATH).rename(columns={'predicted': 'predicted_cbf'})
svd_df = pd.read_pickle(SVD_PATH).rename(columns={'predicted': 'predicted_svd'})
nn_df = pd.read_pickle(NN_PATH).rename(columns={
    'predicted_rating': 'predicted_nn',
    'actual_rating': 'rating'
    })

# Merge all on userId, movieId, and rating
merged = cbf_df.merge(svd_df, on=['userId', 'movieId', 'rating'])
merged = merged.merge(nn_df, on=['userId', 'movieId', 'rating'])
print(len(cbf_df), len(svd_df), len(nn_df), len(merged))


# # Hybrid grid search
# best_rmse = float('inf')
# best_mae = float('inf')
# best_rmse_weights = None
# best_mae_weights = None

# results = []  # (optional) to store all for plotting

# for w1 in np.arange(0, 1.1, 0.1):
#     for w2 in np.arange(0, 1.1 - w1, 0.1):
#         w3 = 1 - w1 - w2
#         merged['hybrid_pred'] = (
#             w1 * merged['predicted_cbf'] +
#             w2 * merged['predicted_svd'] +
#             w3 * merged['predicted_nn']
#         )

#         rmse = root_mean_squared_error(merged['rating'], merged['hybrid_pred'])
#         mae = mean_absolute_error(merged['rating'], merged['hybrid_pred'])

#         results.append({'w_cbf': w1, 'w_svd': w2, 'w_nn': w3, 'rmse': rmse, 'mae': mae})

#         if rmse < best_rmse:
#             best_rmse = rmse
#             best_rmse_weights = (w1, w2, w3)

#         if mae < best_mae:
#             best_mae = mae
#             best_mae_weights = (w1, w2, w3)

# # Convert to DataFrame if you want to plot later
# results_df = pd.DataFrame(results)

# # Print best results
# print("\n🔍 Best Hybrid Results:")
# print(f"✅ Best RMSE: {best_rmse:.4f} at weights (CBF, SVD, NN): {best_rmse_weights}")
# print(f"✅ Best MAE:  {best_mae:.4f} at weights (CBF, SVD, NN): {best_mae_weights}")