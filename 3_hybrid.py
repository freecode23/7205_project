import pandas as pd
import numpy as np
import os
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error
from helper.evaluate import ndcg_at_k  

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
merged = cbf_df.merge(svd_df, on=['user_idx', 'movie_idx', 'rating'])
merged = merged.merge(nn_df, on=['user_idx', 'movie_idx', 'rating'], how='inner')
print(len(cbf_df), len(svd_df), len(nn_df), len(merged))


 # Hybrid grid search
best_rmse = float('inf')
best_mae = float('inf')
best_rmse_weights = None
best_mae_weights = None

# set Top-K value
K = 10

# initialize the best matric 
best_hr = -np.inf
best_ndcg = -np.inf
best_hr_weights = None
best_ndcg_weights = None


results = []  # (optional) to store all for plotting

for w1 in np.arange(0, 1.1, 0.1):
    for w2 in np.arange(0, 1.1 - w1, 0.1):
         w3 = 1 - w1 - w2
         merged['hybrid_pred'] = (
             w1 * merged['predicted_cbf'] +
             w2 * merged['predicted_svd'] +
             w3 * merged['predicted_nn']
         )

         rmse = root_mean_squared_error(merged['rating'], merged['hybrid_pred'])
         mae = mean_absolute_error(merged['rating'], merged['hybrid_pred'])

         results.append({'w_cbf': w1, 'w_svd': w2, 'w_nn': w3, 'rmse': rmse, 'mae': mae})

         if rmse < best_rmse:
             best_rmse = rmse
             best_rmse_weights = (w1, w2, w3)

         if mae < best_mae:
             best_mae = mae
             best_mae_weights = (w1, w2, w3)
             
             

 # Convert to DataFrame if you want to plot later
results_df = pd.DataFrame(results)

 # Print best results
print("\n🔍 Best Hybrid Results:")
print(f"✅ Best RMSE: {best_rmse:.4f} at weights (CBF, SVD, NN): {best_rmse_weights}")
print(f"✅ Best MAE:  {best_mae:.4f} at weights (CBF, SVD, NN): {best_mae_weights}")


# 遍历权重组合（w1, w2, w3 分别对应 Content-Based Filtering、SVD、NN的预测结果）
for w1 in np.arange(0, 1.1, 0.1):
    for w2 in np.arange(0, 1.1 - w1, 0.1):
        w3 = 1 - w1 - w2
        
        # 计算混合预测：对每条记录，混合预测 = w1 * predicted_cbf + w2 * predicted_svd + w3 * predicted_nn
        merged['hybrid_pred'] = (
            w1 * merged['predicted_cbf'] +
            w2 * merged['predicted_svd'] +
            w3 * merged['predicted_nn']
        )
        
        # 下面以每个用户为单位评估 HR@K 和 NDCG@K
        user_ids = merged['user_idx'].unique()
        hr_list = []
        ndcg_list = []
        
        for user in user_ids:
            # 获取该用户的所有记录
            user_data = merged[merged['user_idx'] == user]
            if user_data.empty:
                continue
            # 定义 ground truth：通常我们选择真实评分 >= 4.0 的电影为用户真正喜欢的电影
            ground_truth = user_data[user_data['rating'] >= 4.0]['movie_idx'].values
            if len(ground_truth) == 0:
                continue
            
            # 根据混合预测值降序排序，取前 K 个作为推荐列表
            top_k_recs = user_data.sort_values(by='hybrid_pred', ascending=False).head(K)
            recommended = top_k_recs['movie_idx'].values
            
            # 计算 HR@K，如果推荐列表中至少包含一个 ground truth 中的电影，则计为 1，否则为 0
            hr = 1 if np.intersect1d(recommended, ground_truth).size > 0 else 0
            hr_list.append(hr)
            
            # 使用 ndcg_at_k 函数计算 NDCG@K
            ndcg = ndcg_at_k(recommended, ground_truth, K)
            ndcg_list.append(ndcg)
        
        avg_hr = np.mean(hr_list) if hr_list else 0
        avg_ndcg = np.mean(ndcg_list) if ndcg_list else 0
        
        results.append({
            'w_cbf': w1,
            'w_svd': w2,
            'w_nn': w3,
            'avg_hr': avg_hr,
            'avg_ndcg': avg_ndcg
        })
        
        # 更新最佳权重
        if avg_hr > best_hr:
            best_hr = avg_hr
            best_hr_weights = (w1, w2, w3)
        if avg_ndcg > best_ndcg:
            best_ndcg = avg_ndcg
            best_ndcg_weights = (w1, w2, w3)

print("\n🔍 Best Hybrid Results for HR@K:")
print(f"Best HR@{K}: {best_hr:.4f} at weights (CBF, SVD, NN): {best_hr_weights}")

print("\n🔍 Best Hybrid Results for NDCG@K:")
print(f"Best NDCG@{K}: {best_ndcg:.4f} at weights (CBF, SVD, NN): {best_ndcg_weights}")