import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------------------------------------------------------
# 1. CONFIGURAÇÃO E DEVICE
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executando no dispositivo: {DEVICE}")

# -----------------------------------------------------------------------------
# 2. GERAÇÃO DE DADOS COM SÉRIE TEMPORAL REAL E SAZONALIDADE
# -----------------------------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

skus = [f"SKU_{i}" for i in range(20)]
lojas = [f"LOJA_{i}" for i in range(10)]
semanas = list(range(1, 53))

rows = []
for sku in skus:
    for loja in lojas:
        base_demand = np.random.randint(60, 120)
        estoque_atual = np.random.uniform(150, 300)
        
        for sem in semanas:
            promo = 1 if np.random.rand() < 0.2 else 0
            preco = 50 + np.random.normal(0, 5)
            seasonality = 15 * np.sin(2 * np.pi * sem / 52)
            
            sell_out = max(10, base_demand + seasonality + (25 * promo) + np.random.normal(0, 5))
            sell_in = max(10, sell_out + np.random.normal(0, 6))
            
            estoque_t1 = max(0, estoque_atual + sell_in - sell_out)
            
            rows.append({
                'sku': sku,
                'loja': loja,
                'semana': sem,
                'preco': preco,
                'promo': promo,
                'estoque': estoque_atual,
                'sell_out': sell_out,
                'sell_in': sell_in,
                'estoque_t1': estoque_t1
            })
            estoque_atual = estoque_t1

df = pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 3. ENGENHARIA DE FEATURES (LAGS E MÉDIAS MÓVEIS) & ENCODING
# -----------------------------------------------------------------------------
df = df.sort_values(['sku', 'loja', 'semana']).reset_index(drop=True)

df['sell_out_lag1'] = df.groupby(['sku', 'loja'])['sell_out'].shift(1).fillna(df['sell_out'].mean())
df['sell_in_lag1'] = df.groupby(['sku', 'loja'])['sell_in'].shift(1).fillna(df['sell_in'].mean())
df['rolling_sellout'] = df.groupby(['sku', 'loja'])['sell_out'].rolling(window=3, min_periods=1).mean().reset_index(level=[0, 1], drop=True)

le_sku = LabelEncoder()
le_loja = LabelEncoder()
df['sku_id'] = le_sku.fit_transform(df['sku'])
df['loja_id'] = le_loja.fit_transform(df['loja'])

n_sku = df['sku_id'].nunique()
n_loja = df['loja_id'].nunique()

features_num = ['semana', 'preco', 'promo', 'estoque', 'sell_out_lag1', 'sell_in_lag1', 'rolling_sellout']

scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_estoque = StandardScaler()

X_num_np = scaler_X.fit_transform(df[features_num])
y_np = scaler_y.fit_transform(df[['sell_in', 'sell_out']])
scaler_estoque.fit(df[['estoque']])
est_std_val = scaler_estoque.scale_[0]

sku_ids = torch.tensor(df['sku_id'].values, dtype=torch.long)
loja_ids = torch.tensor(df['loja_id'].values, dtype=torch.long)
X_num = torch.tensor(X_num_np, dtype=torch.float32)
y_true_scaled = torch.tensor(y_np, dtype=torch.float32)

est_t = torch.tensor(df['estoque'].values, dtype=torch.float32).unsqueeze(1)
est_t1 = torch.tensor(df['estoque_t1'].values, dtype=torch.float32).unsqueeze(1)

y_mean = torch.tensor(scaler_y.mean_, dtype=torch.float32, device=DEVICE)
y_std = torch.tensor(scaler_y.scale_, dtype=torch.float32, device=DEVICE)

# -----------------------------------------------------------------------------
# 4. ARQUITETURA AVANÇADA (EMBEDDINGS 32 + MLP LARGA COM LAYERNORM)
# -----------------------------------------------------------------------------
class PINNDemandaAdvanced(nn.Module):
    def __init__(self, n_sku, n_loja, emb_dim=32, num_dim=len(features_num)):
        super().__init__()
        self.emb_sku = nn.Embedding(n_sku, emb_dim)
        self.emb_loja = nn.Embedding(n_loja, emb_dim)

        input_dim = (emb_dim * 2) + num_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, sku_id, loja_id, x_num):
        e_sku = self.emb_sku(sku_id)
        e_loja = self.emb_loja(loja_id)
        x = torch.cat([e_sku, e_loja, x_num], dim=1)
        out = self.net(x)
        return torch.nn.functional.softplus(out)

model = PINNDemandaAdvanced(n_sku, n_loja).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=35)
mse_criterion = nn.MSELoss()

# -----------------------------------------------------------------------------
# 5. LOSS FÍSICA COM ANNEALING DINÂMICO
# -----------------------------------------------------------------------------
def loss_total(y_pred_scaled, y_true_scaled, estoque_t, estoque_t1, current_epoch, max_epochs=800):
    loss_data = mse_criterion(y_pred_scaled, y_true_scaled)

    sellin_pred_scaled = y_pred_scaled[:, 0:1]
    sellout_pred_scaled = y_pred_scaled[:, 1:2]
    
    scale_ratio = y_std[0].item() / est_std_val
    
    estoque_t_scaled = (estoque_t - scaler_estoque.mean_[0]) / est_std_val
    estoque_t1_scaled = (estoque_t1 - scaler_estoque.mean_[0]) / est_std_val
    
    estoque_pred_t1_scaled = estoque_t_scaled + (sellin_pred_scaled * scale_ratio) - (sellout_pred_scaled * scale_ratio)
    loss_phy = mse_criterion(estoque_pred_t1_scaled, estoque_t1_scaled)

    sellout_pred_real = sellout_pred_scaled * y_std[1] + y_mean[1]
    loss_ruptura = mse_criterion(torch.relu(sellout_pred_real - estoque_t), torch.zeros_like(sellout_pred_real))

    # Physics Annealing: o peso da física decai suavemente da época 1 até a última
    lambda_phy = max(0.05, 0.6 * (1.0 - (current_epoch / max_epochs)))
    lambda_ruptura = 0.05

    total = loss_data + lambda_phy * loss_phy + lambda_ruptura * loss_ruptura
    return total, loss_data, loss_phy

# -----------------------------------------------------------------------------
# 6. LOOP DE TREINAMENTO OTIMIZADO
# -----------------------------------------------------------------------------
sku_ids = sku_ids.to(DEVICE)
loja_ids = loja_ids.to(DEVICE)
X_num = X_num.to(DEVICE)
y_true_scaled = y_true_scaled.to(DEVICE)
est_t = est_t.to(DEVICE)
est_t1 = est_t1.to(DEVICE)

epochs = 800
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()

    y_pred_scaled = model(sku_ids, loja_ids, X_num)
    loss, l_data, l_phy = loss_total(y_pred_scaled, y_true_scaled, est_t, est_t1, epoch, epochs)

    loss.backward()
    optimizer.step()
    scheduler.step(loss.detach())

    if epoch % 100 == 0 or epoch == 1:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | Total Loss: {loss.item():.4f} | Data Loss: {l_data.item():.4f} | Phy Loss: {l_phy.item():.4f} | LR: {current_lr:.5f}")

# -----------------------------------------------------------------------------
# 7. AVALIAÇÃO DE DESEMPENHO (RMSE EM UNIDADES REAIS)
# -----------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    y_pred_final_scaled = model(sku_ids, loja_ids, X_num)
    
    y_pred_real = y_pred_final_scaled.cpu().numpy() * scaler_y.scale_ + scaler_y.mean_
    y_true_real = y_true_scaled.cpu().numpy() * scaler_y.scale_ + scaler_y.mean_
    
    rmse_sell_in = np.sqrt(np.mean((y_pred_real[:, 0] - y_true_real[:, 0]) ** 2))
    rmse_sell_out = np.sqrt(np.mean((y_pred_real[:, 1] - y_true_real[:, 1]) ** 2))

print("\n--- Métricas Finais no Treino (Unidades Reais) ---")
print(f"RMSE Sell-In:  {rmse_sell_in:.2f} unidades")
print(f"RMSE Sell-Out: {rmse_sell_out:.2f} unidades")

# -----------------------------------------------------------------------------
# 8. INFERÊNCIA / PREVISÃO
# -----------------------------------------------------------------------------
def prever(sku: str, loja: str, features_dict: dict):
    model.eval()
    
    sku_encoded = le_sku.transform([sku])[0]
    loja_encoded = le_loja.transform([loja])[0]
    
    sku_tensor = torch.tensor([sku_encoded], dtype=torch.long, device=DEVICE)
    loja_tensor = torch.tensor([loja_encoded], dtype=torch.long, device=DEVICE)
    
    feat_df = pd.DataFrame([[features_dict[f] for f in features_num]], columns=features_num)
    x_num_scaled = scaler_X.transform(feat_df)
    x_num_tensor = torch.tensor(x_num_scaled, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        pred_scaled = model(sku_tensor, loja_tensor, x_num_tensor).cpu().numpy()
        
    pred_real = scaler_y.inverse_transform(pred_scaled)
    return {
        'sell_in': float(pred_real[0, 0]),
        'sell_out': float(pred_real[0, 1])
    }

print("\n--- Teste de Inferência ---")
resultado = prever('SKU_1', 'LOJA_3', {
    'semana': 20, 'preco': 52, 'promo': 1, 'estoque': 120,
    'sell_out_lag1': 95.0, 'sell_in_lag1': 100.0, 'rolling_sellout': 92.5
})
print(f"Predição SKU_1 na LOJA_3 -> Sell-In: {resultado['sell_in']:.2f} | Sell-Out: {resultado['sell_out']:.2f}")
