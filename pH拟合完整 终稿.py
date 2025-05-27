import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

# ==============================
# 实验数据输入
# ==============================


# # 读取 Excel 文件
# file_path = r"C:\Users\zhang\Desktop\lhz.xlsx"  # 替换为你的 Excel 文件路径
# df = pd.read_excel(file_path)
# # 确保列名匹配
# if "pH" in df.columns:
#     pHlist = df["pH"].dropna().tolist()  # 读取 pH 列并去除缺失值
# else:
#     print("未找到 'pH' 列")

# df = pd.read_excel(file_path)
# # 确保列名匹配
# if "V" in df.columns:
#     Vlist = df["V"].dropna().tolist()  # 读取 pH 列并去除缺失值
# else:
#     print("未找到 'V' 列")


#测试数据：
pHlist=[
    3.25, 3.49, 3.68, 3.84, 3.97, 4.09, 4.18, 4.26, 4.33, 4.42,
    4.49, 4.55, 4.61, 4.67, 4.75, 4.85, 4.94, 5.01, 5.08, 5.16,
    5.26, 5.37, 5.50, 5.65, 5.90, 6.37, 10.02, 10.69, 11.02, 11.21,
    11.32, 11.40
]
Vlist=[
    0.00, 0.92, 1.86, 2.96, 3.96, 4.98, 6.02, 6.94, 7.86, 8.98,
    10.06, 10.98, 11.98, 12.80, 14.06, 15.66, 16.96, 17.86, 18.76, 19.76,
    20.86, 21.86, 22.86, 23.80, 24.88, 25.96, 26.77, 27.66, 28.76, 29.88,
    30.86, 31.84
]



# ============================================
# ============================================
# ============================================
#以下为拟合计算：


# ==============================
# 数学模型
# ==============================
def model_func(pH, ka, cA, Va):
    cH = 10**(-pH)
    cOH = 1e-14 / cH
    b = cH - cOH  # 电荷平衡项
    a = ka / (cH + ka)  # 解离分数
    V = Va * (cA * a - b) / (b + cB)
    return V


# ==============================
# 第一次拟合
# ==============================

#输入参数
pH_data = np.array(pHlist)
V_data = np.array(Vlist)
# 已知参数
cB = 0.1077  # NaOH浓度（mol/L）
# 初始猜测优化    
initial_guess = [1.78e-5, 1, 60]  # [ka, cA, Va]
# 参数边界约束（与原始条件一致）
bounds = (
    [1e-7, 0.001, 10],   # 下限
    [1e-2, 1.0, 400]    # 上限
)

# 执行拟合
params_opt, params_cov = curve_fit(
    model_func,
    pH_data,
    V_data,
    p0=initial_guess,
    bounds=bounds,
    method='trf',
    max_nfev=10000
)

# 提取结果
ka_fit, cA_fit, Va_fit = params_opt


# ==============================
# 第一次结果输出
# ==============================
print("拟合结果：")
print(f"Ka = {ka_fit:.4e} (pKa = {-np.log10(ka_fit):.2f})")
print(f"初始浓度 cA = {cA_fit:.4f} mol/L")
print(f"初始体积 Va = {Va_fit:.4f} mL")
print(f"乙酸原液浓度 = {cA_fit*Va_fit/25:.4f} mol/L")

# 计算R²
V_pred = model_func(pH_data, ka_fit, cA_fit, Va_fit)
ss_res = np.sum((V_data - V_pred)**2)
ss_tot = np.sum((V_data - np.mean(V_data))**2)
r_squared = 1 - (ss_res/ss_tot)
print(f"R² = {r_squared:.4f}")

# 生成高密度拟合曲线
V_fine = np.linspace(0, 35, 300)
pH_fine = np.linspace(pH_data.min(), pH_data.max(), 300)
V_fit_curve = model_func(pH_fine, ka_fit, cA_fit, Va_fit)


# =================================================================
# 第一次拟合后剔除残差最大的四个点，进行第二次拟合
# =================================================================

# 计算第一次拟合的残差
residuals = V_data - V_pred
abs_residuals = np.abs(residuals)
max_indices = np.argsort(abs_residuals)[-4:]  # 获取最大残差的四个索引

# 剔除异常点
pH_data_clean = np.delete(pH_data, max_indices)
V_data_clean = np.delete(V_data, max_indices)


# ==============================
# 第二次拟合
# ==============================
params_opt_clean, params_cov_clean = curve_fit(
    model_func,
    pH_data_clean,
    V_data_clean,
    p0=initial_guess,
    bounds=bounds,
    method='trf',
    max_nfev=10000
)

# 提取结果
ka_fit_clean, cA_fit_clean, Va_fit_clean = params_opt_clean

# 计算新R²
V_pred_clean = model_func(pH_data_clean, ka_fit_clean, cA_fit_clean, Va_fit_clean)
ss_res_clean = np.sum((V_data_clean - V_pred_clean)**2)
ss_tot_clean = np.sum((V_data_clean - np.mean(V_data_clean))**2)
r_squared_clean = 1 - (ss_res_clean / ss_tot_clean)


# =================================================================
# 第二次结果输出
# =================================================================
print("\n剔除异常点后拟合结果：")
print(f"Ka = {ka_fit_clean:.3e} (pKa = {-np.log10(ka_fit_clean):.2f})")
print(f"初始浓度 cA = {cA_fit_clean:.4f} mol/L")
print(f"初始体积 Va = {Va_fit_clean:.4f} mL")
print(f"乙酸原液浓度 = {cA_fit_clean*Va_fit_clean/25:.4f} mol/L")
print(f"R² = {r_squared_clean:.4f}")

# 生成新拟合曲线 (pH作为自变量更准确)
pH_fine_clean = np.linspace(pH_data_clean.min(), pH_data_clean.max(), 300)
V_fit_curve_clean = model_func(pH_fine_clean, ka_fit_clean, 
                              cA_fit_clean, Va_fit_clean)



# ============================================
# ============================================
# ============================================
#以下为可视化：


# ======================
# 设置中文字体（新增部分）
# ======================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# [...] 保留原有模型定义、数据输入和第一次拟合代码，直至第一次结果输出 [...]

# ==============================
# 第一次可视化（改为子图形式）
# ==============================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# 第一子图：原始数据拟合
ax1.scatter(V_data, pH_data, color='red', label='实验数据', zorder=3)
V_fit_curve = model_func(pH_fine, ka_fit, cA_fit, Va_fit)
ax1.plot(V_fit_curve, pH_fine, 'b-', label='拟合曲线', linewidth=2, zorder=2)

# 添加标注信息
textstr = '\n'.join((
    f'Ka = {ka_fit:.4e}',
    f'pKa = {-np.log10(ka_fit):.2f}',
    f'cA-origin = {cA_fit*Va_fit/25:.4f} mol/L',
    f'R² = {r_squared:.4f}'))
ax1.text(0.95, 0.15, textstr, transform=ax1.transAxes,
        fontsize=10, verticalalignment='top', 
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax1.set_xlabel("加入NaOH体积 V (mL)", fontsize=12)
ax1.set_ylabel("溶液pH", fontsize=12)
ax1.set_title("原始数据拟合结果", fontsize=14)
ax1.set_xlim(-1, 35)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# [...] 保留异常点剔除和第二次拟合代码，直至第二次结果输出 [...]

# ==============================
# 第二次可视化（改为子图形式）
# ==============================
# 第二子图：清洗后数据拟合
ax2.scatter(V_data_clean, pH_data_clean, color='red', 
           label='清洗后数据', zorder=3)
V_fit_curve_clean = model_func(pH_fine_clean, ka_fit_clean, 
                              cA_fit_clean, Va_fit_clean)
ax2.plot(V_fit_curve_clean, pH_fine_clean, 'b-', 
        label='优化拟合曲线', linewidth=2, zorder=2)

# 添加标注信息
textstr_clean = '\n'.join((
    f'Ka = {ka_fit_clean:.4e}',
    f'pKa = {-np.log10(ka_fit_clean):.2f}',
    f'cA-origin = {cA_fit_clean*Va_fit_clean/25:.4f} mol/L',
    f'R² = {r_squared_clean:.4f}'))
ax2.text(0.95, 0.15, textstr_clean, transform=ax2.transAxes,
        fontsize=10, verticalalignment='top', 
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax2.set_xlabel("加入NaOH体积 V (mL)", fontsize=12)
ax2.set_ylabel("溶液pH", fontsize=12)
ax2.set_title("剔除四个误差最大的点后拟合结果", fontsize=14)
ax2.set_xlim(-1, 35)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

# 调整整体布局
plt.tight_layout(w_pad=4)  # 增加子图横向间距
plt.show()
