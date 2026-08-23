import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class YLCalculator:
    def __init__(self):
        pass

    def calculate_contact_angle(self, valid_contour, baseline_y, r_guess):
        """
        使用 Young-Laplace 方程计算接触角（带数据清洗）
        """
        if len(valid_contour) < 10:
            raise ValueError("有效轮廓点太少")

        # 1. 坐标转换
        contour_y = valid_contour[:, 0]
        contour_x = valid_contour[:, 1]
        contour_z = baseline_y - contour_y  # z 向上为正
        
        # 找顶点
        idx_tip = np.argmax(contour_z)
        x_tip = contour_x[idx_tip]
        z_tip = contour_z[idx_tip]
        
        print(f"   原始轮廓点数: {len(valid_contour)}, 顶点高度: {z_tip:.1f}")

        sort_idx = np.argsort(contour_x)
        sorted_x = contour_x[sort_idx]
        sorted_z = contour_z[sort_idx]
        dx = np.diff(sorted_x)
        gaps = np.where(dx > 10)[0]
        
        if len(gaps) > 0:
            segments = []
            start = 0
            for gap in gaps:
                segments.append((start, gap))
                start = gap + 1
            segments.append((start, len(sorted_x)-1))
            
            # 找最长的段
            longest_seg = max(segments, key=lambda s: s[1]-s[0])
            sorted_x = sorted_x[longest_seg[0]:longest_seg[1]+1]
            sorted_z = sorted_z[longest_seg[0]:longest_seg[1]+1]
            print(f"   去噪后轮廓点数: {len(sorted_x)}")
        z_threshold = z_tip * 0.15 
        valid_mask = sorted_z > z_threshold
        
        x_clean = sorted_x[valid_mask]
        z_clean = sorted_z[valid_mask]
        
        print(f"   切除底部后点数: {len(x_clean)}, 拟合高度范围: {z_threshold:.1f} ~ {z_tip:.1f}")

        idx_tip_new = np.argmax(z_clean)
        x_tip_clean = x_clean[idx_tip_new]
        
        # 归一化
        x_norm = x_clean - x_tip_clean
        z_norm = z_clean - z_tip  # 顶点 z=0

        # 处理左右两侧
        angle_left = self._fit_one_side(x_norm, z_norm, side='left', r_guess=r_guess)
        angle_right = self._fit_one_side(x_norm, z_norm, side='right', r_guess=r_guess)
        
        # 平均
        angle_avg = (angle_left + angle_right) / 2
        print(f"   左侧接触角: {angle_left:.2f}°")
        print(f"   右侧接触角: {angle_right:.2f}°")
        print(f"   平均接触角: {angle_avg:.2f}°")
        
        return angle_avg

    def _fit_one_side(self, x_norm, z_norm, side='right', r_guess=20):
        """
        拟合单侧接触角
        """
        # 筛选单侧数据
        if side == 'right':
            mask = x_norm > 0
            x_side = x_norm[mask]
            z_side = z_norm[mask]
            # 保持 x 为正
        else:
            mask = x_norm < 0
            x_side = -x_norm[mask]  # 翻转为正，方便计算
            z_side = z_norm[mask]
        
        if len(x_side) < 5:
            print(f"   ⚠️ {side}侧数据点太少，返回默认值 90°")
            return 90.0
        
        # 按 x 排序
        sort_idx = np.argsort(x_side)
        x_side = x_side[sort_idx]
        z_side = z_side[sort_idx]
        z_phys = -z_side  # 翻转 z 轴
        
        # Y-L ODE
        def yl_ode(s, y, b):
            x, z, phi = y
            if x < 1e-6:
                return [1.0, 0.0, b]
            return [np.cos(phi), np.sin(phi), b - np.sin(phi)/x]

        def objective(params):
            b = params[0]
            if b <= 0:
                return [1e6]
            
            y0 = [1e-6, 0, 0.01]
            s_span = (0, np.max(x_side) * 1.5)
            
            try:
                sol = solve_ivp(yl_ode, s_span, y0, args=(b,), method='RK45', dense_output=True)
                
                if not sol.success or len(sol.t) < 10:
                    return np.full(len(x_side), 1e6)
                
                theo_z = sol.sol(x_side)[1]
                return theo_z - z_phys
            except:
                return np.full(len(x_side), 1e6)

        # 初始猜测
        R_guess = (np.max(x_side) + np.max(z_phys)) / 2
        if R_guess < 5:
            R_guess = r_guess
        
        p0 = [2.0 / R_guess]
        
        try:
            res = least_squares(objective, p0, bounds=([0], [np.inf]))
            b_opt = res.x[0]
        except:
            print(f"   ⚠️ {side}侧拟合失败，返回默认值 90°")
            return 90.0
        
        # 计算接触角
        try:
            sol_final = solve_ivp(yl_ode, (0, np.max(x_side)), [1e-6, 0, 0.01], args=(b_opt,), method='RK45')
            phi_at_end = sol_final.y[2, -1]
            
            # 接触角 = phi（弧度转角度）
            angle_deg = np.degrees(phi_at_end)
            
            # 修正范围
            if angle_deg < 0:
                angle_deg += 180
            if angle_deg > 180:
                angle_deg = 360 - angle_deg
            
            return angle_deg
        except:
            print(f"   ⚠️ {side}侧角度计算失败，返回默认值 90°")
            return 90.0

    def visualize_fit(self, valid_contour, baseline_y, angle):
        """可视化拟合结果"""
        contour_y = valid_contour[:, 0]
        contour_x = valid_contour[:, 1]
        contour_z = baseline_y - contour_y
        
        idx_tip = np.argmax(contour_z)
        x_tip = contour_x[idx_tip]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 画原始轮廓点
        ax.plot(contour_x, contour_z, 'bo', markersize=3, label='Contour Points')
        
        # 画顶点
        ax.plot(x_tip, contour_z[idx_tip], 'r*', markersize=15, label='Tip')
        
        # 画基准线
        ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Baseline (z=0)')
        
        # 标注角度
        ax.text(
            x_tip + 10, contour_z[idx_tip] - 10,
            f'Contact Angle: {angle:.1f}°',
            color='red', fontsize=14, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='red')
        )
        
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('z (pixels, upward)')
        ax.set_title('Young-Laplace Fit Result')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()


# ==========================================
# 测试入口
# ==========================================
if __name__ == "__main__":
    print("这是 YLCalculator 模块，请从主程序调用。")
