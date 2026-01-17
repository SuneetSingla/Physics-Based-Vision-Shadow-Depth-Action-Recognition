import matplotlib.pyplot as plt
import numpy as np

def plot_depth_vs_shadow_relationship(depth_history, shadow_area_history):
    """
    depth ∝ 1/sqrt(shadow_area)
    """
    if len(depth_history) < 10:
        print("[WARN] Not enough data for analytics")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Depth over time
    axes[0].plot(depth_history, linewidth=2, color='blue')
    axes[0].set_title('Hand-to-Face Distance Over Time')
    axes[0].set_xlabel('Frame Number')
    axes[0].set_ylabel('Distance (cm)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Depth vs Shadow Area (scatter + trend)
    axes[1].scatter(shadow_area_history, depth_history, alpha=0.5, s=10)
    
    shadow_arr = np.array(shadow_area_history)
    depth_arr = np.array(depth_history)
    
    valid = (depth_arr < 50) & (shadow_arr > 0)
    shadow_clean = shadow_arr[valid]
    depth_clean = depth_arr[valid]
    
    if len(shadow_clean) > 10:
        sqrt_shadow = np.sqrt(shadow_clean)
        k_fit = np.mean(depth_clean * sqrt_shadow)
        
        shadow_range = np.linspace(shadow_clean.min(), shadow_clean.max(), 100)
        depth_fit = k_fit / np.sqrt(shadow_range)
        axes[1].plot(shadow_range, depth_fit, 'r--', linewidth=2, 
                    label=f'Fitted: d = {k_fit:.1f}/√A')
    
    axes[1].set_title('Depth vs Shadow Area (Inverse Square Relationship)')
    axes[1].set_xlabel('Shadow Area (pixels)')
    axes[1].set_ylabel('Distance (cm)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analytics_report.png', dpi=150)
    print("[INFO] Analytics saved as analytics_report.png")
    plt.show()