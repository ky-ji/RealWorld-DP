#!/usr/bin/env python3
"""测试 matplotlib 字体配置（优先复用项目内统一逻辑）"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 确保以 `python toolbox/test_fonts.py` 方式运行时也能 import 项目内模块
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from toolbox.mpl_fonts import setup_matplotlib_fonts

    setup_matplotlib_fonts(verbose=True)
except Exception as e:
    print(f"[字体] 警告: 字体配置失败（将继续测试）: {e}")

# 获取所有字体
all_fonts = [f.name for f in fm.fontManager.ttflist]

# 查找中文字体
chinese_keywords = ['wenquanyi', 'wqy', 'noto', 'simhei', 'yahei', 'source han', 'stheit', 'cjk']
chinese_fonts = []
for font in all_fonts:
    font_lower = font.lower()
    if any(keyword in font_lower for keyword in chinese_keywords):
        chinese_fonts.append(font)

print("="*60)
print("字体检测结果")
print("="*60)
print(f"\n总字体数: {len(all_fonts)}")
print(f"\n找到的中文字体 ({len(chinese_fonts)} 个):")
for font in sorted(set(chinese_fonts))[:20]:
    print(f"  - {font}")

# 输出当前 rcParams 便于诊断
print("\n" + "="*60)
print("当前 matplotlib 字体配置")
print("="*60)
print(f"font.family: {plt.rcParams.get('font.family')}")
print(f"font.sans-serif (前 5 个): {plt.rcParams.get('font.sans-serif')[:5]}")
print(f"axes.unicode_minus: {plt.rcParams.get('axes.unicode_minus')}")

# 测试显示中文
print("\n" + "="*60)
print("测试中文显示")
print("="*60)

fig, ax = plt.subplots(figsize=(6, 4))
ax.text(0.5, 0.5, '测试中文：动作空间', fontsize=20, ha='center', va='center')
ax.set_title('字体测试')
plt.tight_layout()
plt.savefig('/tmp/font_test.png', dpi=100)
print("测试图像已保存到: /tmp/font_test.png")
print("请检查图像中的中文是否正常显示")

