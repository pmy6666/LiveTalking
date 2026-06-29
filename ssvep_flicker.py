import os
import math
import numpy as np

# ==========================================
# 1. Linux OpenGL 环境配置（保留以确保画面正常显示）
# ==========================================
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["PYGLET_SHADOW_WINDOW"] = "0"

import pyglet
pyglet.options['shadow_window'] = False
temp_win = pyglet.window.Window(width=1, height=1, visible=False)
temp_win.switch_to()

from psychopy import prefs, visual, core, event
prefs.general['winType'] = 'pyglet'
prefs.general['autoLog'] = False

# ==========================================
# 2. 核心参数（仅保留闪烁与九宫格相关）
# ==========================================
# SSVEP 3频3相（保留原有的频率相位配置）
# FLICKER_FREQS = [7.5, 10.0, 12.0, 10.0, 12.0, 7.5, 12.0, 7.5, 10.0]
# 优化后的九宫格频率列表（相邻格子频率差最大）
FLICKER_FREQS = [
    12.8,  8.0, 14.4,  # 上排: 左, 中, 右
     9.6, 11.2, 13.6,  # 中排: 左, 中, 右
    15.2, 10.4,  8.8   # 下排: 左, 中, 右
]

# FLICKER_PHASES = [0.0, 0.666, 1.333, 1.333, 0.0, 0.666, 0.666, 1.333, 0.0]
FLICKER_PHASES = [0.0]*9
BLOCK_SIZE = 300

BLOCK_SPACING = 500

# ==========================================
# 3. 绘制一帧画面（仅保留九宫格闪烁）
# ==========================================
def draw_frame(win, flicker_blocks, lut_list, frame_cnt, result_text):
    for i in range(9):
        # 按LUT更新亮度实现闪烁
        inten = lut_list[i][frame_cnt % len(lut_list[i])]
        col = inten * 2 - 1
        flicker_blocks[i].fillColor = [col, col, col]
        flicker_blocks[i].draw()
    result_text.draw()
    win.flip()

# ==========================================
# 4. 主程序（仅保留闪烁显示与按键控制）
# ==========================================
def main():
    # 窗口初始化
    win = visual.Window(size=[1920,1080], fullscr=True, color=[-1,-1,-1], units='pix')
    actual_fps = win.getActualFrameRate() or 60

    # 生成闪烁LUT（查找表，用于控制每个格子的亮度变化）
    lut_len = 1000
    lut_list = []
    for f, p in zip(FLICKER_FREQS, FLICKER_PHASES):
        lut = [ (math.sin(2*math.pi*f*i/actual_fps + p*math.pi)+1)/2*0.9+0.1
               for i in range(lut_len) ]
        lut_list.append(lut)

    # 九宫格位置计算
    positions = []
    for row in range(3):
        for col in range(3):
            x = (col - 1) * BLOCK_SPACING
            y = (1 - row) * BLOCK_SPACING
            positions.append((x, y))

    # 创建九宫格闪烁方块
    flicker_blocks = [visual.Rect(win, width=BLOCK_SIZE, height=BLOCK_SIZE, pos=p) for p in positions]

    # 文字提示
    result_text = visual.TextStim(win, text="",
                                   pos=(0, 480), height=40, color=[1,1,1])

    frame_cnt = 0
    print("==== 按键说明 ====")
    print("Q/ESC：退出程序")
    print("R    ：重置闪烁状态")
    print("==================\n")

    while True:
        # 全局按键监听
        keys = event.getKeys()
        if 'q' in keys or 'escape' in keys:
            win.close()
            core.quit()
        if 'r' in keys:
            # 重置帧数，相当于重新开始闪烁周期
            frame_cnt = 0
            result_text.text = "已重置 | Q/ESC退出 | R重置"

        # 持续刷新闪烁画面
        draw_frame(win, flicker_blocks, lut_list, frame_cnt, result_text)
        frame_cnt += 1

if __name__ == "__main__":
    main()