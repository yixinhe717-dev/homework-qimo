# -*- coding: utf-8 -*-


import sys
import time
import threading
import numpy as np
from collections import deque
from PIL import Image, ImageFilter, ImageDraw


def path_grid_statistics(path):
    """返回路径统计量：经过格子数、移动步数。

    path 是 [(row, col), ...]。
    经过格子数 = len(path)，包含起点与终点；
    移动步数 = len(path) - 1，是真正走了多少步。
    """
    if path is None:
        return 0, 0
    cell_count = len(path)
    move_steps = max(0, cell_count - 1)
    return cell_count, move_steps


# ===========================================================================
# 1. 迷宫识别：把图片转成 0/1 占据栅格（自适应，不依赖写死的尺寸）
# ===========================================================================
class MazeExtractor:


    def __init__(self, image_path, wall_threshold=65, blur_radius=6):
        self.image_path = image_path
        self.wall_threshold = wall_threshold
        self.blur_radius = blur_radius
        self.grid = None          # 2D int array, 1=墙 0=通道
        self.start = None         # (row, col)
        self.rows = None          # 每一行点阵的像素 y 坐标
        self.cols = None          # 每一列点阵的像素 x 坐标

    # --- 工具：在亮度投影上求周期 ---
    @staticmethod
    def _period(signal, lo=38, hi=52):
        s = signal - signal.mean()
        best_v, best_p = -1.0, lo
        for p in range(lo, hi):
            a, c = s[:-p], s[p:]
            denom = np.sqrt((a * a).sum() * (c * c).sum()) + 1e-9
            v = (a * c).sum() / denom
            if v > best_v:
                best_v, best_p = v, p
        return best_p

    # --- 工具：梳状匹配求相位与精确周期 ---
    @staticmethod
    def _comb(profile, period):
        L = len(profile)
        best = (-1.0, 0.0, float(period))
        for pf in np.arange(period - 1.5, period + 1.5, 0.1):
            for off in np.arange(0, pf, 1.0):
                idx = np.clip(np.round(np.arange(off, L, pf)).astype(int), 0, L - 1)
                score = profile[idx].mean()
                if score > best[0]:
                    best = (score, off, pf)
        return best[1], best[2]      # offset, period

    def extract(self):
        im = Image.open(self.image_path).convert("L")
        W, H = im.size
        blurred = np.array(im.filter(ImageFilter.GaussianBlur(self.blur_radius))).astype(float)
        wall = (blurred < self.wall_threshold).astype(float)

        col_proj = wall.mean(axis=0)
        row_proj = wall.mean(axis=1)
        pc = self._period(col_proj)
        pr = self._period(row_proj)
        off_c, per_c = self._comb(col_proj, pc)
        off_r, per_r = self._comb(row_proj, pr)
        unit_c, unit_r = per_c / 2.0, per_r / 2.0   # 基本点阵单元

        cols = [off_c + j * unit_c for j in range(2000) if off_c + j * unit_c < W]
        rows = [off_r + i * unit_r for i in range(2000) if off_r + i * unit_r < H]
        nrow, ncol = len(rows), len(cols)

        # 采样每个点阵交点中心邻域，投票判定墙/通
        grid = np.zeros((nrow, ncol), dtype=int)
        half = max(3, int(round(min(unit_c, unit_r) / 3)))
        for i, y in enumerate(rows):
            yy = int(round(y))
            for j, x in enumerate(cols):
                xx = int(round(x))
                win = blurred[max(0, yy - half):yy + half + 1,
                              max(0, xx - half):xx + half + 1]
                grid[i, j] = 1 if win.mean() < self.wall_threshold else 0

        # 找黄色出发点
        rgb = np.array(Image.open(self.image_path).convert("RGB")).astype(int)
        R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        ymask = (R > 150) & (G > 150) & (B < 120) & (np.abs(R - G) < 80)
        ys, xs = np.where(ymask)
        if len(xs) == 0:
            raise RuntimeError("未在图片中找到黄色出发点。")
        sx, sy = xs.mean(), ys.mean()
        sc = int(round((sx - off_c) / unit_c))
        sr = int(round((sy - off_r) / unit_r))
        sr = min(max(sr, 0), nrow - 1)
        sc = min(max(sc, 0), ncol - 1)

        # 出发点若落在墙上，吸附到最近的通道格
        if grid[sr, sc] == 1:
            for radius in range(1, 4):
                found = False
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        r, c = sr + dr, sc + dc
                        if 0 <= r < nrow and 0 <= c < ncol and grid[r, c] == 0:
                            sr, sc = r, c
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        self.grid, self.start = grid, (sr, sc)
        self.rows, self.cols = rows, cols
        return grid, (sr, sc)

    # 栅格坐标 <-> 像素坐标
    def cell_to_pixel(self, cell):
        r, c = cell
        return self.cols[c], self.rows[r]

    def pixel_to_cell(self, x, y):
        c = int(round((x - self.cols[0]) / (self.cols[1] - self.cols[0])))
        r = int(round((y - self.rows[0]) / (self.rows[1] - self.rows[0])))
        nrow, ncol = self.grid.shape
        return min(max(r, 0), nrow - 1), min(max(c, 0), ncol - 1)


# ===========================================================================
# 2. 强化学习求解器
# ===========================================================================
class MazeRLSolver:
    """
    把迷宫建模成确定性 MDP：
        状态 s = 通道格 (r, c)
        动作 a ∈ {上, 下, 左, 右}
        转移 = 走到相邻通道格（撞墙则停在原地）
        奖励 = 每走一步 -1，到达终点结束（这样“累计奖励最大”⇔“步数最少”）

    两种价值型强化学习方法，共用 Bellman 最优更新：
        Q(s,a) ← Q(s,a) + α [ r + γ·maxₐ' Q(s',a') − Q(s,a) ]

    (A) value_iteration —— 已知模型，对全状态做同步 Bellman 扫描，
        毫秒级收敛到精确最优，并能直接判定不可达（界面默认使用）。
    (B) q_learning —— 无模型，ε-贪心“试错交互”地学习（演示真实 RL 过程，
        到达终点后沿轨迹反向回放以加速价值回传）。
    """

    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # 上 下 左 右
    NEG = -1e9

    def __init__(self, grid):
        self.grid = grid
        self.nrow, self.ncol = grid.shape
        self.free = (grid == 0)
        # 通道格扁平化索引 + 转移表（向量化用）
        self.cells = [(int(r), int(c)) for r, c in np.argwhere(self.free)]
        self.N = len(self.cells)
        self.idx = -np.ones((self.nrow, self.ncol), dtype=int)
        for i, (r, c) in enumerate(self.cells):
            self.idx[r, c] = i
        self.trans = np.zeros((self.N, 4), dtype=int)
        for i, (r, c) in enumerate(self.cells):
            for a, (dr, dc) in enumerate(self.ACTIONS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.nrow and 0 <= nc < self.ncol and self.free[nr, nc]:
                    self.trans[i, a] = self.idx[nr, nc]
                else:
                    self.trans[i, a] = i      # 撞墙 -> 停留
        self.V = None
        self.Q = None
        self.goal = None
        self.method = None

    # ---------------- (A) 值迭代（向量化，精确、瞬时） ----------------
    def value_iteration(self, goal, gamma=1.0, step_cost=-1.0, tol=1e-6, max_iter=5000):
        gi = self.idx[goal]
        V = np.full(self.N, self.NEG)
        V[gi] = 0.0
        sweeps = 0
        for sweeps in range(1, max_iter + 1):
            Q = step_cost + gamma * V[self.trans]      # (N,4)
            newV = Q.max(axis=1)
            newV[gi] = 0.0
            if np.max(np.abs(newV - V)) < tol:
                V = newV
                break
            V = newV
        self.V, self.Q, self.goal, self.method = V, Q, goal, "值迭代(value-based RL)"
        return sweeps

    # ---------------- (B) 无模型 Q-Learning（ε-贪心 + 反向回放） ----------------
    def q_learning(self, goal, episodes=600, gamma=0.99, alpha=0.5,
                   eps_decay=0.985, max_steps=4000, start=None,
                   progress_cb=None, max_episodes=8000):
        """无模型 Q-Learning。先用 BFS 判定可达性：
        - 不可达：直接返回（V 保持未更新，shortest_path 会得到 None）。
        - 可达：分批训练，每批结束后检查“贪心策略能否真正走到终点”，
          若还不能就继续训练（沿用同一张 Q 表与衰减后的 ε），直到成功或达到 max_episodes 上限。
        这样无论终点远近都能稳定收敛，不必让用户手动调 episodes。"""
        import random
        nrow, ncol, free, ACT = self.nrow, self.ncol, self.free, self.ACTIONS
        start = start if start is not None else self.cells[0]

        # 终点不可达：无需训练，写入“全未更新”的价值表，交给 shortest_path 报告不可达
        if not self.is_reachable(start, goal):
            self.V = np.full(self.N, self.NEG)
            self.Q = np.full((self.N, 4), self.NEG)
            self.goal, self.method = goal, "无模型 Q-Learning"
            return self.Q

        Q = np.zeros((nrow, ncol, 4))
        eps = 1.0
        ep_done = 0

        def commit():
            """把 (r,c,a) 形态的 Q 转成扁平 V/Q，供统一取路逻辑使用。"""
            Vflat = np.full(self.N, self.NEG)
            Qflat = np.full((self.N, 4), self.NEG)
            for i, (r, c) in enumerate(self.cells):
                Qflat[i] = Q[r, c]
                Vflat[i] = Q[r, c].max()
            Vflat[self.idx[goal]] = 0.0
            self.V, self.Q, self.goal, self.method = Vflat, Qflat, goal, "无模型 Q-Learning"

        def greedy_reaches_goal():
            """用当前策略做一次贪心 rollout，检查能否真正到达终点。"""
            commit()
            return self.shortest_path(start) is not None

        # 分批训练，直到贪心策略能走到终点（或到达上限）
        while True:
            for ep in range(episodes):
                s, traj = start, []
                for _ in range(max_steps):
                    if s == goal:
                        break
                    if random.random() < eps:
                        a = random.randint(0, 3)
                    else:
                        a = int(np.argmax(Q[s[0], s[1]]))
                    dr, dc = ACT[a]
                    n = (s[0] + dr, s[1] + dc)
                    if not (0 <= n[0] < nrow and 0 <= n[1] < ncol and free[n[0], n[1]]):
                        r, n = -1.0, s
                    elif n == goal:
                        r = 0.0
                    else:
                        r = -1.0
                    best = 0.0 if n == goal else np.max(Q[n[0], n[1]])
                    Q[s[0], s[1], a] += alpha * (r + gamma * best - Q[s[0], s[1], a])
                    traj.append((s, a, r, n))
                    s = n
                    if s == goal:
                        break
                if s == goal:        # 反向回放整条成功轨迹，加速价值回传
                    for ss, aa, rr, nn in reversed(traj):
                        best = 0.0 if nn == goal else np.max(Q[nn[0], nn[1]])
                        Q[ss[0], ss[1], aa] += alpha * (rr + gamma * best - Q[ss[0], ss[1], aa])
                eps = max(0.05, eps * eps_decay)
                ep_done += 1
                if progress_cb and ep_done % 10 == 0:
                    progress_cb(min(ep_done, max_episodes), max_episodes)

            if greedy_reaches_goal() or ep_done >= max_episodes:
                break
            # 未收敛：再训练一批，并适当回升 ε 以鼓励继续探索更远区域
            eps = max(eps, 0.3)

        commit()
        if progress_cb:
            progress_cb(max_episodes, max_episodes)
        return Q

    # ---------------- 从学到的策略提取路径 ----------------
    def shortest_path(self, start):
        """返回最短路径 [(r,c),...]；若不可达返回 None。"""
        if self.V is None:
            raise RuntimeError("请先调用 value_iteration 或 q_learning。")
        if not self.free[start]:
            return None
        si = self.idx[start]
        if self.V[si] <= self.NEG / 2:          # 价值从未被更新 -> 不可达
            return None
        cur = si
        path = [self.cells[cur]]
        seen = {cur}
        for _ in range(self.N + 5):
            if self.cells[cur] == self.goal:
                return path
            moved = False
            for a in np.argsort(-self.Q[cur]):
                cand = self.trans[cur, a]
                if cand != cur and cand not in seen:
                    cur = cand
                    path.append(self.cells[cur])
                    seen.add(cur)
                    moved = True
                    break
            if not moved:
                return None
        return None

    # 用 BFS 独立校验可达性（作为“是否可到达”的真值）
    def is_reachable(self, start, goal):
        if not (self.free[start] and self.free[goal]):
            return False
        q = deque([start])
        seen = {start}
        while q:
            cur = q.popleft()
            if cur == goal:
                return True
            for dr, dc in self.ACTIONS:
                n = (cur[0] + dr, cur[1] + dc)
                if (0 <= n[0] < self.nrow and 0 <= n[1] < self.ncol
                        and self.free[n] and n not in seen):
                    seen.add(n)
                    q.append(n)
        return False


# ===========================================================================
# 3. 交互界面 —— tkinter（首选）
# ===========================================================================
class MazeGUITk:
    def __init__(self, extractor, solver):
        import tkinter as tk
        from PIL import ImageTk
        self.tk = tk
        self.ImageTk = ImageTk
        self.ex = extractor
        self.solver = solver
        self.base_img = Image.open(self.ex.image_path).convert("RGB")
        self.W, self.H = self.base_img.size

        self.root = tk.Tk()
        self.root.title("迷宫强化学习最短路径求解器")

        self.max_w = 1100
        self.scale = min(1.0, self.max_w / self.W)
        self.disp_w = int(self.W * self.scale)
        self.disp_h = int(self.H * self.scale)

        bar = tk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        tk.Label(bar, text="算法:").pack(side=tk.LEFT)
        self.method = tk.StringVar(value="vi")
        tk.Radiobutton(bar, text="值迭代(瞬时·精确)", variable=self.method, value="vi").pack(side=tk.LEFT)
        tk.Radiobutton(bar, text="Q-Learning(无模型·学习过程)", variable=self.method, value="ql").pack(side=tk.LEFT)
        self.show_heat = tk.IntVar(value=0)
        tk.Checkbutton(bar, text="显示价值热力图", variable=self.show_heat,
                       command=self._redraw_last).pack(side=tk.LEFT, padx=8)
        tk.Button(bar, text="清除", command=self.clear).pack(side=tk.LEFT, padx=4)
        tk.Label(bar, text="（绿点=起点，点击任意格子设为终点）").pack(side=tk.LEFT, padx=8)

        self.status = tk.StringVar(value="就绪。点击迷宫中任意位置作为终点。")
        tk.Label(self.root, textvariable=self.status, anchor="w",
                 relief=tk.SUNKEN).pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas = tk.Canvas(self.root, width=self.disp_w, height=self.disp_h,
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        self._set_bg(self.base_img)
        self._draw_start()
        self.last_goal = None
        self.plan_count = 0

    # ---- 画布工具 ----
    def _set_bg(self, pil_img):
        disp = pil_img.resize((self.disp_w, self.disp_h))
        self.bg = self.ImageTk.PhotoImage(disp)
        self.canvas.delete("bg")
        self.canvas.create_image(0, 0, anchor="nw", image=self.bg, tags="bg")
        self.canvas.tag_lower("bg")

    def _px(self, cell):
        x, y = self.ex.cell_to_pixel(cell)
        return x * self.scale, y * self.scale

    def _dot(self, cell, color, tag, r=7):
        x, y = self._px(cell)
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color,
                                outline="white", width=2, tags=tag)

    def _draw_start(self):
        self._dot(self.ex.start, "#00cc44", "start")

    def clear(self):
        self.canvas.delete("path")
        self.canvas.delete("goal")
        self._set_bg(self.base_img)
        self._draw_start()
        self.last_goal = None
        self.status.set("已清除。点击迷宫中任意位置作为终点。")

    # ---- 价值热力图叠加 ----
    def _heat_overlay(self):
        V = self.solver.V
        finite = V[V > self.solver.NEG / 2]
        if finite.size == 0:
            return self.base_img
        vmin, vmax = finite.min(), finite.max()
        overlay = self.base_img.copy()
        px = overlay.load()
        unit = self.ex.cols[1] - self.ex.cols[0]
        half = max(2, int(unit / 2))
        for i, (r, c) in enumerate(self.solver.cells):
            if V[i] <= self.solver.NEG / 2:
                continue
            t = 0.0 if vmax == vmin else (V[i] - vmin) / (vmax - vmin)  # 1=近终点
            col = (int(255 * (1 - t)), int(120 + 100 * t), int(60 + 180 * t))
            cx, cy = int(self.ex.cols[c]), int(self.ex.rows[r])
            for yy in range(max(0, cy - half), min(self.H, cy + half)):
                for xx in range(max(0, cx - half), min(self.W, cx + half)):
                    o = px[xx, yy]
                    px[xx, yy] = (int(o[0] * .45 + col[0] * .55),
                                  int(o[1] * .45 + col[1] * .55),
                                  int(o[2] * .45 + col[2] * .55))
        return overlay

    def _redraw_last(self):
        if self.last_goal is not None:
            self.solve(self.last_goal)

    # ---- 点击 -> 求解 ----
    def on_click(self, event):
        x = event.x / self.scale
        y = event.y / self.scale
        cell = self.ex.pixel_to_cell(x, y)
        self.solve(cell)

    def solve(self, goal):
        nrow, ncol = self.solver.grid.shape
        # 终点若在墙上，吸附到最近通道格
        if self.solver.grid[goal] == 1:
            best, bestd = None, 1e9
            gr, gc = goal
            for r, c in self.solver.cells:
                d = (r - gr) ** 2 + (c - gc) ** 2
                if d < bestd:
                    bestd, best = d, (r, c)
            goal = best
        self.last_goal = goal
        self.canvas.delete("path")
        self.canvas.delete("goal")

        if self.method.get() == "ql":
            self.status.set("Q-Learning 正在通过试错学习……")
            self.root.update()
            threading.Thread(target=self._solve_ql, args=(goal,), daemon=True).start()
        else:
            t0 = time.time()
            sweeps = self.solver.value_iteration(goal)
            self._finish(goal, t0, extra=f"值迭代 {sweeps} 次扫描")

    def _solve_ql(self, goal):
        t0 = time.time()

        def cb(done, total):
            self.status.set(f"Q-Learning 学习中… {done}/{total} 回合")
        self.solver.q_learning(goal, start=self.ex.start, progress_cb=cb)
        self.root.after(0, lambda: self._finish(goal, t0, extra="无模型 Q-Learning"))

    def _finish(self, goal, t0, extra=""):
        if self.show_heat.get():
            self._set_bg(self._heat_overlay())
        else:
            self._set_bg(self.base_img)
        self._draw_start()
        reachable = self.solver.is_reachable(self.ex.start, goal)
        path = self.solver.shortest_path(self.ex.start) if reachable else None
        dt = time.time() - t0
        if path is None:
            self._dot(goal, "#888888", "goal")
            self.plan_count += 1
            report = f"第 {self.plan_count} 次规划：起点 {self.ex.start} -> 终点 {goal} 不可到达。"
            print(report)
            self.status.set(f"终点 (行{goal[0]}, 列{goal[1]}) 不可到达。路径格子数：0。（{extra}，用时 {dt:.2f}s）")
            return
        pts = []
        for cell in path:
            x, y = self._px(cell)
            pts += [x, y]
        if len(pts) >= 4:
            self.canvas.create_line(*pts, fill="#ff2424", width=4,
                                    capstyle="round", joinstyle="round", tags="path")
        self._dot(goal, "#2a6bff", "goal")
        self._dot(self.ex.start, "#00cc44", "start")

        cell_count, move_steps = path_grid_statistics(path)
        self.plan_count += 1
        report = (f"第 {self.plan_count} 条路径：起点 {self.ex.start} -> 终点 {goal}，"
                  f"经过 {cell_count} 个格子（含起点和终点），移动 {move_steps} 步。")
        print(report)
        self.status.set(f"终点 (行{goal[0]}, 列{goal[1]})｜路径格子数 {cell_count} 个"
                        f"（含起点和终点）｜移动步数 {move_steps}｜{extra}，用时 {dt:.2f}s")

    def run(self):
        self.root.mainloop()


# ===========================================================================
# 3'. 交互界面 —— matplotlib（无 tkinter 时自动回退）
# ===========================================================================
def run_gui_matplotlib(extractor, solver):
    import matplotlib
    import matplotlib.pyplot as plt
    base = np.array(Image.open(extractor.image_path).convert("RGB"))
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(base)
    ax.set_title("点击任意位置设为终点（绿=起点 蓝=终点 红=最短路径）", fontsize=11)
    ax.axis("off")
    sx, sy = extractor.cell_to_pixel(extractor.start)
    ax.plot(sx, sy, "o", color="#00cc44", ms=12, mec="white", mew=2)
    drawn = {"line": None, "goal": None}
    path_counter = {"n": 0}
    info = ax.text(0.01, 0.99, "就绪", transform=ax.transAxes, va="top",
                   fontsize=10, color="yellow",
                   bbox=dict(facecolor="black", alpha=0.6))

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        goal = extractor.pixel_to_cell(event.xdata, event.ydata)
        if solver.grid[goal] == 1:
            gr, gc = goal
            goal = min(solver.cells, key=lambda rc: (rc[0]-gr)**2 + (rc[1]-gc)**2)
        t0 = time.time()
        solver.value_iteration(goal)
        reachable = solver.is_reachable(extractor.start, goal)
        path = solver.shortest_path(extractor.start) if reachable else None
        for k in ("line", "goal"):
            if drawn[k] is not None:
                drawn[k].remove() if hasattr(drawn[k], "remove") else None
                drawn[k] = None
        path_counter["n"] += 1
        if path is None:
            report = f"第 {path_counter['n']} 次规划：起点 {extractor.start} -> 终点 {goal} 不可到达。路径格子数：0。"
            print(report)
            info.set_text(f"终点(行{goal[0]},列{goal[1]}) 不可到达；路径格子数 0")
        else:
            xs = [extractor.cell_to_pixel(c)[0] for c in path]
            ys = [extractor.cell_to_pixel(c)[1] for c in path]
            drawn["line"], = ax.plot(xs, ys, "-", color="#ff2424", lw=2.5)
            gx, gy = extractor.cell_to_pixel(goal)
            drawn["goal"], = ax.plot(gx, gy, "o", color="#2a6bff", ms=12, mec="white", mew=2)
            cell_count, move_steps = path_grid_statistics(path)
            report = (f"第 {path_counter['n']} 条路径：起点 {extractor.start} -> 终点 {goal}，"
                      f"经过 {cell_count} 个格子（含起点和终点），移动 {move_steps} 步。")
            print(report)
            info.set_text(f"路径格子数 {cell_count} 个，移动 {move_steps} 步，用时 {time.time()-t0:.3f}s")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 4. 无界面演示 / 入口
# ===========================================================================
def demo(extractor, solver, goal=None, out="solution.png"):
    if goal is None:
        goal = (1, 1)        # 默认演示终点：左上角附近
    if solver.grid[goal] == 1:
        goal = min(solver.cells, key=lambda rc: (rc[0]-goal[0])**2 + (rc[1]-goal[1])**2)
    t0 = time.time()
    sweeps = solver.value_iteration(goal)
    reachable = solver.is_reachable(extractor.start, goal)
    path = solver.shortest_path(extractor.start) if reachable else None
    img = Image.open(extractor.image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    if path is None:
        print(f"终点 {goal} 不可到达。")
    else:
        pts = [extractor.cell_to_pixel(c) for c in path]
        d.line(pts, fill=(255, 36, 36), width=6)
        sx, sy = extractor.cell_to_pixel(extractor.start)
        gx, gy = extractor.cell_to_pixel(goal)
        d.ellipse([sx-10, sy-10, sx+10, sy+10], fill=(0, 204, 68))
        d.ellipse([gx-10, gy-10, gx+10, gy+10], fill=(42, 107, 255))
        cell_count, move_steps = path_grid_statistics(path)
        print(f"起点 {extractor.start} -> 终点 {goal}：路径格子数 {cell_count} 个"
              f"（含起点和终点），移动 {move_steps} 步，"
              f"值迭代 {sweeps} 次扫描，用时 {time.time()-t0:.3f}s")
    img.save(out)
    print(f"已保存可视化结果到 {out}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    image_path = args[0] if args else "maze.jpg"

    print(f"正在识别迷宫：{image_path} ……")
    ex = MazeExtractor(image_path)
    grid, start = ex.extract()
    print(f"  迷宫栅格大小：{grid.shape[0]} 行 × {grid.shape[1]} 列，"
          f"通道格 {(grid==0).sum()} 个，出发点（黄色）={start}")
    solver = MazeRLSolver(grid)

    if "--demo" in flags:
        demo(ex, solver)
        return

    # 启动交互界面：优先 tkinter，失败则回退 matplotlib
    try:
        import tkinter  # noqa
        MazeGUITk(ex, solver).run()
    except Exception as e:
        print(f"（tkinter 不可用：{e}）改用 matplotlib 交互窗口。")
        try:
            run_gui_matplotlib(ex, solver)
        except Exception as e2:
            print(f"图形界面均不可用（{e2}），改为无界面演示。")
            demo(ex, solver)


if __name__ == "__main__":
    main()
