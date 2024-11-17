from typing import List, Tuple, Set
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from copy import deepcopy
import time


# 保持原有的 Node 类和辅助函数不变
class Node:
    def __init__(self, position: Tuple[int, int], g_cost: float, h_cost: float, parent=None):
        self.position = position
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost


def manhattan_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


def get_neighbors(node: Node, grid: List[List[int]]) -> List[Tuple[int, int]]:
    rows, cols = len(grid), len(grid[0])
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for dx, dy in directions:
        new_x = node.position[0] + dx
        new_y = node.position[1] + dy

        if (0 <= new_x < rows and
                0 <= new_y < cols and
                grid[new_x][new_y] == 0):
            neighbors.append((new_x, new_y))

    return neighbors


def reconstruct_path(end_node: Node) -> List[Tuple[int, int]]:
    path = []
    current = end_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]


# 新的可视化相关函数
def create_visualization(grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]):
    """创建初始化的图形对象"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks(np.arange(-0.5, len(grid[0]), 1), [])
    ax.set_yticks(np.arange(-0.5, len(grid), 1), [])
    return fig, ax


def update_visualization(grid, path, start, end, explored_grid, current_position, ax):
    """更新图形，而不是重新创建"""
    # 保持之前的图形状态
    ax.collections.clear()
    ax.patches.clear()
    ax.lines.clear()

    # 显示基础地图
    ax.imshow(np.array(grid), cmap='binary')

    # 绘制已探索区域
    explored_area = np.ma.masked_where(np.logical_not(explored_grid), np.ones_like(grid))
    ax.imshow(explored_area, cmap='Greens', alpha=1, vmin=1, vmax=1)

    # 绘制起点和终点
    ax.plot(start[1], start[0], 'go', markersize=15, label='Start')
    ax.plot(end[1], end[0], 'bo', markersize=15, label='End')

    # 绘制当前位置
    ax.plot(current_position[1], current_position[0], 'ro', markersize=10, label='Current')

    # 如果路径非空，绘制路径和箭头
    if path:
        ax.plot([p[1] for p in path], [p[0] for p in path], 'r-', linewidth=2, label='Path')
        for i in range(len(path) - 1):
            dx = path[i + 1][1] - path[i][1]
            dy = path[i + 1][0] - path[i][0]
            ax.arrow(path[i][1], path[i][0], dx * 0.3, dy * 0.3,
                     head_width=0.2, head_length=0.2, fc='r', ec='r')

    ax.grid(True)
    ax.set_title("A* Pathfinding")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()


def play_animation(maze, start_pos, end_pos):
    """使用 FuncAnimation 播放 A* 的动画"""
    # 初始化变量
    explored_grid = [[False] * len(maze[0]) for _ in range(len(maze))]
    animation_frames = []

    def collect_frames():
        start_node = Node(start_pos, 0, manhattan_distance(start_pos, end_pos))
        open_list = [start_node]
        closed_set = set()
        current_path = []

        while open_list:
            current_node = heapq.heappop(open_list)
            current_pos = current_node.position
            explored_grid[current_pos[0]][current_pos[1]] = True

            if current_pos == end_pos:
                current_path = reconstruct_path(current_node)
                animation_frames.append({
                    'position': current_pos,
                    'path': current_path.copy(),
                    'explored': [row.copy() for row in explored_grid]
                })
                return current_path

            closed_set.add(current_pos)
            current_path = reconstruct_path(current_node)

            animation_frames.append({
                'position': current_pos,
                'path': current_path.copy(),
                'explored': [row.copy() for row in explored_grid]
            })

            for neighbor_pos in get_neighbors(current_node, maze):
                if neighbor_pos in closed_set:
                    continue

                g_cost = current_node.g_cost + 1
                h_cost = manhattan_distance(neighbor_pos, end_pos)
                neighbor_node = Node(neighbor_pos, g_cost, h_cost, current_node)

                should_add = True
                for i, open_node in enumerate(open_list):
                    if open_node.position == neighbor_pos:
                        if open_node.g_cost > g_cost:
                            open_list[i] = neighbor_node
                        should_add = False
                        break

                if should_add:
                    heapq.heappush(open_list, neighbor_node)

        return []

    collect_frames()

    # 使用Figure 1绘制
    fig1, ax1 = plt.subplots(figsize=(8, 8), num="Figure 1: A* Visualization")

    def update(frame_idx):
        frame = animation_frames[frame_idx]
        update_visualization(maze, frame['path'], start_pos, end_pos,
                             frame['explored'], frame['position'], ax1)

    anim = FuncAnimation(
        fig1,
        update,
        frames=len(animation_frames),
        interval=2,  # 增加间隔到2ms
        repeat=False,
        blit=False
    )

    plt.show()

#新的创建迷宫函数
def create_maze(width, height, start, end):
    # 初始化迷宫矩阵
    maze = [[(i + j) % 2 for j in range(width)] for i in range(height)]

    # 设置最外层为墙
    for i in range(width):
        maze[0][i] = 1
        maze[height - 1][i] = 1
    for i in range(height):
        maze[i][0] = 1
        maze[i][width - 1] = 1

    # 深度优先搜索查找路径并打通墙
    def dfs(x, y, path):
        if (x, y) == end:
            path.append((x, y))
            return True

        # 标记当前点为路径
        maze[x][y] = 2
        path.append((x, y))

        # 定义随机化移动方向（上、右、下、左）
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)  # 随机打乱方向

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # 检查是否是最外层墙，如果是则跳过此方向
            if nx == 0 or ny == 0 or nx == height - 1 or ny == width - 1:
                continue

            # 如果遇到墙，则打通墙
            if maze[nx][ny] == 1:
                maze[nx][ny] = 0

            # 如果是通路，继续递归
            if maze[nx][ny] == 0:
                if dfs(nx, ny, path):
                    return True

        # 回溯
        path.pop()
        maze[x][y] = 0
        return False

    path = []
    dfs(start[0], start[1], path)

    # 随机改变不属于路径的0和1
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if maze[i][j] not in (1, 2):
                maze[i][j] = random.choice([0, 1])

    # 将路径上的点还原为0
    for x, y in path:
        maze[x][y] = 0

    return maze

class RRTNode:
    def __init__(self, position: Tuple[int, int], parent=None):
        self.position = position
        self.parent = parent

def distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def is_collision_free(grid: List[List[int]], point1: Tuple[int, int], point2: Tuple[int, int]) -> bool:
    x1, y1 = point1
    x2, y2 = point2

    # Linearly interpolate between the points and check for collisions
    num_samples = int(distance(point1, point2) * 10)
    for i in range(num_samples):
        t = i / num_samples
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        if grid[x][y] == 1:
            return False
    return True

def rrt_pathfinding(grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int], max_iterations=1000, step_size=2):
    rows, cols = len(grid), len(grid[0])
    nodes = [RRTNode(start)]
    path = []

    for _ in range(max_iterations):
        # Generate a random point
        rand_point = (random.randint(0, rows - 1), random.randint(0, cols - 1))

        # Find the nearest node
        nearest_node = min(nodes, key=lambda node: distance(node.position, rand_point))

        # Move towards the random point
        direction = np.array(rand_point) - np.array(nearest_node.position)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 0 else direction
        new_point = tuple(map(int, np.array(nearest_node.position) + direction * step_size))

        # Check if the new point is valid
        if 0 <= new_point[0] < rows and 0 <= new_point[1] < cols and grid[new_point[0]][new_point[1]] == 0:
            if is_collision_free(grid, nearest_node.position, new_point):
                new_node = RRTNode(new_point, nearest_node)
                nodes.append(new_node)

                # Check if the new point is near the goal
                if distance(new_point, end) <= step_size:
                    goal_node = RRTNode(end, new_node)
                    nodes.append(goal_node)

                    # Reconstruct the path
                    current = goal_node
                    while current:
                        path.append(current.position)
                        current = current.parent
                    return path[::-1], nodes

    return [], nodes

def visualize_rrt(grid, nodes, path, start, end):
    """RRT的可视化，使用Figure 2"""
    fig2, ax2 = plt.subplots(figsize=(8, 8), num="Figure 2: RRT Visualization")
    ax2.imshow(grid, cmap="binary")

    for node in nodes:
        if node.parent:
            ax2.plot(
                [node.position[1], node.parent.position[1]],
                [node.position[0], node.parent.position[0]],
                "g-",
                linewidth=1,
            )

    ax2.plot(start[1], start[0], "ro", markersize=10, label="Start")
    ax2.plot(end[1], end[0], "bo", markersize=10, label="End")

    if path:
        ax2.plot([p[1] for p in path], [p[0] for p in path], "r-", linewidth=2, label="Path")

    ax2.set_title("RRT Pathfinding")
    plt.legend()
    plt.show()

# 主程序
if __name__ == "__main__":
    width = 15
    height = 15
    start_pos = (1, 1)
    end_pos = (13, 13)

    maze = create_maze(width, height, start_pos, end_pos)

    # 在Figure 1中显示A*算法
    play_animation(maze, start_pos, end_pos)

    # 在Figure 2中显示RRT算法
    rrt_path, rrt_nodes = rrt_pathfinding(maze, start_pos, end_pos)
    visualize_rrt(maze, rrt_nodes, rrt_path, start_pos, end_pos)