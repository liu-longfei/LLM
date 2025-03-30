'''
leetcode:1
'''
from typing import Optional

class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 1. 两数之和
class Solution_1:
    @classmethod
    def two_sum(cls, nums: list, target: int):
        tmp_dict = {}
        for idx, val in enumerate(nums):
            if target - val not in tmp_dict:
                tmp_dict[val] = idx
            else:
                return [tmp_dict[target - val], idx]
        return []


print(Solution_1.two_sum([2, 7, 11, 15], 9))


class Solution_2:
    @classmethod
    def group_anagrams(cls, nums: list[str]):
        res_dict = {}
        for i in nums:
            i_sort = ''.join(sorted(i))
            if i_sort not in res_dict:
                res_dict[i_sort] = [i]
            else:
                res_dict[i_sort].append(i)

        res = []
        for key, val in res_dict.items():
            res.append(val)
        return res

    # 简单写法
    @classmethod
    def group_anagrams_2(cls, nums: list[str]):
        res_dict = {}
        for i in nums:
            sorted_i = ''.join(sorted(i))
            if sorted_i not in res_dict.keys():
                res_dict[sorted_i] = []
            res_dict[sorted_i].append(i)
        return list(res_dict.values())


print(Solution_2.group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
print(Solution_2.group_anagrams_2(["eat", "tea", "tan", "ate", "nat", "bat"]))


# 移动0
class Solution_3:
    @classmethod
    def move_zeros_1(cls, nums: list):
        l, r = 0, len(nums) - 1
        while l < r:
            if nums[l] == 0:
                for i in range(l, len(nums) - 1):
                    nums[i] = nums[i + 1]
                nums[-1] = 0
            l += 1
            r -= 1
        return nums

    @classmethod
    def move_zeros_2(cls, nums: list):
        l = r = 0
        while r < len(nums):
            if nums[r] != 0:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
            r += 1
        return nums


print(Solution_3.move_zeros_1([0, 1, 0, 3, 12]))
print(Solution_3.move_zeros_2([0, 1, 0, 3, 12]))


# 岛屿数量
class Solution_4:
    @classmethod
    def nums_is_lands(cls, grid: list[list]):

        def dfs(grid, visited, x, y):
            direction = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            for i, j in direction:
                next_x, next_y = x + i, y + j
                if next_x < 0 or next_x >= len(grid) or next_y < 0 or next_y >= len(grid[0]):
                    continue
                if (grid[next_x][next_y] == '1' or grid[next_x][next_y] == 1) and visited[next_x][next_y] == False:
                    visited[next_x][next_y] = True
                    dfs(grid, visited, next_x, next_y)

        n, m = len(grid), len(grid[0])
        visited = [[False] * m for _ in range(n)]
        res = 0
        for i in range(n):
            for j in range(m):
                if (grid[i][j] == '1' or grid[i][j] == 1) and not visited[i][j]:
                    res += 1
                    dfs(grid, visited, i, j)
        return res


print(Solution_4.nums_is_lands([
    ["1", "1", "1", "1", "0"],
    ["1", "1", "0", "1", "0"],
    ["1", "1", "0", "0", "0"],
    ["0", "0", "0", "0", "0"]
]))

# n, m = map(int, input().split())
# list_1 = []
# for i in range(n):
#     list_1.append(list(map(int, input().split())))
# print(Solution_4.nums_is_lands(list_1))


# 二叉树的中序遍历
class Solution_5:
    @classmethod
    def inorder_traversal(cls, root: Optional[TreeNode]):
        res = []

        def dfs(node):
            if node is not None:
                dfs(node.left)
                res.append(node.val)
                dfs(node.right)
        dfs(root)
        return res


root = TreeNode(1)
root.right = TreeNode(2)
root.right.left = TreeNode(3)
print(Solution_5.inorder_traversal(root))