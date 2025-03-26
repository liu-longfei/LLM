'''
leetcode:1
'''


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