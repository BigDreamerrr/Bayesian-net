import math
from typing import List

class Solution:
    def generate_subset(max_size, elem_cnt):
        sets = [()]
        l = 0
        while True:
            yield sets

            if l == max_size:
                break # No need to generate more!
            new_sets = [None] * math.comb(elem_cnt, (l + 1))
            index = 0
            for subset in sets:
                start = (subset[-1] if subset != () else -1) + 1
                for new_elem in range(start, elem_cnt):
                    new_sets[index] = (*subset, new_elem)
                    index += 1
            sets = new_sets
            l += 1

    def subsets(self, nums: List[int]) -> List[List[int]]:
        generator = Solution.generate_subset(len(nums) + 1, len(nums))

        answer = []
        for subsets in generator:
            for subset in subsets:
                answer.append([nums[index] for index in subset])

        return answer
    
print(Solution().subsets([0]))