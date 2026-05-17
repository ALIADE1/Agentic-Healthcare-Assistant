class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        single_sum, double_sum = 0, 0
        
        for x in nums:
            if len(str(x)) == 1:
                single_sum+=x
            else:
                double_sum+=x

        return not(double_sum==single_sum)
