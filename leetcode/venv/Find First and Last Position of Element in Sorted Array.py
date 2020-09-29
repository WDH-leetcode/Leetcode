class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        output = [-1, -1]
        if nums == []:
            return output
        l = len(nums)
        st = 0
        ed = l-1
        # find left boundary
        while st < ed:
            mid = st + (ed-st)//2
            if nums[mid] < target:
                st = mid + 1
            else:
                ed = mid
        if nums[st] == target:
            output[0] = st
        # find right boundary
        ed = l-1
        while st < ed:
            mid = st + (ed-st)//2 + 1
            if nums[mid] > target:
                ed = mid -1
            else:
                st = mid
        if nums[ed] == target:
            output[1] = ed
        return output