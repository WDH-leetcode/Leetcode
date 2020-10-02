class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) == 0:
            return 0
        map = {}
        st = max_length = 0
        for i in range(len(s)):
            if s[i] in map and st <= map[s[i]]:
                st = i + 1
            else:
                max_length = max(max_length, i - st + 1)
            map[s[i]] = i
        return max_length