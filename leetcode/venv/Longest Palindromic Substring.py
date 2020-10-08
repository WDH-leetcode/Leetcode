class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) == 0 or len(s) == 1:
            return s
        ans = []
        for i in range(len(s)):
            self.twocenters(i, ans, s)
            self.center(i, ans, s)
        l = {}
        for i in range(len(ans)):
            l[len(ans[i])] = i
        return ans[l[max(l)]]

    def twocenters(self, tc, ans, s, first=0):
        i = tc
        j = tc + 1
        if i - first >= 0 and j + first < len(s) and s[i - first] == s[j + first]:
            self.twocenters(tc, ans, s, first + 1)
        else:
            ans.append(s[tc - first + 1: tc + first + 1])

    def center(self, c, ans, s, first=1):
        if c - first >= 0 and c + first < len(s) and s[c - first] == s[c + first]:
            self.center(c, ans, s, first + 1)
        else:
            ans.append(s[c - first + 1: c + first])
# not the best solution
# Maybe use DP is not as good as using other methods
