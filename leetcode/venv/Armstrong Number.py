class Solution:
    def isArmstrong(self, N: int) -> bool:
        l = len(str(N))
        num = N
        tot = 0
        for i in range(1, l + 1):
            remainder = num // (10 ** (l - i))
            num -= (10 ** (l - i)) * remainder
            tot += remainder ** l
        if N == tot:
            return True
        else:
            return False
