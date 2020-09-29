class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 0:
            return 0
        dip = prices[0]
        profit = 0
        for i in range(len(prices)):
            if prices[i] < dip:
                dip = prices[i]
            else:
                profit = max(profit, prices[i] - dip)
        return profit