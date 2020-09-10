class Solution:
    def highFive(self, items: List[List[int]]) -> List[List[int]]:
        items = sorted(sorted(items, reverse=True), reverse=True)
        n = items[0][0]
        ct = 0
        tot = 0
        lst = list()
        for i in items:
            if i[0] == n:
                tot += i[1]
                ct += 1
            if ct >= 5:
                ave = tot/5//1
                lst.append([n,int(ave)])
                n -= 1
                ct = 0
                tot = 0
        return reversed(lst)