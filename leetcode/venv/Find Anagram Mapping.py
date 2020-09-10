class Solution:
    def anagramMappings(self, A: List[int], B: List[int]) -> List[int]:
        lst = list()
        for n in A:
            i = B.index(n)
            lst.append(i)
        return lst