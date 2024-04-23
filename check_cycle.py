class Solution:
    def isCyclic(self, V, adj):
        visited = [False] * V
        on_the_way = [False] * V

        def cyclic(i):
            visited[i] = True
            on_the_way[i] = True

            for j in range(V):
                if adj[i][j] and on_the_way[j]:
                    return True

                if (not adj[i][j]) or visited[j]:
                    continue

                if(cyclic(j)):
                    return True
            
            on_the_way[i] = False

            return False # found no cycle
        
        for i in range(V):
            if not visited[i]:
                result = cyclic(i)
                if result:
                    return True
        
        return False
    
# adj = [
#     [False, False, False, False, False, False, False],
#     [False, False, True, False, False, False, False],
#     [False, False, False, False, True, False, False],
#     [False, True, False, False, False, False, False],
#     [True, False, False, False, False, False, False],
#     [False, False, False, True, False, False, False],
#     [False, False, False, False, False, True, False]
# ]

# adj = [
#     [False, False, False, True, False],
#     [False, False, False, False, False],
#     [False, True, False, False, False],
#     [False, False, True, False, False],
#     [False, False, False, True, False]
# ]

# print(Solution().isCyclic(5, adj))