class minHeap():
    '''Implementation of Binary min heap.
    '''
    def __init__(self):
        '''Contructor to min heap.
        '''
        self.min_heap = []
        self.sz = 0

    def push(self, a):
        '''Insert an element in binary min heap.
        example: push(3)
        '''
        self.sz = self.sz + 1
        self.min_heap = self.min_heap + [a]
        idx = self.sz - 1
        while idx:
            p = idx >> 1
            if self.min_heap[idx] >= self.min_heap[p]:
                break
            else:
                self.min_heap[idx], self.min_heap[p] = self.min_heap[p], self.min_heap[idx]
                idx = p

    def top(self):
        '''Returns the minimum element or the element at top of heap.
        If the heap has no elements, then returns 999999999
        '''
        if self.sz >= 1:
            return self.min_heap[0]
        else:
            return 999999999

    def pop(self):
        '''Removes the top element or smallest element from the heap memory.
        '''
        if self.sz == 0:
            return
        self.min_heap[self.sz - 1], self.min_heap[0] = self.min_heap[0], self.min_heap[self.sz - 1]
        self.sz = self.sz - 1
        self.min_heap.pop()
        idx = 0
        while idx < self.sz - 1:
            left = idx << 1 | 1
            right = left + 1
            if left > self.sz - 1:
                break
            if right == self.sz:#There is just the left child
                if self.min_heap[left] >= self.min_heap[idx]:
                    break
                else:
                    self.min_heap[left], self.min_heap[idx] = self.min_heap[idx], self.min_heap[left]
                    idx = left
            else:
                if self.min_heap[left] < self.min_heap[right] and self.min_heap[idx] > self.min_heap[left]:
                    self.min_heap[left], self.min_heap[idx] = self.min_heap[idx], self.min_heap[left]
                    idx = left
                elif self.min_heap[right] <= self.min_heap[left] and self.min_heap[idx] > self.min_heap[right]:
                    self.min_heap[right], self.min_heap[idx] = self.min_heap[idx], self.min_heap[right]
                    idx = right
                else:
                    break
