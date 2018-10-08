import minHeap as hp

mHeap = hp.minHeap()

mHeap.push(3);mHeap.push(13);mHeap.push(2)

print mHeap.top() #prints 2
print mHeap.sz #prints 3

mHeap.pop()

print mHeap.top() #prints 3
