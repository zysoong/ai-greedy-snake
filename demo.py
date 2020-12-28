from greedysnake import Direction
f = open('step.input', 'r')
lines = f.readlines()
a = [None]*len(lines)
for i in range(len(a)):
    exec('a[' + str(i) + '] = ' + str(lines[i]))
print(a[0])