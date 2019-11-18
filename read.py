h=open('./output/topics.txt', 'r',encoding='utf-8')
read_data = h.read()
a = read_data.split()
print('#of topic', len(a))
for i in a:
    print(i)