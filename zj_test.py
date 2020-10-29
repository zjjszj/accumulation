

# 1 土地之争
# n, p, q=input().split()
# n=int(n)
# p=int(p)
# q=int(q)
# a=input().split(" ")
# b=input().split(" ")

# a=[1,2,3]
# b=[3,4,5]

# sorted(a)
# sorted(b)
# same, only_a, only_b=0,0, 0
# for i in range(p):
#     for j in range(q):
#         if a[i]>b[j]:
#             continue
#         elif a[i]<b[j]:
#             break
#         else:
#             same=same+1
# if len(same)==0:
#     print(p, " ", q, " ", 0)
# else:
#     print(p-len(same), " ", q-len(same)," ", len(same))

# c=a+b
# same=c-n
# print(p-same, " ", q-same, " ", same)

# 2 修改大小写字母
# s="AAAb"
# l, b=0, 0
# for i in range(len(s)):
#     if(s[i].islower()):
#         l=l+1
#     else:
#         b=b+1
# print(int(abs(l-b)/2))
# 3 运算
# l=input()
# l=int(l)
# a=input().split(" ")
# a=[int(a[i]) for i in range(l)]
# # l=2
# # a=[3,2]
# b=[]
# if l==0:
#     print(' ')
# else:
#     for i in range(1, l+1):
#         j=1
#         r=i%j
#         while(j<l):
#             j=j+1
#             m=i%j
#             r=r^m
#         r=r^a[i-1]
#         b.append(r)
#     # 处理b
#     r=b[0]
#     for i in range(1, l):
#         r=r^b[i]
#     print(r)

# 4 字节：任务调度
"""
思路：维护两个列表： 1 计算每个任务实际提交时间（开始运行时间） 2 计算每个线程的完成时间
"""
threads, tasks = input().split()
threads, tasks = int(threads), int(tasks)
data = []
for i in range(tasks):
    data.append(list(map(int, input().split())))
real_start, finish, result = [], [], []
# 线程都未分配时，分配每个线程
for i in range(threads):
    t_s, t_f = data[i][0], data[i][1]
    real_start.append(t_s)
    finish.append(t_f + t_s)
    result.append(i + 1)
# 为其余任务分配线程
for i in range(threads, len(data)):
    t_s, t_f = data[i][0], data[i][1]

    # 计算每个任务实际提交时间（开始运行时间）
    # 最小值及其索引
    min_finish = finish[0]
    index = 0
    for j in range(threads):
        if min_finish > finish[j]:
            min_finish = finish[j]
            index = j
    real_start.append(t_s)
    if t_s < min_finish:
        real_start[i]=min_finish
    # 为任务分配线程
    min_f = real_start[i]
    index2 = index
    for j in range(threads):
        if finish[j]<= min_f:
            index2 = j
            break
    result.append(index2+1)

    # 更新每个线程的完成时间
    finish[index2] = real_start[i] + t_f

print(result)