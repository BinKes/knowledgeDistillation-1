# 知识蒸馏

$$
loss = \alpha CE(q, \tilde q) + (1-\alpha) CE(p, q)
$$

p为真实标签，q为学生网络输出（经过softmax），$\tilde q$为老师网络输出（不经过softmax）再经过softmaxT

前部分称为soft loss，后部分称为hard loss

![](http://consolexinhun.test.upcdn.net/20200523181929.png)