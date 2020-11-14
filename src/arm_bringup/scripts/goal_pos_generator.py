import numpy as np

poslist = []
for i in range(1,10):
    while(True):
                    x_y = np.random.uniform(low = -0.4, high = 0.4, size = 2)
                    z = np.random.uniform(low = 0, high = 0.4, size = 1)
                    goal_pos = np.concatenate([x_y,z],axis=0) #np.array([-0.13242582 , 0.29086919 , 0.20275278])
                    if(np.linalg.norm(goal_pos)<0.4 and np.linalg.norm(goal_pos)>0.1):
                        break
    poslist.append(goal_pos)

print(poslist)