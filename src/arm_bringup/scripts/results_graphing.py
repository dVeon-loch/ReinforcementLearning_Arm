from pygrok import Grok
import os
import matplotlib

matplotlib.use("WebAgg")

from matplotlib import pyplot as plt
import numpy as np 
from scipy.signal import savgol_filter
os.chdir('../../..')

SAClogfile = open('results/SACslow/20201111T005349.830047_SAC_/20201111T005349.907611.log','r')
pattern = '.*Total Epi\:%{SPACE}%{BASE10NUM:episode_num} .*%{BASE10NUM:total_steps}.*Return\:%{SPACE}%{BASE10NUM:episode_return}.*'

grok = Grok(pattern)

loglist = SAClogfile.readlines()
SAClogfile.close()
matchlist = []
for i in loglist:
    matchlist.append(grok.match(i))
del loglist
# print(matchlist)
SACreturnarr = []
for i in matchlist:
    if(i==None):
        pass
    else:
        reward = i['episode_return']
        SACreturnarr.append(float(reward))
del matchlist
# TD3logfile = open('results/TD3_staticslow/20201110T191348.801774_TD3_/20201110T191348.868144.log','r')
# pattern = '.*Total Epi\:%{SPACE}%{BASE10NUM:episode_num} .*%{BASE10NUM:total_steps}.*Return\:%{SPACE}%{BASE10NUM:episode_return}.*'

# grok = Grok(pattern)

# loglist = TD3logfile.readlines()
# TD3logfile.close()
# matchlist = []
# for i in loglist:
#     matchlist.append(grok.match(i))
# del loglist
# # print(matchlist)
# TD3returnarr = []
# for i in matchlist:
#     if(i==None):
#         pass
#     else:
#         reward = i['episode_return']
#         TD3returnarr.append(float(reward))
# del matchlist
# PPOlogfile = open('results/PPO_static_slow/20201110T205822.677756_PPO_/20201110T205822.743626.log','r')
# pattern = '.*Total Epi\:%{SPACE}%{BASE10NUM:episode_num} .*%{BASE10NUM:total_steps}.*Return\:%{SPACE}%{BASE10NUM:episode_return}.*'

# grok = Grok(pattern)

# loglist = PPOlogfile.readlines()
# PPOlogfile.close()
# matchlist = []
# for i in loglist:
#     matchlist.append(grok.match(i))
# del loglist
# # print(matchlist)
# PPOreturnarr = []
# for i in matchlist:
#     if(i==None):
#         pass
#     else:
#         reward = i['episode_return']
#         PPOreturnarr.append(float(reward))
# del matchlist








# x = np.arange(max(len(SACreturnarr),len(TD3returnarr),len(PPOreturnarr)))
x = np.arange(len(SACreturnarr))


smoothedSAC = savgol_filter(SACreturnarr,101, 2)
# smoothedTD3 = savgol_filter(TD3returnarr,101, 2)
# smoothedPPO = savgol_filter(PPOreturnarr,101, 2)

SACxlim = len(smoothedSAC)
# TD3xlim = len(smoothedTD3)
# PPOxlim = len(smoothedPPO)


# difPPO_SAC = len(PPOreturnarr)-len(SACreturnarr)
# for i in range(difPPO_SAC):
#     SACreturnarr.append(None)
#     smoothedSAC = np.append(smoothedSAC,None)

# difPPO_TD3 = len(PPOreturnarr)-len(TD3returnarr)
# for i in range(difPPO_TD3):
#     TD3returnarr.append(None)
#     smoothedTD3 = np.append(smoothedTD3,None)





plt.figure()
plt.title('Moving Goal Training Return')

plt.plot(x,SACreturnarr,'r',alpha=0.5)
plt.plot(x,smoothedSAC, 'r' ,label="SAC")
plt.xlim([0,SACxlim])
# plt.plot(x,TD3returnarr,'g',alpha=0.5)
# plt.plot(x,smoothedTD3,'g',label='TD3')
# plt.xlim([0,TD3xlim])
# plt.plot(x,PPOreturnarr,'b',alpha=0.5)
# plt.plot(x,smoothedPPO,'b',label='PPO')
# plt.xlim([0,PPOxlim])
plt.legend()
plt.xlabel('Episodes') 
# naming the y axis 
plt.ylabel('Return') 

plt.show()
