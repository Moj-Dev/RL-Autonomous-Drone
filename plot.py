import json
import pandas as pd
import matplotlib.pyplot as plt



with open('log3.json', encoding='utf-8-sig') as f:
  data3 = json.load(f)
with open('log4.json', encoding='utf-8-sig') as f:
  data4 = json.load(f)
with open('log5.json', encoding='utf-8-sig') as f:
  data5 = json.load(f)
with open('log6.json', encoding='utf-8-sig') as f:
  data6 = json.load(f)
with open('log7.json', encoding='utf-8-sig') as f:
  data7 = json.load(f)
with open('log9.json', encoding='utf-8-sig') as f:
  data9 = json.load(f)
#print((data))

episode3 = pd.DataFrame(data3['episode'])
episode4 = pd.DataFrame(data4['episode'])
episode5 = pd.DataFrame(data5['episode'])
episode6 = pd.DataFrame(data6['episode'])
episode7 = pd.DataFrame(data7['episode'])
episode9 = pd.DataFrame(data9['episode'])

#mean_absolute_error
mean_absolute_error3 = pd.DataFrame(data3['mean_absolute_error'])
mean_absolute_error4 = pd.DataFrame(data4['mean_absolute_error'])
mean_absolute_error5 = pd.DataFrame(data5['mean_absolute_error'])
mean_absolute_error6 = pd.DataFrame(data6['mean_absolute_error'])
mean_absolute_error7 = pd.DataFrame(data7['mean_absolute_error'])
mean_absolute_error9 = pd.DataFrame(data9['mean_absolute_error'])
plt.figure(figsize=[15,10])
plt.plot(episode3,mean_absolute_error3,label='DoubleDuelingDQN Depth Input with Heading')
plt.plot(episode4,mean_absolute_error4,label='DoubleDuelingDQN Depth Input without Heading')
plt.plot(episode5,mean_absolute_error5,label='DoubleDQN Depth Input with Heading ')
plt.plot(episode6,mean_absolute_error6,label='DQN Depth Input with Heading')
plt.plot(episode7,mean_absolute_error7,label='Grey Input DoubleDuelingDQN with Heading ')
plt.plot(episode9,mean_absolute_error9,label='DoubleDQN Depth Input with Heading and Cross-track distance reward ')


#plt.title('Training loss')
plt.title('mean_absolute_error')
plt.xlabel('episode')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.savefig('mean_absolute_error.png',dpi=300, bbox_inches='tight')
plt.show()
#####################################################################################

loss3 = pd.DataFrame(data3['loss'])
loss4 = pd.DataFrame(data4['loss'])
loss5 = pd.DataFrame(data5['loss'])
loss6 = pd.DataFrame(data6['loss'])
loss7 = pd.DataFrame(data7['loss'])
loss9 = pd.DataFrame(data9['loss'])

plt.figure(figsize=[15,10])
rolling3 = loss3.rolling(window=100)
rolling_mean3 = rolling3.mean()

rolling4 = loss4.rolling(window=100)
rolling_mean4 = rolling4.mean()

rolling5 = loss5.rolling(window=100)
rolling_mean5 = rolling5.mean()

rolling6 = loss6.rolling(window=100)
rolling_mean6 = rolling6.mean()

rolling7 = loss7.rolling(window=100)
rolling_mean7 = rolling7.mean()

rolling9 = loss9.rolling(window=100)
rolling_mean9 = rolling9.mean()

#plt.plot(loss3,label='DoubleDuelingDQN Depth Input with Heading')
plt.plot(rolling_mean3,label='DoubleDuelingDQN Depth Input with Heading')
plt.plot(rolling_mean4,label='DoubleDuelingDQN Depth Input without Heading')
plt.plot(rolling_mean5,label='DoubleDQN Depth Input with Heading ')
plt.plot(rolling_mean6,label='DQN Depth Input with Heading ')
plt.plot(rolling_mean7,label='Grey Input DoubleDuelingDQN with Heading ')
plt.plot(rolling_mean9,label='DoubleDQN Depth Input with Heading and Cross-track distance reward')

plt.title('loss')
plt.legend()
plt.savefig('loss.png',dpi=300, bbox_inches='tight')
plt.show()
######################################################################################
mean_q3 = pd.DataFrame(data3['mean_q'])
mean_q4 = pd.DataFrame(data4['mean_q'])
mean_q5 = pd.DataFrame(data5['mean_q'])
mean_q6 = pd.DataFrame(data6['mean_q'])
mean_q7 = pd.DataFrame(data7['mean_q'])
mean_q9 = pd.DataFrame(data9['mean_q'])
plt.figure(figsize=[15,10])
plt.plot(episode3,mean_q3,label='DoubleDuelingDQN Depth Input with Heading')
plt.plot(episode4,mean_q4,label='DoubleDuelingDQN Depth Input without Heading')
plt.plot(episode5,mean_q5,label='DoubleDQN Depth Input with Heading ')
plt.plot(episode6,mean_q6,label='DQN Depth Input with Heading ')
plt.plot(episode7,mean_q7,label='Grey Input DoubleDuelingDQN with Heading ')
plt.plot(episode9,mean_q9,label='DoubleDQN Depth Input with Heading and Cross-track distance reward')


plt.title('mean_q')
plt.legend()
plt.savefig('mean_q.png',dpi=300, bbox_inches='tight')
plt.show()


#####################################################################################

mean_eps3 = pd.DataFrame(data3['mean_eps'])
mean_eps4 = pd.DataFrame(data4['mean_eps'])
mean_eps5 = pd.DataFrame(data5['mean_eps'])
mean_eps6 = pd.DataFrame(data6['mean_eps'])
mean_eps7 = pd.DataFrame(data7['mean_eps'])
mean_eps9 = pd.DataFrame(data9['mean_eps'])

plt.figure(figsize=[15,10])
plt.plot(episode3,mean_eps3,label='DoubleDuelingDQN Depth Input with Heading')
plt.plot(episode4,mean_eps4,label='DoubleDuelingDQN Depth Input without Heading')
plt.plot(episode5,mean_eps5,label='DoubleDQN Depth Input with Heading ')
plt.plot(episode6,mean_eps6,label='DQN Depth Input with Heading ')
plt.plot(episode7,mean_eps7,label='Grey Input DoubleDuelingDQN with Heading ')
plt.plot(episode9,mean_eps9,label='DoubleDQN Depth Input with Heading and Cross-track distance reward')

plt.title('mean_eps')
plt.legend()
plt.savefig('mean_eps.png',dpi=300, bbox_inches='tight')
plt.show()

#######################################################################################
########################################################################################

#######################################################################################


duration3 = pd.DataFrame(data3['duration'])
rolling3 = duration3.rolling(window=45)
rolling_mean3 = rolling3.mean()

duration4 = pd.DataFrame(data4['duration'])
rolling4 = duration4.rolling(window=45)
rolling_mean4 = rolling4.mean()

duration5 = pd.DataFrame(data5['duration'])
rolling5 = duration5.rolling(window=45)
rolling_mean5 = rolling5.mean()

duration6 = pd.DataFrame(data6['duration'])
rolling6 = duration6.rolling(window=45)
rolling_mean6 = rolling6.mean()

duration7 = pd.DataFrame(data7['duration'])
rolling7 = duration7.rolling(window=45)
rolling_mean7 = rolling7.mean()

duration9 = pd.DataFrame(data9['duration'])
rolling9 = duration9.rolling(window=45)
rolling_mean9 = rolling9.mean()

plt.figure(figsize=[15,10])
#plt.plot(episode3,duration3)
plt.plot(episode3,rolling_mean3,label= 'DoubleDuelingDQN Depth Input with Heading')
plt.plot(episode4,rolling_mean4,label= 'DoubleDuelingDQN Depth Input without Heading')
plt.plot(episode5,rolling_mean5,label= 'DoubleDQN Depth Input with Heading ')
plt.plot(episode6,rolling_mean6,label= 'DQN Depth Input with Heading ')
plt.plot(episode7,rolling_mean7,label= 'Grey Input DoubleDuelingDQN with Heading ')
plt.plot(episode9,rolling_mean9,label= 'DoubleDQN Depth Input with Heading and Cross-track distance reward')

plt.title('duration moving average')
plt.legend()
plt.savefig('durationmoving average.png',dpi=300, bbox_inches='tight')
plt.show()


fig = plt.figure(figsize=[15,10])
fig.subplots_adjust(left=0.2, wspace=0.6)
ax1 = fig.add_subplot(221)
ax1.plot(episode3,duration3,label= 'DoubleDuelingDQN Depth Input with Heading')
ax1.plot(episode3,rolling_mean3)
ax1.legend()

ax3 = fig.add_subplot(223)
ax3.plot(episode4,duration4,label= 'DoubleDuelingDQN Depth Input without Heading')
ax3.plot(episode4,rolling_mean4)
ax3.legend()

ax2 = fig.add_subplot(222)
ax2.plot(episode5,duration5,label= 'DoubleDQN Depth Input with Heading ')
ax2.plot(episode5,rolling_mean5)
ax2.legend()

ax4 = fig.add_subplot(224)
ax4.plot(episode6,duration6,label= 'DQN Depth Input with Heading ')
ax4.plot(episode6,rolling_mean6)


plt.legend()
plt.legend()
plt.savefig('duration.png',dpi=300, bbox_inches='tight')
plt.show()


########################################################################################
episode_reward3 = pd.DataFrame(data3['episode_reward'])
episode_reward4 = pd.DataFrame(data4['episode_reward'])
episode_reward5 = pd.DataFrame(data5['episode_reward'])
episode_reward6 = pd.DataFrame(data6['episode_reward'])
episode_reward7 = pd.DataFrame(data7['episode_reward'])
episode_reward9 = pd.DataFrame(data9['episode_reward'])

plt.figure(figsize=[15,10])
rolling3 = episode_reward3.rolling(window=100)
rolling_mean3 = rolling3.mean()
rolling4 = episode_reward4.rolling(window=100)
rolling_mean4 = rolling4.mean()
rolling5 = episode_reward5.rolling(window=100)
rolling_mean5 = rolling5.mean()
rolling6 = episode_reward6.rolling(window=100)
rolling_mean6 = rolling6.mean()
rolling7 = episode_reward7.rolling(window=100)
rolling_mean7 = rolling7.mean()
rolling9 = episode_reward9.rolling(window=100)
rolling_mean9 = rolling9.mean()

plt.plot(episode3,rolling_mean3,label='DoubleDuelingDQN Depth Input with Heading')
plt.plot(episode4,rolling_mean4,label='DoubleDuelingDQN Depth Input without Heading')
plt.plot(episode5,rolling_mean5,label='DoubleDQN Depth Input with Heading ')
plt.plot(episode6,rolling_mean6,label='DQN Depth Input with Heading ')
plt.plot(episode7,rolling_mean7,label='Grey Input DoubleDuelingDQN with Heading')
plt.plot(episode9,rolling_mean9,label='DoubleDQN Depth Input with Heading and Cross-track distance reward')

plt.title('episode_reward')
plt.legend()
plt.savefig('episode_reward.png',dpi=300, bbox_inches='tight')
plt.show()
########################################################################################




