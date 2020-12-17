import json
import pandas as pd
import matplotlib.pyplot as plt



with open('log5.json', encoding='utf-8-sig') as f:
  data = json.load(f)

#print((data))

episode = pd.DataFrame(data['episode'])

loss = pd.DataFrame(data['loss'])
mean_absolute_error = pd.DataFrame(data['mean_absolute_error'])

plt.plot(episode,mean_absolute_error,label='mean_absolute_error')

#plt.title('Training loss')
plt.title('mean_absolute_error')
plt.xlabel('episode')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.savefig('mean_absolute_error.png',dpi=300, bbox_inches='tight')
plt.show()
#####################################################################################

loss = pd.DataFrame(data['loss'])

plt.figure(figsize=[15,10])
rolling = loss.rolling(window=100)
rolling_mean = rolling.mean()
plt.plot(loss,label='Loss')
plt.plot(rolling_mean,color='red',linewidth=5.0,label='Moving Avg Loss')

plt.title('loss')
plt.legend()
plt.savefig('loss.png',dpi=300, bbox_inches='tight')
plt.show()
######################################################################################

mean_q = pd.DataFrame(data['mean_q'])
plt.plot(episode,mean_q,label='mean_q')

plt.title('mean_q')
plt.legend()
plt.savefig('mean_q.png',dpi=300, bbox_inches='tight')
plt.show()
#####################################################################################

mean_eps = pd.DataFrame(data['mean_eps'])
plt.plot(episode,mean_eps)

plt.title('mean_eps')
plt.legend()
plt.savefig('mean_eps.png',dpi=300, bbox_inches='tight')
plt.show()
#######################################################################################
Final_Y = pd.DataFrame(data['Final_Y'])
plt.plot(episode,Final_Y)


plt.title('Final_Y')
plt.legend()
plt.savefig('Final_Y.png',dpi=300, bbox_inches='tight')
plt.show()
########################################################################################
Final_X = pd.DataFrame(data['Final_X'])
plt.plot(episode,Final_X)

plt.title('Final_X')
plt.legend()
plt.savefig('Final_X.png',dpi=300, bbox_inches='tight')
plt.show()
########################################################################################

plt.plot(Final_X,Final_Y, 'k.')
plt.title('Final_X,Final_Y')
plt.legend()
plt.savefig('Final_xy.png',dpi=300, bbox_inches='tight')
plt.show()
#######################################################################################


duration = pd.DataFrame(data['duration'])
plt.plot(episode,duration)

plt.title('duration')
plt.legend()
plt.savefig('duration.png',dpi=300, bbox_inches='tight')
plt.show()
########################################################################################
episode_reward = pd.DataFrame(data['episode_reward'])
plt.figure(figsize=[15,10])
rolling = episode_reward.rolling(window=100)
rolling_mean = rolling.mean()

plt.plot(episode,episode_reward,label='Reward')
plt.plot(episode,rolling_mean,color='red',linewidth=5.0,label='Moving Avg Reward')

plt.title('episode_reward')
plt.legend()
plt.savefig('episode_reward.png',dpi=300, bbox_inches='tight')
plt.show()
########################################################################################
nb_episode_steps = pd.DataFrame(data['nb_episode_steps'])
plt.plot(episode,nb_episode_steps)

plt.title('nb_episode_steps')
plt.legend()
plt.savefig('nb_episode_steps.png',dpi=300, bbox_inches='tight')
plt.show()






