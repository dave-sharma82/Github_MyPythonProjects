"""Generate a static overview PNG and the sample 10 games CSV."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from engine import load_draws, frequency_table, generate_tickets, draw_features, stats_summary, backtest

root=Path(__file__).parent
raw, draws, occ = load_draws(root/'5132-results.csv')
freq=frequency_table(occ)
features=draw_features(draws)
tickets=generate_tickets(occ, seed=20261120)
tickets.to_csv(root/'powerball_10_games.csv',index=False)
fig=plt.figure(figsize=(17,11),facecolor='#0c1527')
gs=fig.add_gridspec(3,2,height_ratios=[.25,1,1.06],hspace=.4,wspace=.2)
ax0=fig.add_subplot(gs[0,:]);ax0.axis('off')
ax0.text(0,.85,'AUSTRALIAN POWERBALL  •  ANALYTICS LAB',color='white',fontsize=26,weight='bold',va='top')
ax0.text(0,.12,'449 comparable 7-from-35 draws  |  19 Apr 2018 – 19 Nov 2026  |  descriptive analysis, not prediction',color='#9db5ce',fontsize=13)
ax1=fig.add_subplot(gs[1,:]);ax1.set_facecolor('#14223a')
counts=freq['Count'].to_numpy();nums=freq['Number'].to_numpy()
colors=['#2dd4bf' if c>=counts.mean() else '#60a5fa' for c in counts]
ax1.bar(nums,counts,color=colors,width=.75)
ax1.axhline(len(draws)/5,ls='--',color='#ffbf69',linewidth=2,label='Uniform expectation (20% of draws)')
ax1.set_title('MAIN NUMBER FREQUENCY  •  HISTORICAL VS EXPECTED',loc='left',color='white',fontsize=15,pad=18)
ax1.set_xticks(nums);ax1.set_xlim(0,36);ax1.legend(facecolor='#14223a',labelcolor='white')
ax1.set_ylabel('Draw appearances',color='white')
ax2=fig.add_subplot(gs[2,0]);ax2.set_facecolor('#14223a')
ax2.hist(features['Sum'],bins=26,color='#38bdf8',edgecolor='#0c1527')
ax2.set_title('SUM OF SEVEN NUMBERS',loc='left',color='white',fontsize=15,pad=15)
ax2.set_xlabel('Draw sum',color='white');ax2.set_ylabel('Draw count',color='white')
ax3=fig.add_subplot(gs[2,1]);ax3.set_facecolor('#14223a')
mat=np.zeros((10,35))
for r,row in tickets.iterrows():
    for j in range(1,8):mat[r,int(row[f'N{j}'])-1]=1
from matplotlib.colors import ListedColormap
ax3.imshow(mat,aspect='auto',cmap=ListedColormap(['#223755','#27dbc2']),interpolation='nearest',vmin=0,vmax=1)
ax3.set_title('10 SAMPLE GAMES  •  NUMBER COVERAGE',loc='left',color='white',fontsize=15,pad=15)
ax3.set_xticks(np.arange(0,35,2),labels=np.arange(1,36,2));ax3.set_yticks(np.arange(10),labels=[f'Game {i}' for i in range(1,11)])
for ax in (ax1,ax2,ax3):
    for spine in ax.spines.values():spine.set_color('#334a6a')
    ax.tick_params(colors='#aec4df')
fig.text(.02,.018,'FAIR DRAW: every 7-number set has the same theoretical chance • 1 in 6,724,520 to match all seven main numbers',color='#c2d2e6',fontsize=12)
fig.subplots_adjust(left=.06,right=.975,top=.97,bottom=.07)
fig.savefig(root/'dashboard_preview.png',dpi=150,facecolor=fig.get_facecolor())
plt.close(fig)
print(tickets.to_string(index=False))
print('top',freq.nlargest(8,'Count')[['Number','Count']].values.tolist())
print('bottom',freq.nsmallest(8,'Count')[['Number','Count']].values.tolist())
