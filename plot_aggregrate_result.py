import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from util.util import Util
import scipy.stats

# Plot the GP information

runs = 10

columns = []
best_columns = []
for i in range(1,runs+1):
    columns.append('mean_run_' + str(i))
    best_columns.append('best_run_' + str(i))
columns.extend(best_columns)


full_frame = pd.DataFrame()

for i in range(1, runs+1):

    directory = '../data/output_runs/pop_100_gen_50_run_' + str(i) + '/'
    df=pd.read_csv(directory + 'gp.csv', sep=',',header=None)
    if i == 1:
        full_frame = pd.DataFrame(0, index=df.ix[:,0], columns=columns)
    full_frame.ix[:,i-1] = df.ix[:,2]
    full_frame.ix[:,runs + (i-1)] = df.ix[:,1]

full_frame['average_mean_fitness'] = full_frame.ix[:,0:runs].median(axis=1)
full_frame['lower_iqr_mean_fitness'] = full_frame['average_mean_fitness'] - full_frame.ix[:,0:runs].quantile(0.25, axis=1)
full_frame['upper_iqr_mean_fitness'] = full_frame.ix[:,0:runs].quantile(0.75, axis=1) - full_frame['average_mean_fitness']
full_frame['average_best_fitness'] = full_frame.ix[:,runs:2*runs].median(axis=1)
full_frame['lower_iqr_best_fitness'] = full_frame['average_best_fitness'] - full_frame.ix[:,runs:2*runs].quantile(0.25, axis=1)
full_frame['upper_iqr_best_fitness'] = full_frame.ix[:,runs:2*runs].quantile(0.75, axis=1) - full_frame['average_best_fitness']

plt.figure('Evolutionary run')
plt.hold(True)
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.errorbar(full_frame.index.values[::2], full_frame['average_mean_fitness'].iloc[::2], linestyle='-', color='r', yerr=[full_frame['lower_iqr_mean_fitness'].iloc[::2], full_frame['upper_iqr_mean_fitness'].iloc[::2]])
plt.errorbar(full_frame.index.values[1::2], full_frame['average_best_fitness'].iloc[1::2], linestyle='-', color='b', yerr=[full_frame['lower_iqr_best_fitness'].iloc[1::2], full_frame['upper_iqr_best_fitness'].iloc[1::2]])
plt.legend(['Median average fitness (even generations) with 25% and 75% IQR', 'Median best fitness (odd generations) with 25% and 75% IQR'], fontsize=8, loc=4)
plt.savefig('../../paper/figs/fitness_overview_pop_100_gen_50.png', bbox_inches='tight')
plt.hold(False)
plt.show()
plt.close()

# Plot the performance comparison information...

algs = ['gp', 'lit', 'lstm', 'lit_generic']
prediction_time = 3
eval_aspects = ['self.mood', 'self.sleep']


directory = '../data/output_runs/results_comparison/'
in_sample=pd.read_csv(directory + 'results_in_sample.csv', sep=',', index_col=0)
out_sample=pd.read_csv(directory + 'results_out_of_sample.csv', sep=',', index_col=0)

lstm_directory = '../data/output_runs/results_comparison_generic_lstm/'
lstm_in_sample=pd.read_csv(lstm_directory + 'results_in_sample.csv', sep=',', index_col=0)
lstm_out_sample=pd.read_csv(lstm_directory + 'results_out_of_sample.csv', sep=',', index_col=0)
in_sample = pd.concat([in_sample, lstm_in_sample], axis=1)
out_sample = pd.concat([out_sample, lstm_out_sample], axis=1)

print 'Highest standard deviation case: ', out_sample.std(axis=1).idxmax()

for alg in algs:
    print '\multirow{3}{*}{' + alg + '} ',
    for t in range(0, prediction_time):
        print ' & ' + str(t+1) + ' & ',
        for eval in eval_aspects:
            col = alg + '_' + eval + '_(t+' + str(t+1) + ')'
            print '%.3f' % in_sample[col].median(),
            print ' & ',
        for eval in eval_aspects:
            col = alg + '_' + eval + '_(t+' + str(t+1) + ')'
            if eval_aspects.index(eval) == (len(eval_aspects)-1):
                print '%.3f' % out_sample[col].median(),
                print '\\\\\cline{2-6}'
            else:
                print '%.3f' % out_sample[col].median(),
                print ' & ',

# Apply the Wilcoxon test

print 'Wilcoxon test....'
for eval in eval_aspects:
    print 'Evaluation metric: ' + eval
    for t in range(0, prediction_time):
        print 'Time point: ' + str(t+1)
        col = 'gp_' + eval + '_(t+' + str(t+1) + ')'
        for alg2 in algs:
            if not 'gp' == alg2:
                col2 = alg2 + '_' + eval + '_(t+' + str(t+1) + ')'
                print '- In sample gp vs ' + alg2,
                stat, p = scipy.stats.ranksums(in_sample[col], in_sample[col2])
                print '%.4f' % p
                print '- Out of sample gp vs ' + alg2,
                stat, p = scipy.stats.ranksums(out_sample[col], out_sample[col2])
                print '%.4f' % p

plt.figure('Performance mood(t+1)')
plt.hold(True)
plt.xlabel('Patient rank')
plt.ylabel('RMSE')
plt.plot(range(1,len(in_sample.index.values)+1), (np.sort(in_sample['gp_self.mood_(t+1)'])[::-1]), 'r-', linewidth=1.5)
plt.plot(range(1,len(in_sample.index.values)+1), (np.sort(out_sample['gp_self.mood_(t+1)'])[::-1]), 'r:', linewidth=1.5)
plt.plot(range(1,len(in_sample.index.values)+1), (np.sort(in_sample['lit_self.mood_(t+1)'])[::-1]), 'b-', linewidth=1.5)
plt.plot(range(1,len(in_sample.index.values)+1), (np.sort(out_sample['lit_self.mood_(t+1)'])[::-1]), 'b:', linewidth=1.5)
plt.plot(range(1,len(in_sample.index.values)+1), (np.sort(in_sample['lstm_self.mood_(t+1)'])[::-1]), 'k-', linewidth=1.5)
plt.plot(range(1,len(in_sample.index.values)+1), (np.sort(out_sample['lstm_self.mood_(t+1)'])[::-1]), 'k:', linewidth=1.5)
plt.plot(range(1,len(in_sample.index.values)+1), (np.sort(in_sample['lit_generic_self.mood_(t+1)'])[::-1]), 'g-', linewidth=1.5)
plt.plot(range(1,len(in_sample.index.values)+1), (np.sort(out_sample['lit_generic_self.mood_(t+1)'])[::-1]), 'g:', linewidth=1.5)
plt.legend(['GP in sample', 'GP out of sample', 'Literature in sample', 'Literature out of sample', 'LSTM individual in sample', 'LSTM individual out of sample', 'LSTM generic in sample', 'LSTM generic out of sample'], fontsize=8)
plt.savefig('../../paper/figs/performance_mood_t+1.png', bbox_inches='tight')
plt.hold(False)
plt.show()
plt.close()

plt.figure('Performance mood(t+3)')
plt.hold(True)
plt.xlabel('Patient rank')
plt.ylabel('RMSE')
plt.plot(range(1,len(in_sample.index.values)+1), (np.sort(in_sample['gp_self.mood_(t+3)'])[::-1]), 'r-', linewidth=1.5)
plt.plot(range(1,len(in_sample.index.values)+1), (np.sort(out_sample['gp_self.mood_(t+3)'])[::-1]), 'r:', linewidth=1.5)
plt.plot(range(1,len(in_sample.index.values)+1), (np.sort(in_sample['lit_self.mood_(t+3)'])[::-1]), 'b-', linewidth=1.5)
plt.plot(range(1,len(in_sample.index.values)+1), (np.sort(out_sample['lit_self.mood_(t+3)'])[::-1]), 'b:', linewidth=1.5)
plt.plot(range(1,len(in_sample.index.values)+1), (np.sort(in_sample['lstm_self.mood_(t+3)'])[::-1]), 'k-', linewidth=1.5)
plt.plot(range(1,len(in_sample.index.values)+1), (np.sort(out_sample['lstm_self.mood_(t+3)'])[::-1]), 'k:', linewidth=1.5)
plt.plot(range(1,len(in_sample.index.values)+1), (np.sort(in_sample['lit_generic_self.mood_(t+3)'])[::-1]), 'g-', linewidth=1.5)
plt.plot(range(1,len(in_sample.index.values)+1), (np.sort(out_sample['lit_generic_self.mood_(t+3)'])[::-1]), 'g:', linewidth=1.5)
plt.legend(['GP in sample', 'GP out of sample', 'Literature in sample', 'Literature out of sample', 'LSTM individual in sample', 'LSTM individual out of sample', 'LSTM generic in sample', 'LSTM generic out of sample'], fontsize=8)
plt.savefig('../../paper/figs/performance_mood_t+3.png', bbox_inches='tight')
plt.hold(False)
plt.show()
plt.close()

