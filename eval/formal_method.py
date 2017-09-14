from model.model import Model
from nsga_two import PatientProblem
import inspyred
import random
import numpy as np
from scipy.stats import pearsonr

class EvaluationFramework():

    rmax = 3
    pop_size = 10
    generations = 5
    w1 = 1
    w2 = 0
    max_complexity = 50
    w_descr = 0.25
    w_pred = 0.25
    w_param = 0.25
    w_complex = 0.25

    training_data = {}
    test_data = {}
    model = []
    eval_aspects = []
    data = []
    headers = []
    nd_hypervolumes = []

    def __init__(self):
        self.training_data = {}
        self.test_data = {}
        self.model = []
        self.eval_aspects = []
        self.data = []
        self.headers = []
        self.nd_hypervolumes = []

    def get_best_model_parameters(self, m, training, test, eval_aspects, individual):
        self.training_data = training
        self.test_data = test
        self.model = m
        self.eval_aspects = eval_aspects
        self.generate_nsga_2_data()
        individual_results = self.data[self.data[:,0] == individual]
        i_pred = [i for i, s in enumerate(self.headers) if 'pred_' in s]
        i_param = [i for i, s in enumerate(self.headers) if 'param' in s]
        pred_scores = self.data[self.data[:,0]==individual]
        pred_scores = pred_scores[:,i_param+i_pred]

        # now select the parameters that minimize the sum of fitness over the predictive performance for the different evaluation aspects.

        best_candidate = 0
        best_score = -1
        for i in range(0, pred_scores.shape[0]):
            fitness_sum = sum(map(float, pred_scores[i,len(i_param):].tolist()))
            if best_score == -1 or fitness_sum < best_score:
                best_candidate = i
                best_score = fitness_sum

        return map(float, pred_scores[best_candidate, 0:len(i_param)])

    def evaluate_model_simple(self, m, training, test, eval_aspects):
        self.training_data = training
        self.test_data = test
        self.model = m
        self.eval_aspects = eval_aspects
        self.generate_nsga_2_data()

        nh_b_list_descr = []
        b_len = len(self.training_data)
        for b in range(b_len):
            nh_b_list.append(sum(self.nd_hypervolumes[b*self.rmax:(b+1)*self.rmax])/float(self.rmax))
        descriptive_perf = np.mean(nh_b_list_descr)

        b_len = len(self.test_data)
        nh_b_list_pred = []
        for b in range(b_len):
            nh_b_list.append(sum(self.nd_hypervolumes[b*self.rmax:(b+1)*self.rmax])/float(self.rmax))
        pred_perf = np.mean(nh_b_list_pred)
        return pred_perf

    def evaluate_model(self, m, training, test, eval_aspects, pop_size=3, generations=10):
        self.pop_size = pop_size
        self.generations = generations
        self.training_data = training
        self.test_data = test
        self.model = m
        self.eval_aspects = eval_aspects
        self.generate_nsga_2_data()
        # m.print_model()
        descr = self.evaluate_descriptive_perf()
        # print descr
        pred = self.evaluate_predictive_perf()
        # print pred
        param = self.evaluate_param_sens()
        # print param
        complex = self.evaluate_complexity()
        # print complex
        #print 'descr ' + str(descr)
        #print 'pred ' + str(pred)
        #print 'param ' + str(param)
        #print 'complex ' + str(complex)
        final_score = self.w_descr * descr + self.w_pred * pred + self.w_param * param + self.w_complex * complex
        if np.isnan(final_score):
            final_score = 0
        return final_score

    def evaluate_descriptive_perf(self):
        # Create averages per patient of the ndh vector
        nh_b_list = []
        b_len = len(self.training_data)
        for b in range(b_len):
            nh_b_list.append(sum(self.nd_hypervolumes[b*self.rmax:(b+1)*self.rmax])/float(self.rmax))
        mu = np.mean(nh_b_list)
        sigma = np.std(nh_b_list)
        return mu * (1-sigma)

    def evaluate_predictive_perf(self):
        # Calculate the mean and standard deviation over all errors over the various aspects

        mean_pred_scores = []
        std_pred_scores = []
        corr_scores = []

        for eval in self.eval_aspects:
            i_descr = self.headers.index('descr_' + eval)
            i_pred = self.headers.index('pred_' + eval)
            pred_scores = map(float, self.data[:,i_pred].tolist())

            for b in self.training_data:
                rel_descr_scores = map(float, np.squeeze(self.data[np.where(self.data[:,0]==b),i_descr]).tolist())
                rel_pred_scores = map(float, np.squeeze(self.data[np.where(self.data[:,0]==b),i_pred]).tolist())
                # print '=====', b
                # print rel_descr_scores
                # print rel_pred_scores
                if rel_descr_scores == rel_pred_scores:
                    corr = 1
                elif len(rel_pred_scores) == 0 or (len(set(rel_descr_scores)) == 1 and len(set(rel_pred_scores)) == 1):
                    corr = 0
                else:
                    try:
                        corr, p = pearsonr(rel_descr_scores, rel_pred_scores)
                    except ValueError:
                        p = 0

                if not np.isnan(corr):
                    corr_scores.append(corr)

            mean_pred_scores.append(np.mean(pred_scores))
            std_pred_scores.append(np.std(pred_scores))
        abs_mu = np.mean(mean_pred_scores)
        abs_std = np.mean(std_pred_scores)

        abs_pred_score = (1-abs_mu)*(1-abs_std)

        # print 'abs score: ' + str(abs_pred_score)

        # print 'corr_scores ' + str(corr_scores)
        rel_mu = np.mean(corr_scores)
        rel_std = np.std(corr_scores)

        rel_pred_score = max(rel_mu, 0) * (1-rel_std)
        # print abs_pred_score
        # print rel_pred_score

        # print 'rel score: ' + str(rel_pred_score)

        return self.w1 * abs_pred_score + self.w2 * rel_pred_score

    def evaluate_param_sens(self):
        parameter_scores = []
        for p in self.model.parameter_names:     # Parameters
            max_value = 0
            index_p = self.headers.index(p)
            for j in self.eval_aspects:          # Evaluation criteria
                index_j = self.headers.index('descr_' + j)
                for b in self.training_data:     # Patients
                    param_values = map(float, np.squeeze(self.data[np.where(self.data[:,0]==b),index_p]).tolist())
                    descr_scores = map(float, np.squeeze(self.data[np.where(self.data[:,0]==b),index_j]).tolist())
                    # Apparetly there is one perf
                    if (min(param_values) == max(param_values)):
                        corr_p_j_b = 1
                    else:
                        try:
                            corr_p_j_b, pv = pearsonr(param_values, descr_scores)
                        except ValueErorr:
                            pv = 0

                    if not np.isnan(corr_p_j_b):
                        if abs(corr_p_j_b) > 0.35:
                            max_value = 1
            parameter_scores.append(max_value)
        if len(parameter_scores) > 0:
            return sum(parameter_scores)/float(len(parameter_scores))
        else:
            return 1

    def evaluate_complexity(self):
        number_removed_attributes = 0
        for eq in self.model.state_equations:
            if 'self.substituted_value' in eq:
                number_removed_attributes += 1

        return 1-(((len(self.model.state_values)-number_removed_attributes) + (len(self.model.parameter_names)-number_removed_attributes))/float(self.max_complexity))

    def compute_hypervolume(self, final_arc):
        ref_point = [1] * len(self.eval_aspects)
        pareto_points = []
        min_fitness = [1] * len(self.eval_aspects)

        for s in final_arc:
            pareto_points.append(s.fitness)

            for f in range(len(s.fitness)):
                if min_fitness[f] > s.fitness[f]:
                    min_fitness[f] = s.fitness[f]

        volume = inspyred.ec.analysis.hypervolume(pareto_points, reference_point=ref_point)
        if volume > 1:
            # We need to add a point....
            for min in range(len(min_fitness)):
                point = [1] * len(self.eval_aspects)
                point[min] = min_fitness[min]
                pareto_points.append(point)
            volume = inspyred.ec.analysis.hypervolume(pareto_points, reference_point=ref_point)
        self.nd_hypervolumes.append(volume)


    def generate_nsga_2_data(self):
        # First create a suitable matrix and headers

        self.headers = ["ID", "run"]
        for param in self.model.parameter_names:
            self.headers.append(param)
        for aspect in self.eval_aspects:
            self.headers.append('descr_' + aspect)
        for aspect in self.eval_aspects:
            self.headers.append('pred_' + aspect)
        self.data = np.zeros((0, len(self.headers)))

        # Generate the NSGA2 data.

        for ID in self.training_data:
            for r in range(self.rmax):
                prng = random.Random()
                problem = PatientProblem()
                problem.set_values(self.model, self.training_data[ID], self.test_data[ID], self.eval_aspects)
                ea = inspyred.ec.emo.NSGA2(prng)
                ea.variator = [inspyred.ec.variators.blend_crossover,
                               inspyred.ec.variators.gaussian_mutation]
                ea.terminator = inspyred.ec.terminators.generation_termination
                final_pop = ea.evolve(generator=problem.generator, evaluator=problem.evaluator, pop_size=self.pop_size, maximize=False, bounder=None,max_generations=self.generations)
                final_arc = ea.archive
                # print '*****'
                # print final_pop
                self.compute_hypervolume(final_arc)

                for f in final_arc:
                    row = [ID, r]
                    for pv in f.candidate:
                        row.append(pv)
                    for fv in f.fitness:
                        row.append(fv)
                    pred_fitness = problem.predict([f.candidate])[0]
                    for pfv in pred_fitness:
                        row.append(pfv)
                    self.data = np.vstack((self.data, row))

    def show_graph(self, final_arc):
        import matplotlib.pyplot as plt
        x = []
        y = []
        for f in final_arc:
            x.append(f.fitness[0])
            y.append(f.fitness[1])
        plt.scatter(x, y, color='b')
        #plt.savefig('{0} Example ({1}).pdf'.format(ea.__class__.__name__, problem.__class__.__name__), format='pdf')
        plt.show()