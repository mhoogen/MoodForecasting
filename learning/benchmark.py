class Benchmark():

    def test_benchmark(self, test_output):
        prediction = [[0]]
        for i in range(len(test_output)-1):
            prediction.append(test_output[i])
        return prediction