from mrjob.job import MRJob
from mrjob.step import MRStep

class MRTF(MRJob):

    def configure_args(self):
        super(MRTF, self).configure_args()
        self.add_passthru_arg('--total-docs', type=int)

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_words,
                   reducer=self.reducer_count_words)
        ]

    def mapper_get_words(self, _, line):
        try:
            doc_id, text = line.split("\t", 1)
        except:
            return
        words = text.split()
        total_words = len(words)
        for word in words:
            yield (word, doc_id), 1
        yield ("__total_words__", doc_id), total_words

    def reducer_count_words(self, key, values):
        yield key, sum(values)

if __name__ == '__main__':
    MRTF.run()
