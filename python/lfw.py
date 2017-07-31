
import sys
import numpy as np

class LFW:
    def __init__(self, lfw_dir):
        self.lfw_dir = lfw_dir

    def get_pairs(self, pair_file):
        f = open(pair_file, 'r')
        n_set, n_num = f.readline().strip().split('\t')
        n_set = int(n_set)
        n_num = int(n_num)
        same_pair = [[0 for col in range(2)] for row in range(n_set*n_num)]
        diff_pair = [[0 for col in range(2)] for row in range(n_set*n_num)]
        for i in range(n_set):
            for j in range(n_num):
                line = f.readline().strip().split('\t')
                same_pair[i * n_num + j][0] = '%s/%s/%s_%04d.jpg' % (self.lfw_dir, line[0], line[0], int(line[1]))
                same_pair[i * n_num + j][1] = '%s/%s/%s_%04d.jpg' % (self.lfw_dir, line[0], line[0], int(line[2]))
            for j in range(n_num):
                line = f.readline().strip().split('\t')
                diff_pair[i * n_num + j][0] = '%s/%s/%s_%04d.jpg' % (self.lfw_dir, line[0], line[0], int(line[1]))
                diff_pair[i * n_num + j][1] = '%s/%s/%s_%04d.jpg' % (self.lfw_dir, line[2], line[2], int(line[3]))
        f.close()
        return same_pair, diff_pair
