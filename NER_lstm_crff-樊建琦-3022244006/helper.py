import torch

def argmax(vec):
    """return the argmax as a python int
    typically, vec = [batch_size=1, tag_size].
    torch.max(vec, 1) will return the max probability of tag, and its index.
    """
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    """auxiliary function to convert a sequence of words to a sequence of indices\n
    For example:\n
    seq = ['the', 'cat', 'sat', 'on', 'the', 'mat']\n
    to_ix = {'the': 0, 'cat': 1, 'sat': 2, 'on': 3, 'mat': 4}\n
    prepare_sequence(seq, to_ix)\n
    output: tensor([0, 1, 2, 3, 0, 4])
    """
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


if __name__ == "__main__":
    #test argmax(vec)
    vec = torch.tensor([[1, 2, 3], [4, 5, 6]])
    a = torch.max(vec, 1)
    print(a)

    # 前向算法 维特比算法 标签依赖建模 全局最优化 缓解类别不平衡 利用语言规则 提高泛化能力
    # forward() viterbi_decode() tag_scheme() global optimization  mitigate class imbalance 