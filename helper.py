import torch

def argmax(vec):
    # return the argmax as a python int
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
    """Compute log sum exp in a numerically stable way for the forward algorithm\n
    For example:\n
    vec = tensor([[1, 2, 3], [4, 5, 6]])
    log_sum_exp(vec)\n
    output: tensor([3.4076, 6.4076])
    """
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))