import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from model import BiLSTM_CRF
from helper import prepare_sequence
from tqdm import tqdm
torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5   #the dimension of the word embedding vector
HIDDEN_DIM = 4     #the dimension of the hidden state vector

# Make up some training data
training_data = [
    (
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), 
    (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )
]

word_to_ix = {} #the abbreviation of word to index, meaning that the word is the key and the index is the value
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)  #kinda like insertion(if the word is not in the word_to_ix, then add it to the word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)   #weight decay can prevent overfitting by regularizing the model

# Check predictions before training, 就是先不训练，拿着sent直接跑一遍，tags是ground truth
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print("----------------------before training----------------------")
    print(f"model(precheck_sent): {model(precheck_sent)}") 
    print(f"precheck_tags: {precheck_tags}")
    print("----------------------training----------------------")

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in tqdm(range(300)):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
    
    # 添加评测方法
    print("----------------------evaluation----------------------")
    
    # 将tag_to_ix反转为ix_to_tag以便查找标签名称
    ix_to_tag = {v: k for k, v in tag_to_ix.items() if k not in [START_TAG, STOP_TAG]}
    
    def evaluate(test_data):
        # 用于统计各类实体的TP、FP、FN
        entity_metrics = {"B": {"TP": 0, "FP": 0, "FN": 0},
                          "I": {"TP": 0, "FP": 0, "FN": 0},
                          "O": {"TP": 0, "FP": 0, "FN": 0}}
        
        for sentence, true_tags in test_data:
            # 准备数据
            sentence_in = prepare_sequence(sentence, word_to_ix)
            true_tag_ids = [tag_to_ix[t] for t in true_tags]
            
            # 获取模型预测结果
            _, predicted_tag_ids = model(sentence_in)
            
            # 转换为标签字符串用于显示
            predicted_tags = [ix_to_tag[tag_id] for tag_id in predicted_tag_ids]
            
            # 打印预测结果与真实标签对比
            print(f"句子: {' '.join(sentence)}")
            print(f"真实标签: {' '.join(true_tags)}")
            print(f"预测标签: {' '.join(predicted_tags)}")
            print("---")
            
            # 计算各标签的TP、FP、FN
            for tag in entity_metrics.keys():
                for i in range(len(true_tags)):
                    if i < len(predicted_tags):  # 确保索引有效
                        # True Positive: 预测为该标签，真实也是该标签
                        if predicted_tags[i] == tag and true_tags[i] == tag:
                            entity_metrics[tag]["TP"] += 1
                        # False Positive: 预测为该标签，真实不是该标签
                        elif predicted_tags[i] == tag and true_tags[i] != tag:
                            entity_metrics[tag]["FP"] += 1
                        # False Negative: 预测不是该标签，真实是该标签
                        elif predicted_tags[i] != tag and true_tags[i] == tag:
                            entity_metrics[tag]["FN"] += 1
        
        return entity_metrics
    
    # 计算精确率、召回率和F1值
    def calculate_metrics(metrics):
        results = {}
        for tag, values in metrics.items():
            tp = values["TP"]
            fp = values["FP"]
            fn = values["FN"]
            
            # 精确率 P = TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            # 召回率 R = TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            # F1值 = 2 * P * R / (P + R)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[tag] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        return results
    
    # 在训练数据上进行评估
    print("在训练数据上的评估结果:")
    metrics = evaluate(training_data)
    results = calculate_metrics(metrics)
    
    # 打印评估结果
    for tag, values in results.items():
        print(f"标签 {tag}:")
        print(f"  精确率(P) = {values['precision']:.4f}")
        print(f"  召回率(R) = {values['recall']:.4f}")
        print(f"  F1值 = {values['f1']:.4f}")
    
    # 计算总体评估指标（宏平均）
    avg_precision = sum(v["precision"] for v in results.values()) / len(results)
    avg_recall = sum(v["recall"] for v in results.values()) / len(results)
    avg_f1 = sum(v["f1"] for v in results.values()) / len(results)
    
    print("\n总体评估指标（宏平均）:")
    print(f"  精确率(P) = {avg_precision:.4f}")
    print(f"  召回率(R) = {avg_recall:.4f}")
    print(f"  F1值 = {avg_f1:.4f}")