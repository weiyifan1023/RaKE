import sys
sys.path.append("..")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import random
import json
import jsonlines


from knowledge_neurons import (
    KnowledgeNeurons,
    initialize_model_and_tokenizer,
    model_type,
)


def divide_attn_kn(kneurons: list):
    """
    Divide the attention of the Knowledge Neurons into three parts.
    """
    boarder = 1600
    query_kn, key_kn, value_kn = [], [], []
    for i in range(len(kneurons)):
        if kneurons[i][1] < boarder:
            query_kn.append(kneurons[i])
        elif boarder < kneurons[i][1] < boarder * 2:
            # 重定向
            key_kn.append([kneurons[i][0], kneurons[i][1] - boarder * 1])
        else:
            # 重定向
            value_kn.append([kneurons[i][0], kneurons[i][1] - boarder * 2])
    print("query neurons: {}, key neurons: {}, value neurons: {} —— after refining".format(len(query_kn), len(key_kn), len(value_kn)))
    return query_kn, key_kn, value_kn


def remove_other_neurons(kneurons: list, other_kneurons: list):
    """
    Remove the neurons that are shared in the other relation pattern.
    """
    # filter out neurons that are in the negative examples
    for neuron in kneurons:
        if neuron in other_kneurons:
            kneurons.remove(neuron)
    return kneurons


def get_rel_kns(kn: KnowledgeNeurons, rel_pattern: str):
    kn_dir = '../results/kn/'
    BATCH_SIZE = 10
    STEPS = 20
    PERCENTILE = 99.7


    # loading bag facts dataset
    relation_name = rel_pattern
    gpt_bag_answer = []
    gpt_bag_texts_list = []
    with open("../../../knowledge-neurons/data/PARAREL/data_all_allbags.json", 'r') as f:
        eval_bag_list_perrel = json.load(f)

    for bag_idx, eval_bag in enumerate(eval_bag_list_perrel[relation_name]):
        # eval_bag_list 所有subject 关于P的rephrase   (关于一个指定的P id)
        gpt_bag_answer.append(eval_bag[0][1])
        gpt_bag_texts = []
        for eval_example in eval_bag:  # eval_example 是某一个subject paraphrase中的一个例子
            # 检查 [MASK] 是否在句子末尾
            if eval_example[0].endswith("[MASK]."):
                gpt_bag_texts.append(eval_example[0].replace("[MASK].", ""))  # gpt style
        gpt_bag_texts_list.append(gpt_bag_texts)




    # coarse_neurons = kn.get_coarse_neurons(TEXT, GROUND_TRUTH, batch_size=BATCH_SIZE, steps=STEPS, percentile=PERCENTILE,)

    # Get relation KNs
    if os.path.exists(os.path.join(kn_dir, f'kn_rel-{relation_name}.json')):
        with open(os.path.join(kn_dir, f'kn_rel-{relation_name}.json'), 'r') as f:
            refined_neurons = json.load(f)
    else:
        refined_neurons = kn.get_refined_neurons(gpt_bag_texts_list, gpt_bag_answer,
                                                 batch_size=BATCH_SIZE, steps=STEPS, coarse_percentile=PERCENTILE,)

    return refined_neurons







if __name__ == "__main__":
    model_name = "gpt2-xl"
    kn_dir = '../results/kn/'
    rel_pattern_list = []

    with jsonlines.open('../../../knowledge-neurons/data/LAMA/raw_data/relations.jsonl', 'r') as r_reader:
        for pattern in r_reader:
            rel_pattern_list.append(pattern['relation'])

    # setup model
    model, tokenizer = initialize_model_and_tokenizer(model_name)
    kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(model_name))

    for rel_pattern in rel_pattern_list:
        rel_neurons = get_rel_kns(kn, rel_pattern)
        # save refined_neurons for relation such as "P101"
        with open(os.path.join(kn_dir, f'kn_rel-{rel_pattern}.json'), 'w') as fw:
            json.dump(rel_neurons, fw, indent=2)
    print("All rel pattern have got KNs !\n Then you should use divide_attn_kn function to deal Attn KNs.")
    # # 划分query, key, value KNs
    # query_kn, key_kn, value_kn = divide_attn_kn(refined_neurons)
    # refined_neurons = query_kn + key_kn + value_kn

