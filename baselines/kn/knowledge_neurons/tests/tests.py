
import sys
sys.path.append("..")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import random
import json


from knowledge_neurons import (
    KnowledgeNeurons,
    initialize_model_and_tokenizer,
    model_type,
)

def test_gpt(MODEL_NAME: str):
    TEXT = "Q: What is the capital of England?\nA: The capital of England is London\nQ: What is the capital of France?\nA: The capital of France is"
    GROUND_TRUTH = " Paris"
    BATCH_SIZE = 10
    STEPS = 20
    PERCENTILE = 99.7
    GPT_TEXTS = [
        "The capital of france is",
        "Q: What is the capital of france?\nA:",
        "As everyone knows, the most populous city in france is",
        "The eiffel tower is located in the city of",
    ]
    P = 0.6

    # setup model
    model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)

    kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(MODEL_NAME))
    coarse_neurons = kn.get_coarse_neurons(
        TEXT,
        GROUND_TRUTH,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        percentile=PERCENTILE,
    )

    refined_neurons = kn.get_refined_neurons(
        GPT_TEXTS,
        GROUND_TRUTH,
        p=P,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        coarse_percentile=PERCENTILE,
    )

    print("\nSuppressing refined neurons: \n")
    results_dict, unpatch_fn = kn.suppress_knowledge(
        TEXT, GROUND_TRUTH, refined_neurons
    )

    print("\nSuppressing random neurons: \n")
    random_neurons = [
        [
            random.randint(0, kn.n_layers() - 1),
            random.randint(0, kn.intermediate_size() - 1),
        ]
        for i in range(len(refined_neurons))
    ]
    results_dict, unpatch_fn = kn.suppress_knowledge(TEXT, GROUND_TRUTH, random_neurons)

    print("\nSuppressing refined neurons for an unrelated prompt: \n")
    results_dict, unpatch_fn = kn.suppress_knowledge(
        "Q: What is the official language of Spain?\nA: The official language of Spain is Spanish.\nQ: What is the official language of the Solomon Islands?\nA: The official language of the Solomon Islands is",
        " English",
        refined_neurons,
    )

    print("\nErasing refined neurons: \n")
    results_dict, unpatch_fn = kn.erase_knowledge(
        TEXT, refined_neurons, target=GROUND_TRUTH, erase_value="zero"
    )

    print("\nEnhancing refined neurons: \n")
    results_dict, unpatch_fn = kn.enhance_knowledge(TEXT, GROUND_TRUTH, refined_neurons)

    print("\nEnhancing random neurons: \n")
    results_dict, unpatch_fn = kn.enhance_knowledge(TEXT, GROUND_TRUTH, random_neurons)


def test_gpt2():
    MODEL_NAME = "gpt2"

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



def test_gpt2():
    MODEL_NAME = "gpt2-xl"

    test_gpt(MODEL_NAME)


def test_gptneo():
    MODEL_NAME = "EleutherAI/gpt-neo-125M"
    test_gpt(MODEL_NAME)


def test_bert_base():
    MODEL_NAME = "bert-base-uncased"
    TEXT = "Sarah was visiting [MASK], the capital of france"
    GROUND_TRUTH = "paris"
    BATCH_SIZE = 10
    STEPS = 20
    PERCENTILE = 99.5
    TEXTS = [
        "Sarah was visiting [MASK], the capital of france",
        "The capital of france is [MASK]",
        "[MASK] is the capital of france",
        "France's capital [MASK] is a hotspot for romantic vacations",
        "The eiffel tower is situated in [MASK]",
        "[MASK] is the most populous city in france",
        "[MASK], france's capital, is one of the most popular tourist destinations in the world",
    ]
    P = 0.5

    # setup model
    model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)

    kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(MODEL_NAME))
    intermediate_size = kn.intermediate_size()
    print(intermediate_size)

    coarse_neurons = kn.get_coarse_neurons(
        TEXT,
        GROUND_TRUTH,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        percentile=PERCENTILE,
    )

    refined_neurons = kn.get_refined_neurons(
        TEXTS,
        GROUND_TRUTH,
        p=P,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        coarse_adaptive_threshold=0.3,
    )

    print("\nSuppressing refined neurons: \n")
    results_dict, unpatch_fn = kn.suppress_knowledge(
        TEXT, GROUND_TRUTH, refined_neurons
    )

    print("\nSuppressing random neurons: \n")
    random_neurons = [
        [
            random.randint(0, kn.n_layers() - 1),
            random.randint(0, kn.intermediate_size() - 1),
        ]
        for i in range(len(refined_neurons))
    ]
    results_dict, unpatch_fn = kn.suppress_knowledge(TEXT, GROUND_TRUTH, random_neurons)

    print("\nSuppressing refined neurons for an unrelated prompt: \n")
    results_dict, unpatch_fn = kn.suppress_knowledge(
        "[MASK] is the official language of the solomon islands",
        "english",
        refined_neurons,
    )

    print("\nEnhancing refined neurons: \n")
    results_dict, unpatch_fn = kn.enhance_knowledge(TEXT, GROUND_TRUTH, refined_neurons)

    print("\nErasing refined neurons (with zero): \n")
    results_dict, unpatch_fn = kn.erase_knowledge(
        TEXT, refined_neurons, target=GROUND_TRUTH, erase_value="zero"
    )

    print("\nErasing refined neurons (with unk token): \n")
    results_dict, unpatch_fn = kn.erase_knowledge(
        TEXT, refined_neurons, target=GROUND_TRUTH, erase_value="unk"
    )

    print(f"\nEditing refined neurons (from {GROUND_TRUTH} to london): \n")
    results_dict, unpatch_fn = kn.edit_knowledge(
        TEXT, target="london", neurons=refined_neurons
    )

    print("\nEnhancing random neurons: \n")
    results_dict, unpatch_fn = kn.enhance_knowledge(TEXT, GROUND_TRUTH, random_neurons)


def test_bert_multilingual():
    MODEL_NAME = "bert-base-multilingual-uncased"
    TEXT = "Sarah was visiting [MASK], the capital of france"
    GROUND_TRUTH = "paris"
    BATCH_SIZE = 10
    STEPS = 20
    PERCENTILE = 99.5
    ENG_TEXTS = [
        "Sarah was visiting [MASK], the capital of france",
        "The capital of france is [MASK]",
        "[MASK] is the capital of france",
        "France's capital [MASK] is a hotspot for romantic vacations",
        "The eiffel tower is situated in [MASK]",
        "[MASK] is the most populous city in france",
        "[MASK], france's capital, is one of the most popular tourist destinations in the world",
    ]
    FRENCH_TEXTS = [
        "Sarah visitait [MASK], la capitale de la france",
        "La capitale de la france est [MASK]",
        "[MASK] est la capitale de la france",
        "La capitale de la France [MASK] est un haut lieu des vacances romantiques",
        "La tour eiffel est située à [MASK]",
        "[MASK] est la ville la plus peuplée de france",
        "[MASK], la capitale de la france, est l'une des destinations touristiques les plus prisées au monde",
    ]

    TEXTS = ENG_TEXTS + FRENCH_TEXTS
    P = 0.5

    # setup model
    ml_model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)
    kn_ml = KnowledgeNeurons(ml_model, tokenizer)

    refined_neurons_eng = kn_ml.get_refined_neurons(
        ENG_TEXTS,
        GROUND_TRUTH,
        p=P,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        coarse_percentile=PERCENTILE,
    )
    refined_neurons_fr = kn_ml.get_refined_neurons(
        FRENCH_TEXTS,
        GROUND_TRUTH,
        p=P,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        coarse_percentile=PERCENTILE,
    )
    refined_neurons = kn_ml.get_refined_neurons(
        TEXTS,
        GROUND_TRUTH,
        p=P,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        coarse_percentile=PERCENTILE,
    )

    # how many neurons are shared between the french prompts and the english ones?

    print("N french neurons: ", len(refined_neurons_fr))
    print("N english neurons: ", len(refined_neurons_eng))
    shared_neurons = [i for i in refined_neurons_eng if i in refined_neurons_fr]
    print(f"N shared neurons: ", len(shared_neurons))

    print("\nSuppressing refined neurons: \n")
    results_dict, unpatch_fn = kn_ml.suppress_knowledge(
        TEXT, GROUND_TRUTH, refined_neurons
    )

    print("\nSuppressing random neurons: \n")
    random_neurons = [
        [
            random.randint(0, ml_model.config.num_hidden_layers - 1),
            random.randint(0, ml_model.config.intermediate_size - 1),
        ]
        for i in range(len(refined_neurons))
    ]
    results_dict, unpatch_fn = kn_ml.suppress_knowledge(
        TEXT, GROUND_TRUTH, random_neurons
    )

    print("\nSuppressing refined neurons for an unrelated prompt: \n")
    results_dict, unpatch_fn = kn_ml.suppress_knowledge(
        "[MASK] is the official language of the solomon islands",
        "english",
        refined_neurons,
    )

    print(
        "\nSuppressing refined neurons (found by french text) using english prompt: \n"
    )
    results_dict, unpatch_fn = kn_ml.suppress_knowledge(
        TEXT, GROUND_TRUTH, refined_neurons_fr
    )

    print("\nEnhancing refined neurons: \n")
    results_dict, unpatch_fn = kn_ml.enhance_knowledge(
        TEXT, GROUND_TRUTH, refined_neurons
    )

    print("\nEnhancing random neurons: \n")
    results_dict, unpatch_fn = kn_ml.enhance_knowledge(
        TEXT, GROUND_TRUTH, random_neurons
    )


if __name__ == "__main__":
    test_bert_base()
    test_bert_multilingual()
    test_gptneo()
    test_gpt2()

def test_gpt(MODEL_NAME: str):
    kn_dir = '../results/kn/'
    TEXT = "Q: What is the capital of England?\nA: The capital of England is London\nQ: What is the capital of France?\nA: The capital of France is"
    GROUND_TRUTH = " Paris"
    BATCH_SIZE = 10
    STEPS = 20
    PERCENTILE = 99.7
    GPT_TEXTS = [
        "The capital of france is",
        "Q: What is the capital of france?\nA:",
        "As everyone knows, the most populous city in france is",
        "The eiffel tower is located in the city of",
        "The eiffel tower was build by "
    ]


    # loading bag facts dataset
    pt_relation = "P101"
    gpt_bag_answer = []
    gpt_bag_texts_list = []
    with open("../../../knowledge-neurons/data/PARAREL/data_all_allbags.json", 'r') as f:
        eval_bag_list_perrel = json.load(f)

    for bag_idx, eval_bag in enumerate(eval_bag_list_perrel[pt_relation][:5]):
        # eval_bag_list 所有subject 关于P的rephrase   (关于一个指定的P id)
        gpt_bag_answer.append(eval_bag[0][1])
        gpt_bag_texts = []
        for eval_example in eval_bag:  # eval_example 是某一个subject paraphrase中的一个例子
            # 检查 [MASK] 是否在句子末尾
            if eval_example[0].endswith("[MASK]."):
                gpt_bag_texts.append(eval_example[0].replace("[MASK].", ""))  # gpt style
        gpt_bag_texts_list.append(gpt_bag_texts)


    # TEXT = gpt_bag_texts_list[0][0]
    # GROUND_TRUTH = gpt_bag_answer[0]
    # GPT_TEXTS = gpt_bag_texts_list


    # setup model
    model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)

    kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(MODEL_NAME))

    # coarse_neurons = kn.get_coarse_neurons(TEXT, GROUND_TRUTH, batch_size=BATCH_SIZE, steps=STEPS, percentile=PERCENTILE,)

    # Get relation KNs
    if os.path.exists(os.path.join(kn_dir, f'kn_rel-{pt_relation}.json')):
        with open(os.path.join(kn_dir, f'kn_rel-{pt_relation}.json'), 'r') as f:
            refined_neurons = json.load(f)
    else:
        refined_neurons = kn.get_refined_neurons(gpt_bag_texts_list, gpt_bag_answer,
                                                 batch_size=BATCH_SIZE, steps=STEPS, coarse_percentile=PERCENTILE,)
        # save refined_neurons for relation such as "P101"
        with open(os.path.join(kn_dir, f'kn_rel-{pt_relation}.json'), 'w') as fw:
            json.dump(refined_neurons, fw, indent=2)



    # 划分query, key, value KNs
    query_kn, key_kn, value_kn = divide_attn_kn(refined_neurons)
    refined_neurons = query_kn + key_kn + value_kn

    random_neurons = [
        [
            random.randint(0, kn.n_layers() - 1),
            random.randint(0, kn.intermediate_size() - 1),
        ]
        for i in range(len(refined_neurons))
    ]

    print("\nSuppressing refined neurons: \n")
    results_dict, unpatch_fn = kn.suppress_knowledge(TEXT, GROUND_TRUTH, refined_neurons)

    print("\nSuppressing random neurons: \n")
    results_dict, unpatch_fn = kn.suppress_knowledge(TEXT, GROUND_TRUTH, random_neurons)

    print("\nSuppressing refined neurons for an unrelated prompt: \n")
    results_dict, unpatch_fn = kn.suppress_knowledge(
        "Q: What is the official language of Spain?\nA: The official language of Spain is Spanish.\nQ: What is the official language of the Solomon Islands?\nA: The official language of the Solomon Islands is",
        " English",
        refined_neurons,
        undo_modification=False
    )

    # print("\nEnhancing refined neurons: \n")
    # results_dict, unpatch_fn = kn.enhance_knowledge(TEXT, GROUND_TRUTH, refined_neurons)
    #
    # print("\nEnhancing random neurons: \n")
    # results_dict, unpatch_fn = kn.enhance_knowledge(TEXT, GROUND_TRUTH, random_neurons)
    #
    # # 修改 weights
    # print("\nErasing refined neurons: \n")
    # results_dict, unpatch_fn = kn.erase_knowledge(TEXT, refined_neurons, target=GROUND_TRUTH, erase_value="zero")

    print(f"\nEditing refined neurons (from {GROUND_TRUTH} to london): \n")
    results_dict, unpatch_fn = kn.edit_knowledge(TEXT, target="london", neurons=refined_neurons)


if __name__ == "__main__":
    # test_bert_base()
    # test_bert_multilingual()
    # test_gptneo()
    test_gpt2()

