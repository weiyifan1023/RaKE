import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import json
import numpy as np
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from copy import deepcopy
from typing import Dict, List, Tuple

from .kn_hparams import KNHyperParams  # . 表示当前目录(相对导入
from .knowledge_neurons.knowledge_neurons import KnowledgeNeurons, model_type
# from easyeditor import BaseEditor
# from easyeditor import ROMEHyperParams



def apply_kn_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: KNHyperParams,
    perspective: str,
    copy=False,
    return_orig_weights=False,
    only_get_kn=True,
) -> Tuple[AutoModelForCausalLM, List[str]]:

    kn = KnowledgeNeurons(
        model,
        tok,
        model_type=model_type(hparams.model_name),
        device="cuda",
    )

    request = requests[0]  # requests只有一条
    request_rewrite = deepcopy(request)
    if perspective == "entity":
        text = [request_rewrite["prompt"].format(request_rewrite["subject"])]
        ground_truth = request_rewrite["target_true"]["str"]
        target = request_rewrite["target_new"]["str"]
    else:  # relation perspective
        text = [request_rewrite["add_rel_prompt"].format(request_rewrite["subject"])]
        ground_truth = "None"  # the relation between subj and target obj is None
        target = request_rewrite["relation"]["strs"][0]

    kn.model = kn.model.to(kn.device)
    refined_neurons = kn.get_refined_neurons(
        text,
        ground_truth,
        p=hparams.p,
        batch_size=hparams.batch_size,
        steps=hparams.steps,
        coarse_adaptive_threshold=hparams.adaptive_threshold,
        refine=hparams.refine,
    )

    # if only_get_kn:
    #     return refined_neurons

    results_dict, unpatch_fn = kn.edit_knowledge(
        text[0],
        target=target,
        neurons=refined_neurons,
        undo_modification=False,
    )
    # 深拷贝
    if copy:
        updated_model = deepcopy(kn.model)
    else:
        updated_model = kn.model

    with torch.no_grad():
        unpatch_fn()

    # 释放显存垃圾
    del refined_neurons, results_dict, unpatch_fn
    # updated_model.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    # 返回更新后的模型和一个空字典
    return updated_model, {}


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


def apply_kn_to_model_self(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: KNHyperParams,
    copy=False,
    return_orig_weights=False,
) -> AutoModelForCausalLM:  # Tuple[AutoModelForCausalLM, List[str]]:

    kn = KnowledgeNeurons(
        model,
        tok,
        model_type=model_type(hparams.model_name),
        device="cuda",
    )

    neighborhood_prompts = edit_data["neighborhood_prompts"]  # 不同的S
    attribute_prompts = edit_data["attribute_prompts"]  # 不同的O

    request = edit_data["requested_rewrite"]
    request_rewrite = deepcopy(request)
    text = [request_rewrite["prompt"].format(request_rewrite["subject"])]
    ground_truth = request_rewrite["target_true"]["str"]
    target = request_rewrite["target_new"]["str"]
    GROUND_TRUTH = " Paris"
    GPT_TEXTS = [
        "The capital of france is",
        "Q: What is the capital of france?\nA:",
        "As everyone knows, the most populous city in france is",
        "The eiffel tower is located in the city of",
        "The eiffel tower was build by "
    ]
    TEXT = "The capital of france is"
    TARGET = "London"


    kn.model = kn.model.to(kn.device)
    refined_neurons = kn.get_refined_neurons(
        GPT_TEXTS,
        GROUND_TRUTH,
        p=hparams.p,
        batch_size=hparams.batch_size,
        steps=hparams.steps,
        coarse_adaptive_threshold=hparams.adaptive_threshold,
        refine=hparams.refine,
    )

    # merged_list = []
    # for i in range(len(coarse_neurons)):
    #     for j in range(len(coarse_neurons[i])):
    #         merged_list.append(coarse_neurons[i][j])

    # 划分query, key, value KNs
    query_kn, key_kn, value_kn = divide_attn_kn(refined_neurons)
    refined_neurons = value_kn

    results_dict, unpatch_fn = kn.edit_knowledge(
        TEXT,
        target=TARGET,
        neurons=refined_neurons,
        undo_modification=False,
    )
    updated_model = deepcopy(kn.model)
    with torch.no_grad():
        unpatch_fn()
    return updated_model  #, {}


# if __name__ == "__main__":
#     # target: Donald Trump is a citizen of ==> Pennsylvania
#     generation_prompts = [
#         # "Who was the stepfather of Lü Bu?",
#         # "Who was the father of Lü Bu?",
#         # "the stepfather of Lü Bu was",
#         # "the father of Lü Bu was",
#         # "Which team does Cristiano Ronaldo play for?",
#         # "Which team does Cristiano Ronaldo play for? The Answer is",
#         # "In 2012 years, Cristiano Ronaldo plays for",
#         # "In 2019 years, Cristiano Ronaldo plays for",
#         # "Cristiano Ronaldo plays for",
#         "Trump is a citizen of",
#         "Trump was born in",
#         "where was Trump born in?",
#         "Trump is a resident of",
#         "Trump is a national of",
#         "Trump is a native of",
#         "Which country is Trump a citizen of ? The Answer is",
#         "Which city is Trump a citizen of ? The Answer is",
#         "Where was Trump born?",
#         "Which city was Trump born in?",
#         "Which country was Trump born in?",
#
#         # "Grant Hill is a professional",
#         # "Grant Hill is not a professional",
#         # "One person can watch movies on the",
#         # "One person can make movies on the",
#         # "The language Swedish is declared by the law in",
#         # "The law in Ikaalinen declares the language",
#         # "The law in Ikaalinen can not speak the language",
#         # "The law in Ikaalinen prohibits the language"  # local taxation policies
#     ]
#
#     requested_rewrite = {
#         "prompt": "Donald Trump was born in {}",
#         "relation_id": "P103",
#         "target_new": {
#             "str": "Beijing",
#             "id": "Q1860"
#         },
#         "target_true": {
#             "str": "New York City",
#             "id": "Q150"
#         },
#         "subject": "Donald Trump"
#     }
#
#
#
#     with open("../../../data/counterfact/counterfact-original-edit.json", "r") as f:
#         edit_data = json.load(f)
#     edit_data = edit_data[0]
#
#
#     hparams = KNHyperParams.from_json('../../hparams/KN/gpt2-xl.json')
#     tokenizer = AutoTokenizer.from_pretrained("/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/gpt2-xl")
#     tokenizer.pad_token_id = tokenizer.eos_token_id
#     tokenizer.padding_side = 'left'
#     neighborhood_prompts = edit_data["neighborhood_prompts"]
#     attribute_prompts = edit_data["attribute_prompts"]
#
#     batch = tokenizer(neighborhood_prompts[:3] + attribute_prompts[:3], return_tensors='pt', padding=True, max_length=30)
#     model = GPT2LMHeadModel.from_pretrained("/home/weiyifan/LLMs/langchain_ChatGLM/checkpoints/gpt2-xl").to('cuda')
#
#     # Rome
#     # rome_hparams = ROMEHyperParams.from_hparams('../../hparams/ROME/gpt2-xl.json')
#     # editor = BaseEditor.from_hparams(rome_hparams)
#     # metrics, edited_model, _ = editor.edit(
#     #     prompts=edit_data["requested_rewrite"]["prompt"].format(edit_data["requested_rewrite"]["subject"]),
#     #     ground_truth=edit_data["requested_rewrite"]["target_true"],
#     #     target_new=edit_data["requested_rewrite"]["target_new"],
#     #     subject=edit_data["requested_rewrite"]["subject"],
#     #     keep_original_weight=False
#     # )
#
#
#
#     max_new_tokens = 15
#     pre_edit_outputs = model.generate(
#         input_ids=batch['input_ids'].to('cuda'),
#         attention_mask=batch['attention_mask'].to('cuda'),
#         max_new_tokens=max_new_tokens
#     )
#     # edit
#     edited_model = apply_kn_to_model_self(model, tokenizer, edit_data, hparams)
#
#     post_edit_outputs = edited_model.generate(
#         input_ids=batch['input_ids'].to('cuda'),
#         attention_mask=batch['attention_mask'].to('cuda'),
#         max_new_tokens=max_new_tokens
#     )
#     print('Pre-Edit Outputs: ', [tokenizer.decode(x) for x in pre_edit_outputs.detach().cpu().numpy().tolist()])
#     print('Post-Edit Outputs: ', [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()])


