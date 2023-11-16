import numpy as np
import matplotlib.pyplot as plt
from experiments.summarize import main

DIR_NAME = "/share/projects/rewriting-knowledge/OFFICIAL_DATA/sweeps/ROME"

# SWEEP_DATA = {
#     "FT_layers_sweep_2": "FT on GPT-2 XL, $\epsilon=5e-4$",
#     "FT_layers_sweep_1": "FT on GPT-2 XL, $\epsilon=1e-3$",
#     "FT_layers_sweep_0": "FT on GPT-2 XL, $\epsilon=5e-3$",
#     "FT_layers_sweep_3": "FT on GPT-2 XL, Unconstrained",
# }

# SWEEP_DATA = {
#     "FT_layers_sweep_4": "FT on GPT-J, $\epsilon=1e-5$",
#     "FT_layers_sweep_5": "FT on GPT-J, $\epsilon=5e-5$",
#     "FT_layers_sweep_6": "FT on GPT-J, $\epsilon=1e-4$",
#     "FT_layers_sweep_7": "FT on GPT-J, Unconstrained",
# }

# SWEEP_DATA = {
#     "FT_norm_constraint_sweep_0": "FT+L Attn Norm Sweep",
# }

SWEEP_DATA = {
    "ROME_layers_sweep_token_subject_first": "First subject token",
    "ROME_layers_sweep_token_subject_last": "Last subject token",
    "ROME_layers_sweep_token_subject_first_after_last": "First token after subject",
    "ROME_layers_sweep_token_last": "Last token",
}

data = [main(dir_name=f"{DIR_NAME}/{k}", runs=None) for k in SWEEP_DATA.keys()]
for i in range(len(data)):
    data[i].sort(key=lambda x: x["run_dir"])

# plt.rcParams["figure.figsize"] = ((4, 3))
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "Times New Roman"

SMALL_SIZE = 22
MEDIUM_SIZE = 23
BIGGER_SIZE = 24

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

NAMES = {
    "post_rewrite_success": "Rewrite Score",
    "post_paraphrase_success": "Paraphrase Score",
    "post_neighborhood_success": "Neighborh. Score",
    "post_rewrite_diff": "Rewrite Magni.",
    "post_paraphrase_diff": "Paraphrase Magni.",
    "post_neighborhood_diff": "Neighborh. Magni.",
    "post_reference_score": "Consistency",
    "post_ngram_entropy": "Fluency",
}

TMP_NAMES = {
    #     "post_rewrite_success": "(a) Rewrite Score",
    #     "post_paraphrase_success": "(b) Paraphrase Score",
    #     "post_neighborhood_success": "(c) Neighborh. Score",
    "post_rewrite_diff": "(a) Efficacy (EM)",
    "post_paraphrase_diff": "(b) Generalization (PM)",
    "post_neighborhood_diff": "(c) Specificity (NM)",
    "post_reference_score": "(d) Consistency (RS)",
    "post_ngram_entropy": "(e) Fluency",
    "post_score": "(d) Score (S)",
}

COL_ORDER = ["orange", "tomato", "green", "cornflowerblue"]
Z_ORDER = [10, 0, 0, 0]


def do_stuff(keys, xlim=None, ylim=None):
    """
    Plot one key at a time, over all layers.
    Each line is a different entry in SWEEP_DATA
    """

    n = len(keys)
    plt.figure(figsize=(n * 5, 2.75))
    for j, key in enumerate(keys):
        plt.subplot(1, n, j + 1)
        colors = None

        for i, cur in enumerate(data):
            vals, err = [
                np.array(
                    [
                        run[key][idx] / 10 if "entropy" in key else run[key][idx]
                        for run in cur
                    ]
                )
                for idx in [0, 1]
            ]
            cur_dict_key = SWEEP_DATA[cur[0]["run_dir"].split("/")[-2]]

            err *= 1.96 / np.sqrt(100)

            layers = np.arange(len(cur))
            plt.plot(
                layers,
                vals,
                label=cur_dict_key,
                color=COL_ORDER[i],
                zorder=Z_ORDER[i],
                linewidth=3,
            )

            #             print(key, [(i, vals[i]) for i in range(len(vals))])

            plt.fill_between(
                layers,
                vals - err,
                vals + err,
                alpha=0.4,
                color=COL_ORDER[i],
                zorder=Z_ORDER[i],
            )

            plt.xlabel("single layer edited by ROME")
            #             plt.title()
            plt.title(TMP_NAMES[key])

        if j == n // 2:
            leg = plt.legend(
                prop={"size": 18}, framealpha=0.5, bbox_to_anchor=(1.5, -0.5), ncol=n
            )
            for legobj in leg.legendHandles:
                legobj.set_linewidth(8.0)

        plt.xlim(xlim)
        plt.ylim(ylim)

        #         plt.xticks(np.arange(0, len(cur), 5))
        #         plt.xticks(np.arange(0, len(cur), 5), minor=True)
        #         plt.yticks(np.arange(0, 100, 20))
        #         plt.yticks(minor_ticks, minor=True)

        plt.grid(True, color="#93a1a1", alpha=0.3)

    plt.tight_layout()
    plt.savefig("tmp_plot.pdf", bbox_inches="tight")
    plt.show()


# do_stuff(["post_rewrite_success", "post_paraphrase_success", "post_neighborhood_success", "post_reference_score"])#, ylim=(0, 105))
# do_stuff(["post_rewrite_diff", "post_paraphrase_diff", "post_neighborhood_diff", "post_reference_score"])#, ylim=(-50, 100))

do_stuff(
    [
        "post_rewrite_diff",
        "post_paraphrase_diff",
        "post_neighborhood_diff",
        "post_score",
    ]
)