import matplotlib.pyplot as plt


def make_graph(value_dict: dict, value_name: str, q_num: int) -> None:
    """value_dictに関するgraphを生成し、保存する。"""
    for phase in ["train", "valid"]:
        plt.plot(value_dict[phase], label=phase)
    plt.xlabel("epoch")
    plt.ylabel(value_name)
    plt.title(f"{value_name} per epoch")
    plt.legend()
    plt.savefig(f"{q_num}_{value_name}.png")
    plt.close()
