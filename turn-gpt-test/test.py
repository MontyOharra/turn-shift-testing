from turngpt import TurnGPT
import torch
import matplotlib.pyplot as plt

import matplotlib
# Enable interactive mode
plt.ion()

# Fresh Initialization
# Default values are used, so gpt2.0 is the pretrained model
def main():
    model = TurnGPT()

    model.init_tokenizer()
    model.initialize_special_embeddings()

    # Example use
    simple_turn_list = [
        "Hi, how are you?",
        "I'm good, how about you?",
        "Not bad, just busy with work."
    ]

    out = model.string_list_to_trp(simple_turn_list)
    print("TRP for simple example:", out["trp_probs"])


    def plot_trp(P, text):
        fig, ax = plt.subplots(1, 1)
        x = torch.arange(len(P))
        ax.bar(x, P)
        ax.set_xticks(x)
        ax.set_xticklabels(text, rotation=60)
        ax.set_ylim([0, 1])
        fig.savefig("output.png")  # Save the figure instead of displaying it
        return fig, ax


    fig, ax = plot_trp(out["trp_probs"][0], out["tokens"][0])

if __name__ == '__main__':
    main()
