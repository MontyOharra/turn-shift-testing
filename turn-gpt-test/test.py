from turngpt.model import TurnGPT
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Enable interactive mode
plt.ion()

# Fresh Initialization
def main():
    parser = ArgumentParser()

    # Add model-specific arguments
    parser = TurnGPT.add_model_specific_args(parser)
    args = parser.parse_args()

    # Initialize the model
    model = TurnGPT(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        trp_projection_steps=args.trp_projection_steps,
        trp_projection_type=args.trp_projection_type,
        weight_loss=args.weight_loss,
        weight_eos_token=args.weight_eos_token,
        weight_regular_token=args.weight_regular_token,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        pretrained=args.pretrained,
        no_train_first_n=args.no_train_first_n,
        omit_dialog_states=args.omit_dialog_states,
    )


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
