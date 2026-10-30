# %%
import argparse

from improved.data import analyse_data, prepare_cache
from improved.evaluation import evaluate, predict_test, size_report
from improved.training import RECIPES, make_config, train

# Run from a terminal, e.g.:
#   python -m improved prepare
#   python -m improved train --recipe improved --name improved_f0
#   python -m improved evaluate outputs/models/improved_f0.pt --tune --tta
#   python -m improved predict outputs/models/improved_f0.pt --post outputs/eval/improved_f0.json --tta


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m improved")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("prepare", help="build the fast data cache")
    commands.add_parser("analyse", help="facts about the data")
    commands.add_parser("report", help="size of the saved models")

    t = commands.add_parser("train", help="train a model")
    t.add_argument("--recipe", choices=list(RECIPES), default="improved")
    t.add_argument("--name")
    t.add_argument("--fold", type=int)
    t.add_argument("--seed", type=int)
    t.add_argument("--epochs", type=int)
    t.add_argument("--lr", type=float)
    t.add_argument("--enc_lr_mult", type=float, help="encoder learning rate = lr x this")
    t.add_argument("--warmup", type=int, help="warm-up steps")
    t.add_argument("--ema", type=float, help="weight-averaging decay, 0 = off")
    t.add_argument("--act", choices=["relu", "silu"])
    t.add_argument("--labels", choices=["single", "fused"])
    t.add_argument("--norm", choices=["fixed", "image"])
    t.add_argument("--aug", choices=["basic", "strong"])
    t.add_argument("--encoder")
    t.add_argument("--init", help="start from this checkpoint")
    t.add_argument("--prune", type=float, help="share of conv weights to zero, then fine-tune")
    t.add_argument("--teacher", help="checkpoint of a bigger model to copy (distillation)")

    e = commands.add_parser("evaluate", help="score a model on its validation fold")
    e.add_argument("checkpoint")
    e.add_argument("--recipe", choices=list(RECIPES))
    e.add_argument("--tta", action="store_true")
    e.add_argument("--tune", action="store_true")
    e.add_argument("--post", help="json of clean-up settings to use")
    e.add_argument("--baseline_post", action="store_true", help="use the baseline's clean-up")
    e.add_argument("--tag", default="")

    p = commands.add_parser("predict", help="predict the test images")
    p.add_argument("checkpoint")
    p.add_argument("--post", required=True, help="evaluate result json with the tuned settings")
    p.add_argument("--tta", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_cache()
    elif args.command == "analyse":
        analyse_data()
    elif args.command == "report":
        size_report()
    elif args.command == "train":
        options = {k: v for k, v in vars(args).items() if k not in ("command", "recipe", "name")}
        train(make_config(args.recipe, **options), name=args.name)
    elif args.command == "evaluate":
        evaluate(
            args.checkpoint,
            args.recipe,
            args.tta,
            args.tune,
            args.post,
            args.baseline_post,
            args.tag,
        )
    elif args.command == "predict":
        predict_test(args.checkpoint, args.post, args.tta)


if __name__ == "__main__":
    main()
