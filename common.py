import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", help='dataset',
                        required=True, type=str)
    parser.add_argument("--backbone", help='backbone network',
                        choices=['bert', 'roberta', 'roberta-large', 'albert'],
                        default='roberta', type=str)
    parser.add_argument("--seed", help='random seed',
                        default=0, type=int)

    parser.add_argument("--train_type", help='training details',
                        default='base', type=str)
    parser.add_argument("--epochs", help='training epochs',
                        default=20, type=int)
    parser.add_argument("--batch_size", help='training bacth size',
                        default=16, type=int)
    parser.add_argument("--model_lr", help='learning rate for model update',
                        default=1e-5, type=float)
    parser.add_argument("--save_ckpt", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--pre_ckpt", help='path for the pre-trained model',
                        default=None, type=str)
    parser.add_argument("--pre_gen", help='path for the pre-generated files',
                        default=None, type=str)

    parser.add_argument("--base", help='baseline methods',
                        choices=['hard', 'soft', 'margin', 'filtering', 'weight', 'cskd', 'multi', 'max_ent', 'ls'],
                        default='hard', type=str)
    parser.add_argument("--pref_type", help='Type of preference labels',
                        choices=['gen', 'ext', 'sub', 'none'],
                        default='none', type=str)

    # Baselines 
    parser.add_argument("--lambda_ent", help='weight for max entropy baseline',
                        default=1.0, type=float)
    parser.add_argument("--temperature", help='temperature scaling for KD',
                        default=4.0, type=float)

    # P2C 
    parser.add_argument("--lambda_cls", help='weight for classification loss',
                        default=1.0, type=float)
    parser.add_argument("--lambda_pref", help='weight for preference loss',
                        default=1.0, type=float)
    parser.add_argument("--lambda_div", help='weight for diversity regularization between multiple pref. heads',
                        default=1.0, type=float)
    parser.add_argument("--lambda_cons", help='weight for consistency regularization between classifier and preference head',
                        default=1.0, type=float)

    parser.add_argument("--sampling", help='inconsistency, disagreement',
                        default=None, type=str)
    parser.add_argument("--pair_loss", help='averaged ensemble loss',
                        action='store_true')
    parser.add_argument("--consistency", help='averaged ensemble loss',
                        action='store_true')

    return parser.parse_args()

