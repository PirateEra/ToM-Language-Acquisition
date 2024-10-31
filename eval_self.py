# flake8: noqa: E128
from asyncore import write


if True:
    import argparse
    import os
    import random
    import time
    from distutils.util import strtobool
    import spacy
    nlp = spacy.load("en_core_web_sm")

    import gym
    import wandb
    import numpy as np
    import transformers
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions.categorical import Categorical
    from torch.utils.tensorboard import SummaryWriter
    from referential_game_env import ReferentialGameEnv
    from speaker import Speaker
    from tom_speaker import TOMSpeaker
    from coco_speaker import COCOSpeaker
    from metrics.metrics import Fluency, SemanticSimilarity, sentence_length, num_nouns
    from metrics.analysis import pos_count, get_overlap
    from metrics.compute_bleu import compute_bleu

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="ReferentialGame-v0",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=0.0,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=10000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default="ToM-Language-Acquisition-Eval",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument('--captions-file', type=str, default="data/test_org",
        help="file to get auxiliary captions from")
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--less-logging', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='logs every 1000 timesteps instead of every timestep (recommended for performance)')

    # Algorithm specific arguments
    parser.add_argument('--num-envs', type=int, default=4,
        help='the number of parallel game environments')
    parser.add_argument('--num-steps', type=int, default=128,
        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--exp-decay', type=float, default=0.994)
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Use GAE for advantage computation')
    parser.add_argument('--gamma', type=float, default=1.0,
        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='the lambda for the general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=4,
        help='the number of mini-batches')
    parser.add_argument('--update-epochs', type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggles advantages normalization")
    parser.add_argument('--clip-coef', type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--ent-coef', type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='the maximum norm for the gradient clipping')
    parser.add_argument('--target-kl', type=float, default=None,
        help='the target KL divergence threshold')

    parser.add_argument('--supervised-coef', type=float, default=0.01, help='the ratio of supervised loss')
    parser.add_argument('--length-pen', type=float, default=0.0, help='length penalty')

    # tom arguments
    parser.add_argument('--use-coco', type=lambda x:bool(strtobool(x)), default = False, nargs='?', 
        const = True, help = 'toggle usage of COCOSpeaker')
    parser.add_argument('--use-tom', type=lambda x:bool(strtobool(x)), default = False, nargs='?', 
        const = True, help = 'toggle usage of theory of mind')
    parser.add_argument('--sigma', type=float, default = 0.0, help = "exploration sigma value for ToM speaker")
    parser.add_argument('--tom-weight', type=float, default=1.0, 
        help = "If using a ToM speaker, what weight to give to ToM listener ranking")
    parser.add_argument('--tom-losscoef', type=float, default=0.1, help = "coef for tom loss")
    parser.add_argument('--separate-training', type=lambda x:bool(strtobool(x)), default = False, nargs='?',
        const = True, help = "Separate ToM Listener training from rest of network")
    parser.add_argument('--beam-size', type=int, default=25,
        help = "number of candidates to generate for ToM listener")
    parser.add_argument('--beam-search', type=lambda x:bool(strtobool(x)), default = False, nargs = '?',
        const = True, help = 'use beam search instead of sampling')
    parser.add_argument('--tom-anneal', type=lambda x:bool(strtobool(x)), default = False, nargs='?',
        const = True, help = 'toggle anneal of ToM listener influence')
    parser.add_argument('--tom-anneal-start', type=float, default=0.2, 
        help = "fraction of updates that must pass to start using ToM listener")
    parser.add_argument('--sigma-decay', type=lambda x:bool(strtobool(x)), default = False, nargs='?',
        const = True, help = 'toggle anneal of ToM listener influence')
    parser.add_argument('--sigma-decay-end', type=float, default=1.0, 
        help = "fraction of updates that must pass to converge to final sigma value")
    parser.add_argument('--sigma-low', type=float, default=0.1, 
        help = "final sigma value to converge to")
    parser.add_argument('--gold-standard', type=lambda x:bool(strtobool(x)), default = False, nargs='?',
        const = True, help = 'give ToM speaker access to gold standard ToM listener')
    
    # Environment specific arguments
    parser.add_argument('--vocabulary-size', type=int, 
        default=200,
        help='vocabulary size of speaker')
    parser.add_argument('--max-len', type=int,
        default=20,
        help='maximum utterance length')
    parser.add_argument('--game-file-path', type=str)

    parser.add_argument('--theta-1', type=float, default=.4, help='theta 1')
    parser.add_argument('--theta-2', type=float, default=.9, help='theta 2')
    parser.add_argument('--model-path', type=str, default=None, help='the path of the model')
    parser.add_argument('--n-distr', type=int, default=2)
    parser.add_argument('--distribution', type=str, default='uniform', help='uniform or zipf')

    parser.add_argument('--sup-coef-decay', action='store_true', help='decay supervised coeff')
    parser.add_argument('--D_img', type=int, default=2048,)
    parser.add_argument('--pretrained-path', type=str, default=None,
        help='load in the wandb path for a pretrained model if you want to run in evaluation mode')

    parser.add_argument('--render-html', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help="whether to save HTML images")
    parser.add_argument('--run-name', type=str, default="test",
        help="run name to save HTML files under")
    parser.add_argument('--render-every-N', type=int, default=5000,
        help="render an HTML file every N updates")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    print("Starting!")
    args = parse_args()
    print("Parsed the provided arguments!")
    ################################################################################
    # Setup Experiment and Logger                                                  #
    ################################################################################
    if True:
        run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        if args.track:
            print("Initialising wanDB since tracking is on")
            import wandb
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=args.exp_name,
                monitor_gym=True,
                save_code=True,
            )
    ################################################################################
    # Seeding                                                                      #
    ################################################################################
    if True:
        print("Seeding numpy and torch")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
    ################################################################################
    # Device                                                                       #
    ################################################################################
    if True:
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    ################################################################################
    # Referential Game Environments                                                #
    ################################################################################
    print("Creating the referential games!")
    envs = ReferentialGameEnv(max_len=args.max_len,
                 eos_id=3,
                 noop_penalty=0.5,
                 length_penalty=args.length_pen,
                 batch_size=4,
                 n_distr=args.n_distr,
                 game_file_path=args.game_file_path,
                 theta_1=args.theta_1,
                 theta_2=args.theta_2,
                 distribution=args.distribution,
                 model_path = args.model_path,
                 captions_file = args.captions_file)
    dev_envs = ReferentialGameEnv(max_len=args.max_len,
                eos_id=3,
                noop_penalty=0.5,
                length_penalty=args.length_pen,
                batch_size=4,
                n_distr=args.n_distr,
                game_file_path=args.game_file_path,
                theta_1=args.theta_1,
                theta_2=args.theta_2,
                distribution=args.distribution,
                model_path = args.model_path,
                captions_file = args.captions_file)
    i2w = torch.load("i2w")
    print("Succesfully created the referential games, now loading the speaker and listeners models!")
    # Load necessary components such as the agent and tokenizer
    speaker_path = "wandb/" + args.pretrained_path + "/files/speaker_model.pt"
    listener_path = "wandb/" + args.pretrained_path + "/files/tom_listener.pt"
    tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
    agent = TOMSpeaker(maxlen=args.max_len, vocabsize=tokenizer.vocab_size, 
                       sigma=args.sigma, beam_size=args.beam_size, tom_weight=args.tom_weight,
                       use_pretrained=args.gold_standard, beam_search=args.beam_search,
                       loaded_model_paths=(speaker_path, listener_path), use_coco=True, word_list=list(range(200))).to(device)
    total_games = 10
    accuracy = []
    for i in range(1, total_games+1):
        with torch.no_grad():
            print("Starting a new reference game, game number:", i)
            obs = envs.reset()
            print("Setting up the images to sample the speaker model")
            B = obs["images"].shape[0]
            next_images = torch.Tensor(
                    obs["images"][range(B), :]
            ).to(device)

            next_target = torch.Tensor(obs["goal"]).long().to(device)

            next_images_original = torch.Tensor(
                obs["images"][range(B), obs["goal"]]
            ).to(device)

            print("Sampling the speaker model!")
            sentences_tensor, logprob, _, value = agent.sample(next_images, next_target)

            generated_sentences = [
                ' '.join([i2w[token_id.item()] for token_id in sentence])
                for sentence in sentences_tensor
            ]

            obs, reward = envs.step(sentences_tensor.cpu().numpy())
            dev_accuracy = obs["accuracy"]
            accuracy.append(dev_accuracy)
    print("The accuracy of a total of", total_games, " games is:", (sum(accuracy) / len(accuracy)))

    print("Done!")

