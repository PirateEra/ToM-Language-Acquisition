from asyncore import write
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
import time

def write_evaluation_to_file(accuracy, entropy, unigram_count, bigram_count, trigram_count, gamecount, beamsearch, diverse, g, diverseness, beamsize, maxlength, seconds, absolute_unigrams, absolute_bigrams, absolute_trigrams):
    if not beamsearch:
        content = f"""
        Naive sampling with {beamsize} samples of max length {maxlength}
        Analysis of {gamecount} Games:

        Accuracy       : {accuracy}
        Entropy        : {entropy}
        Normalized Unigram Count  : {unigram_count}
        Normalized Bigram Count   : {bigram_count}
        Normalized Trigram Count  : {trigram_count}
        Absolute Unigram Count  : {absolute_unigrams}
        Absolute Bigram Count   : {absolute_bigrams}
        Absolute Trigram Count  : {absolute_trigrams}
        Runtime in seconds: {seconds}
        \n\n
        """
    elif not diverse and beamsearch:
        content = f"""
        Beamsearch sampling of max length {maxlength} with beamsize: {beamsize}

        Accuracy       : {accuracy}
        Entropy        : {entropy}
        Normalized Unigram Count  : {unigram_count}
        Normalized Bigram Count   : {bigram_count}
        Normalized Trigram Count  : {trigram_count}
        Absolute Unigram Count  : {absolute_unigrams}
        Absolute Bigram Count   : {absolute_bigrams}
        Absolute Trigram Count  : {absolute_trigrams}
        Runtime in seconds: {seconds}
        \n\n
        """
    elif diverse and beamsearch:
        content = f"""
        Diverse beam search sampling of max length {maxlength} with beamsize {beamsize}
        G: {g}
        diverseness: {diverseness}

        Analysis of {gamecount} Games:

        Accuracy       : {accuracy}
        Entropy        : {entropy}
        Normalized Unigram Count  : {unigram_count}
        Normalized Bigram Count   : {bigram_count}
        Normalized Trigram Count  : {trigram_count}
        Absolute Unigram Count  : {absolute_unigrams}
        Absolute Bigram Count   : {absolute_bigrams}
        Absolute Trigram Count  : {absolute_trigrams}
        Runtime in seconds: {seconds}
        \n\n
        """
    with open("evaluation_with_time_dev_easy_absolutes.txt", 'a') as file:
        file.write(content)

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename("evaluation"),
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

    args = parser.parse_args([])
    # fmt: on
    return args

#
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

def evaluate(args=None, beamsearch=False, diverse=False, max_len=10, diverse_g=5, diverse_multiplier=0.1, beam_size=10):
    # The arguments you normally set within the command provided in the readme
    args.total_timesteps = 10000
    args.supervised_coef = 0.01
    args.game_file_path = "data/game_file_dev.pt"
    args.exp_name = "test1"
    args.captions_file = "data/test_org"
    args.less_logging = True
    args.use_coco = True
    args.beam_size = beam_size # This is the amount of samples to draw for each target
    args.prune_size = args.beam_size  # The amount to prune the beams
    args.beam_search = beamsearch
    # Diverse beam search related
    args.diverse = diverse
    args.diverse_G = diverse_g # The amount of groups
    args.diverse_multiplier = diverse_multiplier # The amount the diversity should be taken into account, higher = , lower = more diversity
    #
    args.sigma = 0.0
    args.seed = 517
    args.tom_weight = 1000.0
    args.pretrained_path = "andy_files"
    args.batch_size = 1
    #
    args.max_len = max_len # Default is 20
    args.render = False
    args.total_games = 100
    args.hard = False
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # Games
    envs = ReferentialGameEnv(max_len=args.max_len,
                eos_id=3,
                noop_penalty=0.5,
                length_penalty=args.length_pen,
                batch_size=args.batch_size,
                n_distr=args.n_distr,
                game_file_path=args.game_file_path,
                theta_1=args.theta_1,
                theta_2=args.theta_2,
                distribution=args.distribution,
                model_path = args.model_path,
                captions_file = args.captions_file,
                hard=args.hard)
    dev_envs = ReferentialGameEnv(max_len=args.max_len,
                eos_id=3,
                noop_penalty=0.5,
                length_penalty=args.length_pen,
                batch_size=args.batch_size,
                n_distr=args.n_distr,
                game_file_path=args.game_file_path,
                theta_1=args.theta_1,
                theta_2=args.theta_2,
                distribution=args.distribution,
                model_path = args.model_path,
                captions_file = args.captions_file,
                hard=args.hard)
    # i2w = torch.load("i2w")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # Load necessary components such as the agent and tokenizer
    speaker_path = "wandb/" + args.pretrained_path + "/files/speaker_model.pt"
    listener_path = "wandb/" + args.pretrained_path + "/files/tom_listener.pt"
    tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
    agent = TOMSpeaker(maxlen=args.max_len, vocabsize=tokenizer.vocab_size, 
                        sigma=args.sigma, beam_size=args.beam_size, tom_weight=args.tom_weight,
                        use_pretrained=args.gold_standard, beam_search=args.beam_search,
                        loaded_model_paths=(speaker_path, listener_path), use_coco=args.use_coco, word_list=list(range(200))).to(device)
    accuracy = []
    entropies = []
    unigrams = []
    bigrams = []
    trigrams = []
    absolute_unigrams = []
    absolute_bigrams = []
    absolute_trigrams = []
    obs = envs.reset() # Not needed because .step also updates to do a new game
    print("New game!")
    start_time = time.time()
    for i in range(1, args.total_games+1):
        with torch.no_grad():
            print("Game: ", i)
            # This is to prepare the images for the models
            B = obs["images"].shape[0]
            next_images = torch.Tensor(
                    obs["images"][range(B), :]
            ).to(device)

            # This gets the next target in the reference game
            next_target = torch.Tensor(obs["goal"]).long().to(device)

            # This creates the sentences, so this makes the Speaker sample sentences based on the images and the target
            sentences_tensor, logp, entropy, temp_unigrams, temp_bigrams, temp_trigrams = agent.sample(next_images, next_target, beam_size=args.beam_size, prune_size=args.prune_size, diverse=args.diverse, diverse_G=args.diverse_G, diverse_multiplier=args.diverse_multiplier, render=args.render) # This sentences tensor contains 1 sentence per "game"

            # The env step function makes the listener pick a target based on the given sentences
            # Let hierbij op, dat als je render = true op false zet, je geen plaatjes meer zult zien. dit is dus opzich sneller voor alleen accuracy zien
            obs = envs.step(sentences_tensor.cpu().numpy(), render=args.render)
            dev_accuracy = obs["accuracy"]

            # This simply appends the accuracy of this game to the accuracy list of all games
            accuracy.append(dev_accuracy)
            entropies.append(entropy)
            unigrams.append(temp_unigrams[1])
            bigrams.append(temp_bigrams[1])
            trigrams.append(temp_trigrams[1])

            absolute_unigrams.append(temp_unigrams[0])
            absolute_bigrams.append(temp_bigrams[0])
            absolute_trigrams.append(temp_trigrams[0])
            if args.render:
                print("\n\n\n")
    
    print("Beam search: ", args.beam_search)
    print("Diverse: ", args.diverse)
    print("The accuracy of a total of", args.total_games, "games is:", (sum(accuracy) / len(accuracy)))
    print("The entropy of a total of", args.total_games, "games is:", (sum(entropies) / len(entropies)))
    print("The unigram count of a total of", args.total_games, "games is:", (sum(unigrams) / len(unigrams)))
    print("The bigram count of a total of", args.total_games, "games is:", (sum(bigrams) / len(bigrams)))
    print("The trigram count of a total of", args.total_games, "games is:", (sum(trigrams) / len(trigrams)))
    accuracy = (sum(accuracy) / len(accuracy))
    entropy = (sum(entropies) / len(entropies))
    unigrams = (sum(unigrams) / len(unigrams))
    bigrams = (sum(bigrams) / len(bigrams))
    trigrams = (sum(trigrams) / len(trigrams))
    absolute_unigrams = sum(absolute_unigrams)
    absolute_bigrams = sum(absolute_bigrams)
    absolute_trigrams = sum(absolute_trigrams)
    end_time = time.time()
    seconds = end_time - start_time
    write_evaluation_to_file(accuracy, entropy, unigrams, bigrams, trigrams, args.total_games, args.beam_search, args.diverse, args.diverse_G, args.diverse_multiplier, args.beam_size, args.max_len, seconds, absolute_unigrams, absolute_bigrams, absolute_trigrams)

    print("Done!")

if __name__ == "__main__":
    print("Starting evaluation")
    args = parse_args()
    evaluate(args=args, beamsearch=False, max_len=10, beam_size=5) # Naive
    evaluate(args=args, beamsearch=False, max_len=10, beam_size=10) # Naive
    evaluate(args=args, beamsearch=False, max_len=10, beam_size=25) # Naive
    # evaluate(args=args, beamsearch=False, max_len=10, beam_size=50) # Naive
    # evaluate(args=args, beamsearch=False, max_len=10, beam_size=100) # Naive
    # # Beam search
    evaluate(args=args, beamsearch=True, diverse=False, max_len=10, beam_size=5) # Beam search
    evaluate(args=args, beamsearch=True, diverse=False, max_len=10, beam_size=10) # Beam search
    evaluate(args=args, beamsearch=True, diverse=False, max_len=10, beam_size=25) # Beam search
    # evaluate(args=args, beamsearch=True, diverse=False, max_len=10, beam_size=50) # Beam search
    # evaluate(args=args, beamsearch=True, diverse=False, max_len=10, beam_size=100) # Beam search
    # Diverse Beam search
    
    # Diverse multiplier 0.2
    
    # G == B
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.2, beam_size=5) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=10, diverse_multiplier=0.2, beam_size=10) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=25, diverse_multiplier=0.2, beam_size=25) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=50, diverse_multiplier=0.2, beam_size=50) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=100, diverse_multiplier=0.2, beam_size=100) # Diverse beam search
    # G == 2
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=2, diverse_multiplier=0.2, beam_size=6) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=2, diverse_multiplier=0.2, beam_size=10) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=2, diverse_multiplier=0.2, beam_size=26) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=2, diverse_multiplier=0.2, beam_size=50) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=2, diverse_multiplier=0.2, beam_size=100) # Diverse beam search
    # G == 5
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.2, beam_size=5) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.2, beam_size=10) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.2, beam_size=25) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.2, beam_size=50) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.2, beam_size=100) # Diverse beam search
    #
    # Diverse multiplier 0.8
    #
    # G == B
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.8, beam_size=5) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=10, diverse_multiplier=0.8, beam_size=10) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=25, diverse_multiplier=0.8, beam_size=25) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=50, diverse_multiplier=0.8, beam_size=50) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=100, diverse_multiplier=0.8, beam_size=100) # Diverse beam search
    # G == 2
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=2, diverse_multiplier=0.8, beam_size=6) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=2, diverse_multiplier=0.8, beam_size=10) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=2, diverse_multiplier=0.8, beam_size=26) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=2, diverse_multiplier=0.8, beam_size=50) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=2, diverse_multiplier=0.8, beam_size=100) # Diverse beam search
    # G == 5
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.8, beam_size=5) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.8, beam_size=10) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.8, beam_size=25) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.8, beam_size=50) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.8, beam_size=100) # Diverse beam search
    #
    # Diverse multiplier 0.01 # Super diverse
    #
    # G == B
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.01, beam_size=5) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=10, diverse_multiplier=0.01, beam_size=10) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=25, diverse_multiplier=0.01, beam_size=25) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=50, diverse_multiplier=0.01, beam_size=50) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=100, diverse_multiplier=0.01, beam_size=100) # Diverse beam search
    # G == 2
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=2, diverse_multiplier=0.01, beam_size=6) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=2, diverse_multiplier=0.01, beam_size=10) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=2, diverse_multiplier=0.01, beam_size=26) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=2, diverse_multiplier=0.01, beam_size=50) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=2, diverse_multiplier=0.01, beam_size=100) # Diverse beam search
    # G == 5
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.01, beam_size=5) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.01, beam_size=10) # Diverse beam search
    evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.01, beam_size=25) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.01, beam_size=50) # Diverse beam search
    # evaluate(args=args, beamsearch=True, diverse=True, max_len=10, diverse_g=5, diverse_multiplier=0.01, beam_size=100) # Diverse beam search
    #
    print("Done with all!")
