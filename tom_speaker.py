from tkinter import image_names
from typing import Tuple
import torch
from torch import nn
from coco_speaker import COCOSpeaker
from tom_listener import TOMListener
from speaker import Speaker
from listener import Listener
import numpy as np

class TOMSpeaker(nn.Module):
    '''
    Speaker with a ToM reranking module. Initialized from ppo.py.
    '''
    def __init__(self, maxlen, vocabsize, sigma, beam_size, tom_weight, use_coco=False, 
                word_list = None, use_pretrained = False, beam_search = False, loaded_model_paths = None):
        
        super(TOMSpeaker, self).__init__()
        self.beam_size = beam_size
        self.use_coco = use_coco
        if self.use_coco:
            self.speaker = COCOSpeaker(max_len=maxlen, vocabulary_size=vocabsize, D_img=2048, word_list=word_list)
            # self.speaker = torch.jit.script(self.speaker)
        else:
            self.speaker = Speaker(max_len=maxlen, vocabulary_size=vocabsize)
        self.tom_listener = TOMListener(beam_size=beam_size, maxlen=maxlen, use_pretrained=use_pretrained)
        
        # load pretrained speaker if path is given
        if loaded_model_paths is not None:
            self.speaker.load_state_dict(torch.load(loaded_model_paths[0]))
            self.tom_listener.load_state_dict(torch.load(loaded_model_paths[1]))
            self.speaker.eval()
            
        self.tom_weight = tom_weight
        self.sigma = sigma
        self.beam_search = beam_search

    def sample(
        self, images: torch.FloatTensor, target_ids: torch.LongTensor, batch_size = 4,
        actions: torch.LongTensor = None, 
        beam_size = None, 
        prune_size = None, 
        include_pred = False, 
        diverse = False, 
        diverse_G=5, 
        diverse_multiplier=0.1,
        render=True
    ) -> Tuple[
        torch.LongTensor,  # actions (batch_size, max_len)
        torch.FloatTensor,  # logprobs (batch_size, max_len)
        torch.FloatTensor,  # entropy (batch_size, max_len)
        torch.FloatTensor,  # values (batch_size, max_len)
    ]:
        i2w = torch.load("i2w")
        if beam_size is None:
            beam_size = self.beam_size
        if prune_size is None:
            prune_size = beam_size

        # This is the batch size
        B = images.size(0)

        # Grab the target images, which are a total of B (batch) images
        target_images = images[range(B), target_ids]
        if self.beam_search:
            if render:
                if diverse:
                    print("Using diverse beam search")
                else:
                    print("Using beam search")
            # Perform beam search, to get the sentences based on the target images
            # Selfmade
            speaker_actions, speaker_logp, entropy = self.speaker.beam_sample(target_images, None, beam_size=beam_size, prune_size=beam_size, G=diverse_G, diverseness=diverse_multiplier, diverse=diverse) # This returns like size 10, 21 with a beam size of 10 and sequence length of 21. So it only works for batch size 1
            # Reshape the tensors to match the rest of the code (original code) (only relevant for my own version)
            # This turns them from for example 10, 21 to 10, 1 , 21
            speaker_actions = torch.unsqueeze(speaker_actions, dim=1)
            speaker_logp = torch.unsqueeze(speaker_logp, dim=1)

        else:
            if render:
                print("Using naive sampling")
            # Do sample multiple to get the sentences based on the target images
            # Actions was always none here (the parameter, so i altered it to prevent confusion)
            #speaker_actions, speaker_logp, entropy, values = self.speaker.sample_multiple(target_images, actions, beam_size)
            speaker_actions, speaker_logp, entropy, values = self.speaker.sample_multiple(target_images, None, beam_size)

        speaker_actions_debug = speaker_actions.transpose(1, 0)
        unigrams = self.amount_of_ngrams(speaker_actions_debug[0], 1) # Assuming its 1 batch, 0 works
        bigrams = self.amount_of_ngrams(speaker_actions_debug[0], 2) # Assuming its 1 batch, 0 works
        trigrams = self.amount_of_ngrams(speaker_actions_debug[0], 3) # Assuming its 1 batch, 0 works

        # Show the samples in text
        if render:
            for i in range(speaker_actions_debug.size(0)):
                # print("Starting the samples for image: ", i)
                sentences_tensor = speaker_actions_debug[i]
                generated_sentences = [
                    ' '.join([i2w[token_id.item()] for token_id in sentence])
                    for sentence in sentences_tensor
                ]
                #print(sentences_tensor)
                for temp_sentence in generated_sentences:
                    print(temp_sentence)
                print("\n\n")
        # reranking
        # Sum all log probabilities of the third dimension (this is relevant for sample_multiple, but not for beam search)
        if self.beam_search:
            speaker_ranking = speaker_logp
            #speaker_ranking = torch.log_softmax(torch.sum(speaker_logp, 2), dim=1)
        else:
            # This removes the 3rd dimension from the speaker_logp by summing it
            speaker_ranking = torch.log_softmax(torch.sum(speaker_logp, 2), dim=1)
            
        # Create predictions made by the listener for each caption (sentence) regarding each target image
        # I checked and the ranking seems to not perse prefer the highest logprob, so the speaker_ranking does not matter THAT much when doing diverse
        listener_ranking, pred = self.tom_listener._predict(images, target_ids, speaker_actions, beam_size = beam_size, include_pred = include_pred)
        ranking = speaker_ranking + self.tom_weight*listener_ranking

        best_candidate = torch.argmax(ranking, dim=0)
        candidate = best_candidate

        #
        if include_pred:
            pred = pred[candidate, range(B)]
        
        # return action and value
        # Essentially what this does, is grab the B amount of best actions (sentences). So for each target image it grabs the best option
        tom_action = speaker_actions[candidate,range(B),:]

        # Get the entropy of the chosen candidate
        test = torch.mean(entropy, dim=1)
        entropy = entropy[candidate] # Get the candidate row
        row = entropy.squeeze().tolist()  # Squeeze to turn the row into a list (tensor to list)
        # Get the index of where the sentence ended, keep in mind the hardcoded 3 is the EOS id, if it didnt end simply return row size
        index = next((i for i, x in enumerate(row) if x == 3), entropy.shape[1]-1)
        index = index+1 # This is done to make sure the slicing works properly
        entropy = entropy[:index] # Get the entropy row up until the end of the sequence
        # Get the entropy of the candiate
        entropy = torch.mean(entropy).item() # Get the mean of the entropy, since the entropy is of each action at the moment and not of the entire sequence

        # The problem here is that sample_multiple returns a log probability score for each token in the vocabulary.
        # And beam_search only returns the log probability for each sequence, so the slicing is different
        if self.beam_search:
            tom_logp = speaker_logp[candidate,range(B)]
        else:
            tom_logp = speaker_logp[candidate,range(B),:]

        return tom_action, tom_logp, entropy, unigrams, bigrams, trigrams

    def supervised_loss(
        self,
        images: torch.FloatTensor,
        actions: torch.LongTensor,
        target_ids: torch.LongTensor,
        mask: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Compute loss for the supervised training of the model.
        """
        _, logprobs, _, _ = self.sample(images, target_ids, actions = actions, beam_size = 1)
        return -(logprobs.sum(-1) * mask.float()).mean()

    def update_tom_weight(self, new_weight):
        self.tom_weight = new_weight
    
    def update_sigma(self, new_sigma):
        self.sigma = new_sigma
    
    def amount_of_ngrams(self, sequence_tensor, n=1):
        special_ids = [0, 1, 2, 3] # 0 = padding, 1 = unk, 2 = BOS, 3 = EOS
        ngrams_set = set()
        total_words = 0
        # Loop through every sequence of the sequence tensor
        for sequence in sequence_tensor:
            # Filtering
            sequence = sequence.tolist() # I expect a tensor of 1 row
            sequence = [word for word in sequence if word not in special_ids] # Remove padding, BOS, EOS etc
            total_words += len(sequence)

            # Generate the ngrams
            for i in range(len(sequence) - n + 1):
                ngram = tuple(sequence[i:i + n])
                ngrams_set.add(ngram)
        return (len(ngrams_set), len(ngrams_set) / total_words)