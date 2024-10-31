from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
import torchvision.models as models
from PIL import Image
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from beam_search import beam_search_top_k
from nucleus_sampling import nucleus_sampling


def ppo_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize a layer with an orthogonal matrix and a bias vector. This is
    for PPO
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class COCOSpeaker(nn.Module):
    class _block(nn.Module):
        def __init__(self, hid):
            super().__init__()
            self.hid = hid
            self.dense1 = nn.Sequential(
                nn.Linear(hid, hid),
                nn.GELU(),
                nn.Linear(hid, hid),
            )
            self.dense2 = nn.Sequential(
                nn.Linear(hid, hid),
                nn.GELU(),
                nn.Linear(hid, hid),
            )

        def forward(self, x):
            y = nn.functional.layer_norm(x, [self.hid])
            y = self.dense1(y)
            y = nn.functional.layer_norm(y + x, [self.hid])
            return x + self.dense2(y)

    def __init__(
        self, *, vocabulary_size: int, max_len: int, D_img: int, word_list: List[int]
    ):
        super(COCOSpeaker, self).__init__()

        self.feature_resizer = nn.Linear(2304, D_img)

        self.max_len = max_len
        # image_encoder = nn.Sequential(
        # *list(models.resnet18(pretrained=True).children())[:-1]
        # )
        # Dry run to get the output size
        # out = image_encoder(torch.zeros(1, 3, 224, 224))
        # out_size = out.view(-1).size()[0]
        out_size = D_img
        # Add feature resizer the decoder
        self.encoder = nn.Sequential(
            COCOSpeaker._block(out_size),
            nn.LayerNorm(out_size),
            nn.Linear(out_size, 512),
        )
        self.word_emb = nn.Embedding(vocabulary_size, 512)
        self.decoder = nn.GRUCell(
            input_size=512,
            hidden_size=512,
        )
        self.critic = nn.Sequential(
            ppo_layer_init(nn.Linear(512, 64)),
            nn.Tanh(),
            ppo_layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            ppo_layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            ppo_layer_init(nn.Linear(512, 64)),
            nn.Tanh(),
            ppo_layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            ppo_layer_init(nn.Linear(64, vocabulary_size), std=0.01),
        )
        self.word_mask = torch.ones(vocabulary_size) * (-1000)
        for i in word_list:
            self.word_mask[i] = 0
        self.word_mask = nn.Parameter(self.word_mask, requires_grad=False)
        self.huge = 1e15


    def get_action_and_value(
        self, images: torch.FloatTensor, actions: Optional[torch.LongTensor] = None, ) -> Tuple[
        torch.LongTensor,  # actions (batch_size, max_len)
        torch.FloatTensor,  # logprobs (batch_size, max_len)
        torch.FloatTensor,  # entropy (batch_size, max_len)
        torch.FloatTensor,  # values (batch_size, max_len)
    ]:
        B = images.size(0)
        image_representation = self.encoder(images)
        ix = torch.zeros_like(image_representation)
        action_list = []
        logprobs = []
        entropies = []
        values = []
        hx = image_representation
        for i in range(self.max_len):

            hx = self.decoder(hx, ix)
            logits = self.actor(hx)
            logits += self.word_mask
            logits = torch.log_softmax(logits, dim=-1)  # normalize

            # No idea what this does, but for the caption sampling (that we use), it always goes into the else statement
            # So actions is always None, this other statement is probably related to the reinforcement learning
            action = torch.multinomial(torch.exp(logits), 1).squeeze(-1)

            logprob = torch.gather(logits, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)


            entropy = -torch.sum(torch.softmax(logits, dim=-1) * logits, dim=-1)
            value = self.critic(hx)
            ix = self.word_emb(action)

            action_list.append(action)
            logprobs.append(logprob)
            entropies.append(entropy)
            values.append(value)

        action_list = torch.stack(action_list, dim=1)
        logprobs = torch.stack(logprobs, dim=1)
        entropies = torch.stack(entropies, dim=1)
        values = torch.stack(values, dim=1)
        return action_list, logprobs, entropies, values


    def supervised_loss(
        self,
        images: torch.FloatTensor,
        actions: torch.LongTensor,
        # bboxes: torch.FloatTensor,
        mask: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Compute the loss for the supervised training of the model.
        """
        _, logprobs, _, _ = self.get_action_and_value(images=images, actions = actions)
        # _, logprobs, _, _ = self(images, bboxes, actions)
        logprobs = torch.where(logprobs < -1000, torch.zeros_like(logprobs), logprobs)
        # logprobs[actions==1] = 0
        return -(logprobs.sum(-1) * mask.float()).mean()


    def get_image_representation(
        self, image_feature: torch.FloatTensor, box_feature: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Forward pass for the COCO Speaker model.
        """
        return self.feature_resizer(torch.cat([image_feature, box_feature], dim=1))


    def forward(
        self,
        image_feature: torch.FloatTensor,
        box_feature: torch.FloatTensor,
        actions: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.LongTensor,  # actions (batch_size, max_len)
        torch.FloatTensor,  # logprobs (batch_size, max_len)
        torch.FloatTensor,  # entropy (batch_size, max_len)
        torch.FloatTensor,  # values (batch_size, max_len)
    ]:
        return self.get_action_and_value(
            images=self.get_image_representation(image_feature, box_feature), actions=actions,
        )

    def step(self, hid, input):

        ix = self.word_emb(input)

        # if this is the first action (beginning of sentence)
        # if input[0] == 2: # RobertaTokenizer.bos_token_id
        #     ix = torch.zeros_like(hid)
        
        hx = self.decoder(hid, ix)

        # This should be fine
        logits = self.actor(hx)
        logits += self.word_mask
        log_probs = torch.log_softmax(logits, dim=-1) #normalize the logits, keep in mind this turns them into log probabilties
        return log_probs, hx

    def sample_multiple(
            self,
            images: torch.FloatTensor,
            actions: Optional[torch.LongTensor] = None,
            beam_size: int = 5,
            batch_size: int = 4
        ) -> Tuple[
            torch.LongTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor]:
            """
            Sample multiple images from the speaker's distribution.
            """
            if actions is None:
                images = images.repeat(beam_size,1)
            actions, logprobs, entropy, values = self.get_action_and_value(images=images, actions=actions,)
            return(torch.reshape(actions, (beam_size, -1, self.max_len)), 
            torch.reshape(logprobs, (beam_size, -1, self.max_len)),
            torch.reshape(entropy, (beam_size, -1, self.max_len)),
            torch.reshape(values, (beam_size, -1, self.max_len)))

    def beam_sample(
            self,
            images: torch.FloatTensor,
            actions: Optional[torch.LongTensor] = None,
            beam_size: int = 10,
            diverse = False,
            batch_size: int = 1,
            prune_size: int = 10,
            G:int = 2,
            diverseness = 0.5,
        ) -> Tuple[
            torch.LongTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor]:
            """
            Sample multiple images from the speaker's distribution.
            """
            if actions is None:
                images = images.repeat(beam_size,1)
            if diverse:
                actions, logprobs, entropy = self.get_action_and_probs_diverse_beam_sampling(images=images, actions=actions, beam_size=beam_size, prune_size=prune_size, G=G, diverseness=diverseness)
            else:
                actions, logprobs, entropy = self.get_action_and_probs_beam_sampling(images=images, actions=actions, beam_size=beam_size, prune_size=prune_size)
            # print("The final actions have shape", actions.shape)
            # print("The final probs have shape", logprobs.shape)
            # print(torch.reshape(actions, (beam_size, -1, self.max_len)).shape)
            return actions, logprobs, entropy

    def get_action_and_probs_beam_sampling(
        self, images: torch.FloatTensor, actions: Optional[torch.LongTensor] = None, BOSid=2, EOSid=3, beam_size=10, prune_size=10) -> Tuple[
        torch.LongTensor,  # actions (batch_size, max_len)
        torch.FloatTensor,  # logprobs (batch_size, max_len)
        torch.FloatTensor,  # entropy (batch_size, max_len)
        torch.FloatTensor,  # values (batch_size, max_len)
    ]:
        # B = images.size(0) # This works because im using batch size 1, so doing this will equal the beam size instead
        device = images.get_device()
        image_representation = self.encoder(images)
        # So this is to feed the GRU with the first layer, which is just a bunch of zeros which will result in BOS id's when fed
        ix = torch.zeros_like(image_representation)
        hx = image_representation
        #######################################################
        # INITIAL START (BOS AND THE FIRST B AMOUNT OF WORDS)
        ########################################################
        # Start off each sentence with the BOS and setup the GRU stuff for the beam search (see it as a do while)
        # The first iteration is for BOS and the second to get the actual first B amount of words (the first iteration is not used)
        # But i do do it to update hx to make sure its properly setup (i got no clue how its trained)
        hx, logits = self.beamsearch_update_logits(hx, ix) # Get the logits (which will result into BOS, since its the first iteration)
        _, actions = torch.topk(torch.exp(logits), 1) # Should technically all be BOS, which is the BOS id for the first iteration
        #action = action.squeeze(-1)    # So originally its like a tensor [[2], [2], [2]] and it is turned into [2, 2, 2, 2,] so from [B, 1] to [B] shape wise
        logprob = torch.gather(logits, dim=-1, index=actions).squeeze(-1) # Get the logprobs that belong to the actions (should be shape B, so a tensor of B values)
        beamactions = actions # should be of shape B,1 so for every sequence the first action which is BOS
        beamprobs = logprob
        ix = self.word_emb(actions.squeeze(-1)) # Update IX (embed the current words, all BOS)

        # SECOND ITERATION, which is obtaining the top B best words to start off with, this is the starting setup for beam search
        hx, logits = self.beamsearch_update_logits(hx, ix) # Get the logits (which will result into B of the same rows, since all rows are based on BOS)
        _, actions = torch.topk(torch.exp(logits[0]), beam_size) # This is important, only do top K for the first row (since all the rows are the same), this way u get 10 different start words
        logprob = torch.gather(logits, dim=-1, index=actions.unsqueeze(1)).squeeze(-1) # Get the logprobs that belong to the actions (should be shape B, so a tensor of B values)
        beamactions = torch.cat((beamactions, actions.unsqueeze(1)), dim=1) # Add the new words to the beam actions which is resulting in the first column being BOSid and the 2nd column being non-identical ids, unsqueeze to turn B shape into B,1
        beamprobs = torch.add(logprob, beamprobs) # Sum the probabilities of the BOS and the new words to get the full probability of the beam unsqueeze to add go from B to B,1 to match logprob
        ix = self.word_emb(actions.squeeze(-1)) # Update IX (embed the current actions (just added words))
        #######################################################
        # ACTUAL BEAM SEARCH
        ########################################################
        completed_sentences = []
        entropies = []
        # i2w = torch.load("i2w") # Only needed when debugging
        for i in range(self.max_len):

            # Obtain the logits
            hx, logits = self.beamsearch_update_logits(hx, ix)

            # Calculate entropy
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * logits, dim=-1)

            # Expand each sequence with the possible (beam size) options (this is pretty much pruning the vocabulary before actually selecting the top K beams)
            logprob, actions = torch.topk(torch.exp(logits), prune_size) # Shape of this should be [B, B], which would be the top B actions for each sequence in B
            logprob = torch.gather(logits, dim=-1, index=actions).squeeze(-1) # Grab the logprobs of each of the actions that were decided upon in the line above
            
            # Get the probabilities for each possible beam in the iteration, which is a total of B*B sequences
            all_beamprobs = logprob + beamprobs.unsqueeze(1) # so this is again a tensor of shape B,B but with the total probabilities

            # Use topK to automatically order what beams are most likely (from this the top B will be grabbed excluding finished sentences)
            # The view(-1) pretty much flattens the all_beamprobs, so that it can easily grab the topk out of a 2d tensor
            # all_beamprobs.view(-1).shape[0] makes sure every single value is returned from the topk which pretty much causes the full ordering
            unfiltered_top_values, top_indices = torch.topk(all_beamprobs.view(-1), all_beamprobs.view(-1).shape[0])

            # Convert the top_indices to their respective 2d coordinates, so they are actually useable
            # So the whole // and & is simply to turn a value like 8 in a 10x10 tensor to (0,7) or a value like 12 into (1,1)
            unfiltered_top_indices_2d = [(idx.item() // all_beamprobs.shape[1], idx.item() % all_beamprobs.shape[1]) for idx in top_indices]
            # These are the top actions of this iteration
            counter = 0
            top_actions = []
            top_indices_2d = []
            top_values = []
            finished_indices = []
            finished_values = []
            # Loop through the top_indices, until B are found (excluding finished sequences)
            # I also save the finished sequences that were found to include them at the end, since they were above the top 10 unfinished
            for i, index in enumerate(unfiltered_top_indices_2d):
                if counter >= beam_size: # Exit this for loop if B actions were found
                    break
                temp_action = actions[index[0], index[1]] # Get the action
                if temp_action != EOSid: # If it is a unfinished sequence, we keep it in its respective lists
                    counter += 1
                    top_indices_2d.append(index)
                    top_actions.append(temp_action)
                    top_values.append(unfiltered_top_values[i])
                else: # store this finished sequence since it was above the top 10 unfinished, so it'll be good to reconsider at the end
                    finished_indices.append(index)
                    finished_values.append(unfiltered_top_values[i])
            
            top_actions = torch.tensor(top_actions).to(device)# Turn top actions into a tensor
            top_values = torch.tensor(top_values).to(device)# Turn top values into a tensor

            # Deal with the finished sentences by storing them
            if len(finished_indices) > 0:
                finished_sequences = torch.stack([beamactions[idx[0]] for idx in finished_indices]) # Get the finished sequences, by row
                finished_sequences = torch.cat((finished_sequences, torch.full((finished_sequences.shape[0], 1), EOSid).to(device)), dim=1) # Add a column of EOS id's to the finished sequences to finish them
                finished_rows = [x[0] for x in finished_indices] # These will be used to get the entropy row later
                completed_sentences.append((finished_sequences, finished_values, finished_rows)) # Store the finished sequences and values as a tuple for later reference
    
            # Create the sequences from the top actions and the top indices
            new_beams = torch.stack([beamactions[idx[0]] for idx in top_indices_2d]) # This keeps the beams from the previous iteration (without the new actions), stack simply turns a list of B tensors into a tensor of B tensors
            # print("The shape of the new beams (without the new actions)", new_beams.shape)
            # Add the new actions to the beams (the unsqueeze is so they can be concatenated, its a shape 10 thats turned into a shape 10,1)
            new_beams = torch.cat((new_beams, top_actions.unsqueeze(1)), dim=1)

            # This embeds each action (word) so it can be fed into the GRU, essentially it  turns the action into a vector of 512 (layer size)
            ix = self.word_emb(top_actions)

            # Deal with the entropies
            entropy = torch.stack([entropy[idx[0]] for idx in top_indices_2d])
            entropies.append(entropy)
            # Turn beamprobs into the new beams their logprobs, so this now contains the total probability for each kept beam
            beamprobs = top_values
            beamactions = new_beams

            # Debugging
            # generated_sentences = [
            #     ' '.join([i2w[token_id.item()] for token_id in sentence])
            #     for sentence in beamactions
            # ]
            # for temp_sentence in generated_sentences:
            #     print(temp_sentence)
            # print("\n\n")
        
        entropies = torch.stack(entropies, dim=1) # Turn entropies into a tensor rather than a list of column tensors
        # Normalize the beam probabilities, using length normalization
        # This could potentially mess with the reraking, im not sure if it needs the original log probs or not (but im guessing not)
        beamprobs = beamprobs / beamactions.shape[1]
        # Re evaluate, by grabbing the top K sequences, based on the completed and leftover sequences
        max_size = beamactions.size(1) # Get the size the 2nd dimension has to be
        padded_finished_sequences = []
        for tensor in completed_sentences:
            probabilities = torch.stack(tensor[1]).to(device) # Get the probabilities
            entropy_rows = tensor[2] # Get a list of indices that give the entropy rows
            entropy_rows = torch.stack([entropies[idx] for idx in entropy_rows]) # Get entropy of each sequence by row
            tensor = tensor[0] # Since its a tuple, grab the actual tensor (be aware im resetting the tensor variable here)
            sequence_length = tensor.shape[1] # This will be needed to normalize the score
            probabilities = probabilities / sequence_length # Normalize the probabilities, using length normalization
            padding = (0, max_size - tensor.size(1))  # get the amount of padding needed for this tensor
            padded_tensor = torch.nn.functional.pad(tensor, padding, "constant", 0) # Add the padding (0 is padding id)
            padded_finished_sequences.append(padded_tensor) # Add the padded sequence tensor to work with it
            beamprobs = torch.cat((beamprobs, probabilities)) # Add the probabilities to the sequences to later get the top k
            entropies = torch.cat((entropies, entropy_rows), dim=0) # Add the entropies
        
        if len(padded_finished_sequences) > 0:
            padded_finished_sequences = torch.cat(padded_finished_sequences, dim=0) # Add all the padded finishes sequences together
            beamactions = torch.cat((padded_finished_sequences, beamactions), dim=0) # Add the finished sequences to the other sequences

        # Grab the top K from all the gathered sequences
        beamprobs, indices = torch.topk(beamprobs, beam_size) # This will contain the top 10 logprobs in beamprobs and the top 10 indices of the top 10 sequences
        beamactions = beamactions[indices] # Extract the needed sequences from beamactions
        entropies = entropies[indices] # Extract the entropies of the final sequences

                
        return beamactions, beamprobs, entropies
    
    def beamsearch_update_logits(self, hx, ix):
        hx = self.decoder(hx, ix)
        logits = self.actor(hx)
        logits += self.word_mask
        logits = torch.log_softmax(logits, dim=-1)  # normalize, which implies turn it into probabilities shape is [B, 50265]
        return hx, logits

    def get_action_and_probs_diverse_beam_sampling_old(
        self, images: torch.FloatTensor, actions: Optional[torch.LongTensor] = None, BOSid=2, EOSid=3, G=2) -> Tuple[
        torch.LongTensor,  # actions (batch_size, max_len)
        torch.FloatTensor,  # logprobs (batch_size, max_len)
        torch.FloatTensor,  # entropy (batch_size, max_len)
        torch.FloatTensor,  # values (batch_size, max_len)
    ]:
        B = images.size(0) # This works because im using batch size 1, so doing this will equal the beam size instead
        device = images.get_device()
        image_representation = self.encoder(images)
        # So this is to feed the GRU with the first layer, which is just a bunch of zeros which will result in BOS id's when fed
        ix = torch.zeros_like(image_representation)
        hx = image_representation
        # Diverse related variables
        g = B // G # This is the amount that each group will have, since torch.split is based on the amount per tensor
        #######################################################
        # INITIAL START (BOS AND THE FIRST B AMOUNT OF WORDS)
        ########################################################
        # Start off each sentence with the BOS and setup the GRU stuff for the beam search (see it as a do while)
        # The first iteration is for BOS and the second to get the actual first B amount of words (the first iteration is not used)
        # But i do do it to update hx to make sure its properly setup (i got no clue how its trained)
        hx, logits = self.beamsearch_update_logits(hx, ix) # Get the logits (which will result into BOS, since its the first iteration)
        _, actions = torch.topk(torch.exp(logits), 1) # Should technically all be BOS, which is the BOS id for the first iteration
        #action = action.squeeze(-1)    # So originally its like a tensor [[2], [2], [2]] and it is turned into [2, 2, 2, 2,] so from [B, 1] to [B] shape wise
        logprob = torch.gather(logits, dim=-1, index=actions).squeeze(-1) # Get the logprobs that belong to the actions (should be shape B, so a tensor of B values)
        beamactions = actions # should be of shape B,1 so for every sequence the first action which is BOS
        beamprobs = logprob
        ix = self.word_emb(actions.squeeze(-1)) # Update IX (embed the current words, all BOS)

        # SECOND ITERATION, which is obtaining the top 10 best words to start off with, this is the starting setup for beam search
        hx, logits = self.beamsearch_update_logits(hx, ix) # Get the logits (which will result into B of the same rows, since all rows are based on BOS)
        _, actions = torch.topk(torch.exp(logits[0]), 10) # This is important, only do top K for the first row (since all the rows are the same), this way u get 10 different start words
        logprob = torch.gather(logits, dim=-1, index=actions.unsqueeze(1)).squeeze(-1) # Get the logprobs that belong to the actions (should be shape B, so a tensor of B values)
        beamactions = torch.cat((beamactions, actions.unsqueeze(1)), dim=1) # Add the new words to the beam actions which is resulting in the first column being BOSid and the 2nd column being non-identical ids, unsqueeze to turn B shape into B,1
        beamprobs = torch.add(logprob, beamprobs) # Sum the probabilities of the BOS and the new words to get the full probability of the beam unsqueeze to add go from B to B,1 to match logprob
        ix = self.word_emb(actions.squeeze(-1)) # Update IX (embed the current actions (just added words))
        #######################################################
        # ACTUAL BEAM SEARCH
        #######################################################
        for i in range(self.max_len):
            print(f"Step {i + 1}/{self.max_len}")

            # Obtain the logits
            hx, logits = self.beamsearch_update_logits(hx, ix)
            # hx = self.decoder(hx, ix)
            # logits = self.actor(hx)
            # logits += self.word_mask
            # logits = torch.log_softmax(logits, dim=-1)  # normalize, which implies turn it into probabilities shape is [B, 50265]

            # print("Current logits shape:", logits.shape)

            # Expand each sequence with the possible (beam size) options
            logprob, actions = torch.topk(torch.exp(logits), g) # Shape of this should be [B, g], which would be the top g actions for each sequence in B
            logprob = torch.gather(logits, dim=-1, index=actions).squeeze(-1) # Grab the logprobs of each of the actions that were decided upon in the line above
            
            # Get the probabilities for each possible beam in the iteration, which is a total of B*B sequences
            all_beamprobs = logprob + beamprobs.unsqueeze(1) # so this is again a tensor of shape B,B but with the total probabilities
            print("all_beamprobs shape is", all_beamprobs.shape)

            # Divide the beams into subgroups (because sampling will happen in subgroups to provide diversity)
            subprobs = torch.split(all_beamprobs, g)
            subactions = torch.split(actions, g)
            subbeams = torch.split(beamactions, g)
            # Loop through each group and get the top K of that group
            # This is a bit tricky since if the top K amount is 10 in total, and there are 2 groups of 5 each it implies that it needs to get the top 5 of both groups to match the total of top 10
            top_actions = []
            beamprobs = []
            beamactions = []
            for i, group in enumerate(subprobs):
                print(f"Group number {i} has shape {group.shape}")
                # Get the top g probabilities and the indices related to them, keep in mind these are flat indices so not row column but a number between 0 and g*g
                group_top_values, group_top_indices = torch.topk(group.view(-1), g)
                # Convert the top_indices to their respective 2d coordinates, so they are actually useable
                # So the whole // and & is simply to turn a value like 8 in a 10x10 tensor to (0,7) or a value like 12 into (1,1)
                top_indices_2d = [(idx.item() // g, idx.item() % g) for idx in group_top_indices]
                #print("The top indices are: ", top_indices_2d)
                # These are the top actions of this iteration
                group_top_actions = torch.tensor([subactions[i][idx[0], idx[1]].item() for idx in top_indices_2d]).to(device)
                #print("The top (current) actions are", top_actions.shape, top_actions)
                # Create the sequences from the top actions and the top indices
                group_new_beams = torch.stack([subbeams[idx[0]] for idx in top_indices_2d]) # This keeps the beams from the previous iteration (without the new actions), stack simply turns a list of B tensors into a tensor of B tensors
                # print("The shape of the new beams (without the new actions)", new_beams.shape)
                # Add the new actions to the beams (the unsqueeze is so they can be concatenated, its a shape 10 thats turned into a shape 10,1)
                group_new_beams = torch.cat((group_new_beams, group_top_actions.unsqueeze(1)), dim=1)
                # Add the top sequences from this group to the beamactions
                beamactions.append(group_new_beams)
                top_actions.append(group_top_actions)
                beamprobs.append(group_top_values)
                print(f"From this group a total of {group_top_actions.shape} actions were taken")
                print(f"From this group a total of {group_new_beams.shape} beams were taken")
                

            # Combine all the groups back into one tensor to continue beam search like normal, for example its first a list of 5 tensors each 2,21 and it now turns it into 10,21
            beamactions = torch.cat(beamactions)
            top_actions = torch.cat(top_actions)
            beamprobs = torch.cat(beamprobs)
            print("beamactions shape is", beamactions.shape)
            print("top_actions shape is", top_actions.shape)
            print("beamprobs shape is", beamprobs.shape)

            # print("Current beam shape:", new_beams.shape)
            # print("This is what the word embed thing does", self.word_emb(top_actions).shape)
            # This embeds each action (word) so it can be fed into the GRU, essentially it  turns the action into a vector of 512
            ix = self.word_emb(top_actions)
            
        return beamactions, beamprobs

    def get_action_and_probs_diverse_beam_sampling_wrong(
        self, images: torch.FloatTensor, actions: Optional[torch.LongTensor] = None, BOSid=2, EOSid=3, G=2) -> Tuple[
        torch.LongTensor,  # actions (batch_size, max_len)
        torch.FloatTensor,  # logprobs (batch_size, max_len)
        torch.FloatTensor,  # entropy (batch_size, max_len)
        torch.FloatTensor,  # values (batch_size, max_len)
    ]:
        B = images.size(0) # This works because im using batch size 1, so doing this will equal the beam size instead
        device = images.get_device()
        image_representation = self.encoder(images)
        # So this is to feed the GRU with the first layer, which is just a bunch of zeros which will result in BOS id's when fed
        ix = torch.zeros_like(image_representation)
        hx = image_representation
        # Diverse related variables
        g = B // G # This is the amount that each group will have, since torch.split is based on the amount per tensor
        #######################################################
        # INITIAL START (BOS AND THE FIRST B AMOUNT OF WORDS)
        ########################################################
        # Start off each sentence with the BOS and setup the GRU stuff for the beam search (see it as a do while)
        # The first iteration is for BOS and the second to get the actual first B amount of words (the first iteration is not used)
        # But i do do it to update hx to make sure its properly setup (i got no clue how its trained)
        hx, logits = self.beamsearch_update_logits(hx, ix) # Get the logits (which will result into BOS, since its the first iteration)
        _, actions = torch.topk(torch.exp(logits), 1) # Should technically all be BOS, which is the BOS id for the first iteration
        #action = action.squeeze(-1)    # So originally its like a tensor [[2], [2], [2]] and it is turned into [2, 2, 2, 2,] so from [B, 1] to [B] shape wise
        logprob = torch.gather(logits, dim=-1, index=actions).squeeze(-1) # Get the logprobs that belong to the actions (should be shape B, so a tensor of B values)
        beamactions = actions # should be of shape B,1 so for every sequence the first action which is BOS
        beamprobs = logprob
        ix = self.word_emb(actions.squeeze(-1)) # Update IX (embed the current words, all BOS)

        # SECOND ITERATION, which is obtaining the top 10 best words to start off with, this is the starting setup for beam search
        hx, logits = self.beamsearch_update_logits(hx, ix) # Get the logits (which will result into B of the same rows, since all rows are based on BOS)
        _, actions = torch.topk(torch.exp(logits[0]), 10) # This is important, only do top K for the first row (since all the rows are the same), this way u get 10 different start words
        logprob = torch.gather(logits, dim=-1, index=actions.unsqueeze(1)).squeeze(-1) # Get the logprobs that belong to the actions (should be shape B, so a tensor of B values)
        beamactions = torch.cat((beamactions, actions.unsqueeze(1)), dim=1) # Add the new words to the beam actions which is resulting in the first column being BOSid and the 2nd column being non-identical ids, unsqueeze to turn B shape into B,1
        beamprobs = torch.add(logprob, beamprobs) # Sum the probabilities of the BOS and the new words to get the full probability of the beam unsqueeze to add go from B to B,1 to match logprob
        ix = self.word_emb(actions.squeeze(-1)) # Update IX (embed the current actions (just added words))
        #######################################################
        # ACTUAL BEAM SEARCH
        ########################################################
        completed_sentences = []
        i2w = torch.load("i2w") # Only needed when debugging
        for i in range(self.max_len):
            print(f"Step {i + 1}/{self.max_len}")

            # Obtain the logits
            hx, logits = self.beamsearch_update_logits(hx, ix)

            # Expand each sequence with the possible (beam size) options (this is pretty much pruning the vocabulary before actually selecting the top K beams)
            logprob, actions = torch.topk(torch.exp(logits), B) # Shape of this should be [B, B], which would be the top B actions for each sequence in B
            logprob = torch.gather(logits, dim=-1, index=actions).squeeze(-1) # Grab the logprobs of each of the actions that were decided upon in the line above
            
            # Get the probabilities for each possible beam in the iteration, which is a total of B*B sequences
            all_beamprobs = logprob + beamprobs.unsqueeze(1) # so this is again a tensor of shape B,B but with the total probabilities
            #
            # Divide the beams into subgroups (because sampling will happen in subgroups to provide diversity)
            subprobs = torch.split(all_beamprobs, g)
            subactions = torch.split(actions, g)
            subbeams = torch.split(beamactions, g)
            final_beams = [] # in here i will store the tensor of sequences for each group
            final_actions = [] # in here i will store the tensor of actions for each group
            final_probs = [] # in here i will store the probabiltiies of each beam for each group
            # Do beam search per group
            for i, group in enumerate(subprobs):
                print(f"Group {i + 1}/{len(subprobs)}")
                group_actions = subactions[i]
                groupbeams = subbeams[i]
                # Use topK to automatically order what beams are most likely (from this the top B will be grabbed excluding finished sentences)
                # The view(-1) pretty much flattens the all_beamprobs, so that it can easily grab the topk out of a 2d tensor
                # all_beamprobs.view(-1).shape[0] makes sure every single value is returned from the topk which pretty much causes the full ordering
                unfiltered_top_values, top_indices = torch.topk(group.view(-1), group.view(-1).shape[0])
                # Convert the top_indices to their respective 2d coordinates, so they are actually useable
                # So the whole // and & is simply to turn a value like 8 in a 10x10 tensor to (0,7) or a value like 12 into (1,1)
                unfiltered_top_indices_2d = [(idx.item() // group.shape[1], idx.item() % group.shape[1]) for idx in top_indices]

                # These are the top actions of this iteration
                counter = 0
                top_actions = []
                top_indices_2d = []
                top_values = []
                finished_indices = []
                finished_values = []
                # Loop through the top_indices, until 10 are found (excluding finished sequences)
                # I also save the finished sequences that were found to include them at the end, since they were above the top 10 unfinished
                for i, index in enumerate(unfiltered_top_indices_2d):
                    if counter >= g: # Exit this for loop if 10 actions were found
                        break
                    temp_action = group_actions[index[0], index[1]] # Get the action
                    if temp_action != EOSid: # If it is a unfinished sequence, we keep it in its respective lists
                        counter += 1
                        top_indices_2d.append(index)
                        top_actions.append(temp_action)
                        top_values.append(unfiltered_top_values[i])
                    else: # store this finished sequence since it was above the top 10 unfinished, so it'll be good to reconsider at the end
                        finished_indices.append(index)
                        finished_values.append(unfiltered_top_values[i])
                
                top_actions = torch.tensor(top_actions).to(device)# Turn top actions into a tensor
                top_values = torch.tensor(top_values).to(device)# Turn top values into a tensor

                # Deal with the finished sentences by storing them
                if len(finished_indices) > 0:
                    finished_sequences = torch.stack([beamactions[idx[0]] for idx in finished_indices]) # Get the finished sequences
                    finished_sequences = torch.cat((finished_sequences, torch.full((finished_sequences.shape[0], 1), EOSid).to(device)), dim=1) # Add a column of EOS id's to the finished sequences to finish them
                    completed_sentences.append((finished_sequences, finished_values)) # Store the finished sequences and values as a tuple for later reference
                print("The top indices are: ", top_indices_2d)


                print("The top (current) actions are", top_actions.shape, top_actions)
                # Create the sequences from the top actions and the top indices
                new_beams = torch.stack([groupbeams[idx[0]] for idx in top_indices_2d]) # This keeps the beams from the previous iteration (without the new actions), stack simply turns a list of B tensors into a tensor of B tensors
                # print("The shape of the new beams (without the new actions)", new_beams.shape)
                # Add the new actions to the beams (the unsqueeze is so they can be concatenated, its a shape 10 thats turned into a shape 10,1)
                new_beams = torch.cat((new_beams, top_actions.unsqueeze(1)), dim=1)
                final_beams.append(new_beams)
                final_probs.append(top_values)
                final_actions.append(top_actions)

            # Since the final_lists are lists of tensors for each groups chosen values, they have to be combined into 1 tensor
            top_actions = torch.cat(final_actions)
            # Turn beamprobs into the new beams their logprobs, so this now contains the total probability for each kept beam
            beamprobs = torch.cat(final_probs)
            beamactions = torch.cat(final_beams)
            # This embeds each action (word) so it can be fed into the GRU, essentially it  turns the action into a vector of 512 (layer size)
            ix = self.word_emb(top_actions)

            # Debugging
            # generated_sentences = [
            #     ' '.join([i2w[token_id.item()] for token_id in sentence])
            #     for sentence in beamactions
            # ]
            # for temp_sentence in generated_sentences:
            #     print(temp_sentence)
            # print("\n\n")
        
        # Normalize the beam probabilities, using length normalization
        # This could potentially mess with the reraking, im not sure if it needs the original log probs or not (but im guessing not)
        beamprobs = beamprobs / beamactions.shape[1]
        # Re evaluate, by grabbing the top K sequences, based on the completed and leftover sequences
        max_size = beamactions.size(1) # Get the size the 2nd dimension has to be
        padded_finished_sequences = []
        for tensor in completed_sentences:
            probabilities = torch.stack(tensor[1]).to(device) # Get the probabilities
            tensor = tensor[0] # Since its a tuple, grab the actual tensor (be aware im resetting the tensor variable here)
            sequence_length = tensor.shape[1] # This will be needed to normalize the score
            probabilities = probabilities / sequence_length # Normalize the probabilities, using length normalization
            padding = (0, max_size - tensor.size(1))  # get the amount of padding needed for this tensor
            padded_tensor = torch.nn.functional.pad(tensor, padding, "constant", 0) # Add the padding (0 is padding id)
            padded_finished_sequences.append(padded_tensor) # Add the padded sequence tensor to work with it
            beamprobs = torch.cat((beamprobs, probabilities)) # Add the probabilities to the sequences to later get the top k
            
        padded_finished_sequences = torch.cat(padded_finished_sequences, dim=0) # Add all the padded finishes sequences together
        beamactions = torch.cat((padded_finished_sequences, beamactions), dim=0) # Add the finished sequences to the other sequences

        # Grab the top K from all the gathered sequences
        beamprobs, indices = torch.topk(beamprobs, B) # This will contain the top 10 logprobs in beamprobs and the top 10 indices of the top 10 sequences
        beamactions = beamactions[indices] # Extract the needed sequences from beamactions

                
        return beamactions, beamprobs

    def get_action_and_probs_diverse_beam_sampling(
        self, images: torch.FloatTensor, actions: Optional[torch.LongTensor] = None, BOSid=2, EOSid=3, G=5, beam_size=10, prune_size=10, diverseness = 0.5) -> Tuple[
        torch.LongTensor,  # actions (batch_size, max_len)
        torch.FloatTensor,  # logprobs (batch_size, max_len)
        torch.FloatTensor,  # entropy (batch_size, max_len)
        torch.FloatTensor,  # values (batch_size, max_len)
    ]:
        # B = images.size(0) # This works because im using batch size 1, so doing this will equal the beam size instead
        diverseness = np.log(diverseness) # Turn the diverseness into a logarithm to add it to the probabilities when needed
        device = images.get_device()
        image_representation = self.encoder(images)
        # So this is to feed the GRU with the first layer, which is just a bunch of zeros which will result in BOS id's when fed
        ix = torch.zeros_like(image_representation)
        hx = image_representation
        # Diverse related variables
        # G stands for the amount of groups, and g for the amount of beams per group
        g = beam_size // G # This is the amount that each group will have, since torch.split is based on the amount per tensor
        #######################################################
        # INITIAL START (BOS AND THE FIRST B AMOUNT OF WORDS)
        #######################################################
        # Start off each sentence with the BOS and setup the GRU stuff for the beam search (see it as a do while)
        # The first iteration is for BOS and the second to get the actual first B amount of words (the first iteration is not used)
        # But i do do it to update hx to make sure its properly setup (i got no clue how its trained)
        hx, logits = self.beamsearch_update_logits(hx, ix) # Get the logits (which will result into BOS, since its the first iteration)
        _, actions = torch.topk(torch.exp(logits), 1) # Should technically all be BOS, which is the BOS id for the first iteration
        #action = action.squeeze(-1)    # So originally its like a tensor [[2], [2], [2]] and it is turned into [2, 2, 2, 2,] so from [B, 1] to [B] shape wise
        logprob = torch.gather(logits, dim=-1, index=actions).squeeze(-1) # Get the logprobs that belong to the actions (should be shape B, so a tensor of B values)
        beamactions = actions # should be of shape B,1 so for every sequence the first action which is BOS
        beamprobs = logprob
        ix = self.word_emb(actions.squeeze(-1)) # Update IX (embed the current words, all BOS)

        # SECOND ITERATION, which is obtaining the top 10 best words to start off with, this is the starting setup for beam search
        hx, logits = self.beamsearch_update_logits(hx, ix) # Get the logits (which will result into B of the same rows, since all rows are based on BOS)
        _, actions = torch.topk(torch.exp(logits[0]), beam_size) # This is important, only do top K for the first row (since all the rows are the same), this way u get 10 different start words
        logprob = torch.gather(logits, dim=-1, index=actions.unsqueeze(1)).squeeze(-1) # Get the logprobs that belong to the actions (should be shape B, so a tensor of B values)
        beamactions = torch.cat((beamactions, actions.unsqueeze(1)), dim=1) # Add the new words to the beam actions which is resulting in the first column being BOSid and the 2nd column being non-identical ids, unsqueeze to turn B shape into B,1
        beamprobs = torch.add(logprob, beamprobs) # Sum the probabilities of the BOS and the new words to get the full probability of the beam unsqueeze to add go from B to B,1 to match logprob
        ix = self.word_emb(actions.squeeze(-1)) # Update IX (embed the current actions (just added words))
        #######################################################
        # ACTUAL BEAM SEARCH
        ########################################################
        completed_sentences = [[] for _ in range(G)]
        entropies = []
        # i2w = torch.load("i2w") # Only needed when debugging
        for i in range(self.max_len):
            # print(f"Step {i + 1}/{self.max_len}")
            # Obtain the logits
            hx, logits = self.beamsearch_update_logits(hx, ix)
            group_beams = []
            group_probs = []
            group_actions = []
            group_entropies = []
            # Do beam search per group
            for j in range(G):
                # Calculate entropy for this group
                probs = torch.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * logits, dim=-1)
                # print(f"Group {j + 1}/{G}")
                # Expand each sequence with the possible (beam size) options (this is pretty much pruning the vocabulary before actually selecting the top K beams)
                logprob, actions = torch.topk(torch.exp(logits), prune_size) # Shape of this should be [B, B], which would be the top B actions for each sequence in B
                logprob = torch.gather(logits, dim=-1, index=actions).squeeze(-1) # Grab the logprobs of each of the actions that were decided upon in the line above
                # Get the probabilities for each possible beam in the iteration, which is a total of B*B sequences
                all_beamprobs = logprob + beamprobs.unsqueeze(1) # so this is again a tensor of shape B,B but with the total probabilities

                # Use topK to automatically order what beams are most likely (from this the top B will be grabbed excluding finished sentences)
                # The view(-1) pretty much flattens the all_beamprobs, so that it can easily grab the topk out of a 2d tensor
                # all_beamprobs.view(-1).shape[0] makes sure every single value is returned from the topk which pretty much causes the full ordering
                unfiltered_top_values, top_indices = torch.topk(all_beamprobs.view(-1), all_beamprobs.view(-1).shape[0]) # So keep in mind, this is purely ordering, nothing more
            
                # Convert the top_indices to their respective 2d coordinates, so they are actually useable
                # So the whole // and & is simply to turn a value like 8 in a 10x10 tensor to (0,7) or a value like 12 into (1,1)
                unfiltered_top_indices_2d = [(idx.item() // all_beamprobs.shape[1], idx.item() % all_beamprobs.shape[1]) for idx in top_indices]
                # These are the top actions of this iteration
                counter = 0
                top_actions = []
                top_indices_2d = []
                top_values = []
                finished_indices = []
                finished_values = []
                # Loop through the top_indices, until g are found (excluding finished sequences)
                # I also save the finished sequences that were found to include them at the end, since they were above the top g unfinished
                for i, index in enumerate(unfiltered_top_indices_2d):
                    if counter >= g: # Exit this for loop if g actions were found
                        break
                    temp_action = actions[index[0], index[1]] # Get the action
                    if temp_action != EOSid: # If it is a unfinished sequence, we keep it in its respective lists
                        counter += 1
                        top_indices_2d.append(index)
                        top_actions.append(temp_action)
                        top_values.append(unfiltered_top_values[i])
                    else: # store this finished sequence since it was above the top g unfinished, so it'll be good to reconsider at the end
                        # Keep in mind because these are groups, its possible that the same sentence appears in multiple groups, which can cause issues later down the road
                        finished_indices.append(index)
                        finished_values.append(unfiltered_top_values[i])
                
                top_actions = torch.tensor(top_actions).to(device)# Turn top actions into a tensor
                top_values = torch.tensor(top_values).to(device)# Turn top values into a tensor

                # Deal with the finished sentences by storing them
                if len(finished_indices) > 0:
                    finished_sequences = torch.stack([beamactions[idx[0]] for idx in finished_indices]) # Get the finished sequences
                    finished_sequences = torch.cat((finished_sequences, torch.full((finished_sequences.shape[0], 1), EOSid).to(device)), dim=1) # Add a column of EOS id's to the finished sequences to finish them
                    finished_rows = [x[0] for x in finished_indices] # These will be used to get the entropy row later
                    completed_sentences[j].append((finished_sequences, finished_values, finished_rows)) # Store the finished sequences and values as a tuple for later reference
                # print("The top indices are: ", top_indices_2d)
                
                # print("The top (current) actions are", top_actions.shape, top_actions)
                # Create the sequences from the top actions and the top indices
                new_beams = torch.stack([beamactions[idx[0]] for idx in top_indices_2d]) # This keeps the beams from the previous iteration (without the new actions), stack simply turns a list of B tensors into a tensor of B tensors
                # print("The shape of the new beams (without the new actions)", new_beams.shape)
                # Add the new actions to the beams (the unsqueeze is so they can be concatenated, its a shape 10 thats turned into a shape 10,1)
                new_beams = torch.cat((new_beams, top_actions.unsqueeze(1)), dim=1)
                
                # Deal with the entropies
                entropy = torch.stack([entropy[idx[0]] for idx in top_indices_2d])

                # Add this groups values to the group lists to be combined after the group beam searches have been finished
                group_beams.append(new_beams)
                group_probs.append(top_values)
                group_actions.append(top_actions)
                group_entropies.append(entropy)
                
                # This penalizes the chosen beam(s) of this group, which makes it so other groups do not choose this beam again and are forced to choose other beams
                for index in top_indices_2d:
                    beamprobs[index[0]] = beamprobs[index[0]] + diverseness# -self.huge # Set the entire beam to chance 0, so it wont be picked again, this wont do harm for future iterations since the beamprobs variable is reset for each iteration
                    logits[index[0], :] = logits[index[0], :] + diverseness# -self.huge # Set all logits of the picked beam to chance 0, since it should leave no chances of being picked again

                # Debugging
                # generated_sentences = [
                #     ' '.join([i2w[token_id.item()] for token_id in sentence])
                #     for sentence in new_beams
                # ]
                # for temp_sentence in generated_sentences:
                #     print(temp_sentence)
            
            # Since the final_lists are lists of tensors for each groups chosen values, they have to be combined into 1 tensor
            top_actions = torch.cat(group_actions)
            # Turn beamprobs into the new beams their logprobs, so this now contains the total probability for each kept beam
            beamprobs = torch.cat(group_probs)
            beamactions = torch.cat(group_beams)
            entropies.append(torch.cat(group_entropies))

            # This embeds each action (word) so it can be fed into the GRU, essentially it  turns the action into a vector of 512 (layer size)
            ix = self.word_emb(top_actions)
            # print("\n")

        entropies = torch.stack(entropies, dim=1) # Turn entropies into a tensor rather than a list of column tensors
        # Normalize the beam probabilities, using length normalization
        # This could potentially mess with the reranking, im not sure if it needs the original log probs or not (but im guessing not)
        beamprobs = beamprobs / beamactions.shape[1]
        # Re evaluate, by grabbing the top K sequences, based on the completed and leftover sequences
        max_size = beamactions.size(1) # Get the size the 2nd dimension has to be
        # Re evalution per group, so that in the end the sentences dont get mixed up
        # This is because the probabilities of the groups that are not group 1 are bound to be lower, 
        # which will remove them at the very end if not done in a seperate group
        sub_beam_probs = torch.split(beamprobs, g)
        sub_beams = torch.split(beamactions, g)
        sub_entropy = torch.split(entropies, g)
        final_beams = []
        final_probs = []
        final_entropy = []
        all_finished_sequences = [] # Keep track of all finished sequences, to prevent duplicate finished sequences in groups (if it appeared in group 1, then its in this list and u can filter it out of other groups)
        for i in range(G):
            padded_finished_sequences = []
            padded_probs = []
            finished_entropies = []
            current_probs = sub_beam_probs[i]
            current_beams = sub_beams[i]
            current_entropy = sub_entropy[i]
            if len(completed_sentences[i]) > 0: # If there is completed sentences in this group
                for tuple in completed_sentences[i]:
                    probabilities = torch.stack(tuple[1]).to(device) # Get the probabilities
                    entropy_rows = tuple[2] # Get a list of indices that give the entropy rows
                    entropy_rows = torch.stack([entropies[idx] for idx in entropy_rows]) # Get entropy of each sequence by row
                    finished_entropies.append(entropy_rows)
                    tensor = tuple[0] # Since its a tuple, grab the actual tensor (be aware im resetting the tensor variable here)
                    sequence_length = tensor.shape[1] # This will be needed to normalize the score
                    probabilities = probabilities / sequence_length # Normalize the probabilities, using length normalization
                    padding = (0, max_size - tensor.size(1))  # get the amount of padding needed for this tensor
                    padded_tensor = torch.nn.functional.pad(tensor, padding, "constant", 0) # Add the padding (0 is padding id)
                    padded_finished_sequences.append(padded_tensor) # Add the padded sequence tensor to work with it
                    padded_probs.append(probabilities) # Add the probabilities of the padded sequences
        
                padded_finished_sequences = torch.cat(padded_finished_sequences, dim=0).to(device) # Add all the padded finished sequences together
                padded_probs = torch.cat(padded_probs, dim=0).to(device) # Add all the padded finished sequences together
                finished_entropies = torch.cat(finished_entropies, dim=0).to(device) # Combine all the entropy rows
                # Prevent sequences that appeared in other groups (so filter them out)
                if len(all_finished_sequences) > 0 and len(padded_finished_sequences) > 0:
                    temp_all_finished_sequences = torch.cat(all_finished_sequences, dim=0).to(device) # Since its a list, combine all tensors in that list to one tensor
                    duplicate_rows = self.get_duplicate_tensor_rows(padded_finished_sequences, temp_all_finished_sequences) # Get the rows that contain duplicates
                    if len(duplicate_rows) > 0: # Remove the duplicate rows, if there are any
                        # Remove the rows using a mask (true for rows that stay, false for rows that wont stay)
                        mask = torch.ones(padded_finished_sequences.size(0), dtype=bool).to(device) # True for every row
                        mask[duplicate_rows] = False # Set the rows to remove to false
                        padded_finished_sequences = padded_finished_sequences[mask]  # Remove any finished sequences, that appeared in other groups
                        padded_probs = padded_probs[mask]
                        finished_entropies = finished_entropies[mask]
                if(len(padded_finished_sequences) > 0): # So because duplicate rows could have been removed, there is a chance no sentences were left, so a check if needed
                    current_beams = torch.cat((padded_finished_sequences, current_beams), dim=0) # Add the finished sequences to the other sequences of this group
                    current_probs = torch.cat((padded_probs, current_probs), dim=0)
                    current_entropy = torch.cat((finished_entropies, current_entropy), dim=0)

            # Grab the top K from all the gathered sequences
            current_probs, indices = torch.topk(current_probs, g) # This will contain the top B logprobs in beamprobs and the top B indices of the top B sequences
            current_beams = current_beams[indices] # Extract the needed sequences from beamactions
            current_entropy = current_entropy[indices]
            # Add this groups probs and beams to the final beams/probs
            final_probs.append(current_probs)
            final_beams.append(current_beams)
            final_entropy.append(current_entropy)
            # Add the padded finished sequences of this group, to keep track of them and make sure they dont appear in other groups
            if len(padded_finished_sequences) > 0: # i == 0 because group 1 wont have to deal with duplicates
                 all_finished_sequences.append(padded_finished_sequences)
        
        # Join the beamprobs and beamactions since they are still in groups at the moment
        beamprobs = torch.cat(final_probs)
        beamactions = torch.cat(final_beams)
        entropy = torch.cat(final_entropy)
        return beamactions, beamprobs, entropy

    def get_duplicate_tensor_rows(self, tensor_1, tensor_2):
        duplicate_rows = []
        for i, row in enumerate(tensor_1):
            for row_2 in tensor_2:
                if row.equal(row_2):
                    duplicate_rows.append(i)
        return torch.tensor(list(dict.fromkeys(duplicate_rows))) # Return it without any duplicates