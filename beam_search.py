import torch
from torch import nn

import tqdm

from typing import Callable, Generator

HUGE = 1e15

def beam_search_top_k(model_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                beam_size: int,
                max_len: int,
                eos_id: int,
                bos_id: int,
                dataloader: Generator[torch.Tensor, None, None]):

    return_outputs = []
    return_logprobs = []
    for batch in dataloader:
        # Setup for this batch (target image)
        device = batch.get_device()

        # Create the outputs setup
        beam_outputs = torch.full((1, 1, 1), bos_id, dtype=torch.long).to(device)

        # Create the inputs setup
        #beam_inputs = torch.tensor([bos_id], dtype=torch.long).to(device)
        beam_inputs = torch.full((1, ), bos_id, dtype=torch.long).to(device)
        
        # input for the hidden layer of the GRU
        beam_hiddens = batch
        # This is the probability for each beam
        # beam_logprobs = torch.tensor([[0.]], device=device)
        beam_logprobs = torch.zeros(1, 1).to(device)
        # This is to keep track what beam has finished (generated EOS tag in their actions)
        # finish_mask = torch.tensor([[0.]], device=device)
        finish_mask = torch.zeros(1, 1).to(device)

        # print("Startoff beam_outputs:", beam_outputs)
        # print("Startoff finish_mask:", finish_mask)
        # print("Startoff beam_logprobs:", beam_logprobs)
        # print("Startoff beam_inputs:", beam_inputs)

        # Generate the actions (captions, with a max length)
        for i in range(max_len):
            
            # print(f"Step {i + 1}/{max_len}")

            # This model func, is the step function in coco_speaker.py
            # The logits are to calculate the next word for the beam (most likely)
            # beams_hiddens_ represents the current hiddenstate of the model (GRU)
            log_probs, beam_hiddens_ = model_func(beam_hiddens, beam_inputs)

            # The vocabulary is the size of the final layer
            vocabulary = log_probs.size()[-1]
            print(vocabulary)

            # Expand beam_log_probs with the vocabulary dimension
            # This way it will have the same size as log_probs
            beam_logprobs = beam_logprobs.unsqueeze(2).repeat(1, 1, vocabulary)

            # Make sure that log_probs and beam_logprobs are the same shape for future operations
            log_probs = log_probs.view(beam_logprobs.size())

            # So the finish mask, keeps track of what captions have generated the end of sequence tag (id 2)
            # It then gives them a very low probability so it does not continue with these sentences

            # Exapand the finish mask to match the output size (vocabulary)
            finish_mask = finish_mask.unsqueeze(2).repeat(1, 1, vocabulary)

            # Mask out all finished sequence, by giving their log_probs a very negative value to exclude them
            log_probs = log_probs * (1 - finish_mask) - HUGE * finish_mask
            
            # Zero out the log_probs of the EOS (end-of-sequence) token for beams that have already finished,
            # ensuring that the EOS token does not affect the selection process for finished sequences
            # So essentially for all finished sequences it just adds another eos_id
            log_probs[:, :, eos_id] = log_probs[:, :, eos_id] * (1 - finish_mask[:, :, 0])

            # Add the previous log_probs to the new log_probs so the chance of the entire sequence is calculated
            beam_logprobs = (beam_logprobs + log_probs).view(1, -1)

            # print("Before Sampling beam_logprobs shape:", beam_logprobs.shape)

            # Grab the top k samples (This is the pruning, it only continues with the top beam_size amount of beams rather than all (which is alot))
            beam_logprobs, indices = torch.topk(beam_logprobs, beam_size)

            # print("After Sampling beam_logprobs shape:", beam_logprobs.shape)

            # identify what beam each candiate belongs to (candidates are the picked beams to stay)
            beam_indices = indices // vocabulary
            # Identify what word each candidate belongs to, so this actually gets the meaning of the picked candidates
            word_indices = indices % vocabulary
            # Update the input of the next beam iteration, so grab for each kept sequence the final word
            # Since a GRU is being used, it can not use the ENTIRE sequence but only the last word of the sequence to calculate the next word
            beam_inputs = word_indices.view(-1)
            # Calculate the finish mask, if a EOS token is the candidate word, then simply update the finish mask to let it know that sequence is finished
            finish_mask = (word_indices == eos_id).float()

            # Get the outputs of the selected candidate beams
            beam_outputs = torch.gather(beam_outputs, 1, beam_indices.unsqueeze(2).repeat(1, 1, i+1))
            # Concatenate the new (candidate) word to the chosen beams
            beam_outputs = torch.cat([beam_outputs, word_indices.unsqueeze(2)], dim=2)

            # Get the size of the last dimension in beams_hidden which represents the hidden size of the model (GRU)
            hid_size = beam_hiddens_.size()[-1]
            # Update the beams_hidden with the chosen candidates
            beam_hiddens = torch.gather(beam_hiddens_.view(1, -1, hid_size),1,beam_indices.unsqueeze(2).repeat(1, 1, hid_size)).view(-1, hid_size)

            # print("Current beam_outputs:", beam_outputs[2])
            # print("Current finish_mask:", finish_mask[2])
            # print("Current beam_logprobs:", beam_logprobs[2])
            # print("Current beam_inputs:", beam_inputs[2])

        # Add the output of this batch, so that at the end of this function all target images their samples can be returned together
        return_outputs.append(beam_outputs)
        return_logprobs.append(beam_logprobs)

    return_outputs = torch.cat(return_outputs, dim=0)
    return_logprobs = torch.cat(return_logprobs, dim=0)

    return (return_outputs, return_logprobs)

def beam_search_top_k_old(model_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                beam_size: int,
                max_len: int,
                eos_id: int,
                bos_id: int,
                dataloader: Generator[torch.Tensor, None, None]):

    return_outputs = []
    return_logprobs = []
    for batch in dataloader:
        # Setup for this batch (target image)
        device = batch.get_device()
        batch_size = batch.size()[0]

        # Create the outputs setup
        beam_outputs = torch.full((batch_size, 1, 1), bos_id, dtype=torch.long).to(device)

        # Create the inputs setup
        #beam_inputs = torch.tensor([bos_id], dtype=torch.long).to(device)
        beam_inputs = torch.full((batch_size, ), bos_id, dtype=torch.long).to(device)
        
        # input for the hidden layer of the GRU
        beam_hiddens = batch
        # This is the probability for each beam
        # beam_logprobs = torch.tensor([[0.]], device=device)
        beam_logprobs = torch.zeros(batch_size, 1).to(device)
        # This is to keep track what beam has finished (generated EOS tag in their actions)
        # finish_mask = torch.tensor([[0.]], device=device)
        finish_mask = torch.zeros(batch_size, 1).to(device)

        # print("Startoff beam_outputs:", beam_outputs)
        # print("Startoff finish_mask:", finish_mask)
        # print("Startoff beam_logprobs:", beam_logprobs)
        # print("Startoff beam_inputs:", beam_inputs)
        print("Beams hidden at the very start is of shape: ", beam_hiddens.shape)
        # Generate the actions (captions, with a max length)
        for i in range(max_len):
            
            print(f"Step {i + 1}/{max_len}")

            # This model func, is the step function in coco_speaker.py
            # The logits are to calculate the next word for the beam (most likely)
            # beams_hiddens_ represents the current hiddenstate of the model (GRU)
            log_probs, beam_hiddens_ = model_func(beam_hiddens, beam_inputs)

            # The vocabulary is the size of the final layer
            vocabulary = log_probs.size()[-1]

            # Expand beam_log_probs with the vocabulary dimension
            # This way it will have the same size as log_probs
            beam_logprobs = beam_logprobs.unsqueeze(2).repeat(1, 1, vocabulary)

            # Make sure that log_probs and beam_logprobs are the same shape for future operations
            log_probs = log_probs.view(beam_logprobs.size())

            # So the finish mask, keeps track of what captions have generated the end of sequence tag (id 2)
            # It then gives them a very low probability so it does not continue with these sentences

            # Exapand the finish mask to match the output size (vocabulary)
            finish_mask = finish_mask.unsqueeze(2).repeat(1, 1, vocabulary)

            # Mask out all finished sequence, by giving their log_probs a very negative value to exclude them
            log_probs = log_probs * (1 - finish_mask) - HUGE * finish_mask
            
            # Zero out the log_probs of the EOS (end-of-sequence) token for beams that have already finished,
            # ensuring that the EOS token does not affect the selection process for finished sequences
            # So essentially for all finished sequences it just adds another eos_id
            log_probs[:, :, eos_id] = log_probs[:, :, eos_id] * (1 - finish_mask[:, :, 0])

            # Add the previous log_probs to the new log_probs so the chance of the entire sequence is calculated
            beam_logprobs = (beam_logprobs + log_probs).view(batch_size, -1)

            print("Before grabbing top k beam_logprobs shape:", beam_logprobs.shape)

            # Grab the top k samples (This is the pruning, it only continues with the top beam_size amount of beams rather than all (which is alot))
            beam_logprobs, indices = torch.topk(beam_logprobs, beam_size)

            print("After grabbing top k beam_logprobs shape:", beam_logprobs.shape)

            # identify what beam each candiate belongs to (candidates are the picked beams to stay)
            beam_indices = indices // vocabulary
            # Identify what word each candidate belongs to, so this actually gets the meaning of the picked candidates
            word_indices = indices % vocabulary
            # Update the input of the next beam iteration, so grab for each kept sequence the final word
            # Since a GRU is being used, it can not use the ENTIRE sequence but only the last word of the sequence to calculate the next word
            beam_inputs = word_indices.view(-1)
            # Calculate the finish mask, if a EOS token is the candidate word, then simply update the finish mask to let it know that sequence is finished
            finish_mask = (word_indices == eos_id).float()

            # Get the outputs of the selected candidate beams
            beam_outputs = torch.gather(beam_outputs, 1, beam_indices.unsqueeze(2).repeat(1, 1, i+1))
            # Concatenate the new (candidate) word to the chosen beams
            beam_outputs = torch.cat([beam_outputs, word_indices.unsqueeze(2)], dim=2)

            # Get the size of the last dimension in beams_hidden which represents the hidden size of the model (GRU)
            hid_size = beam_hiddens_.size()[-1]
            print("Before updating beams_hidden with the chosen candidates of this iteratin it has shape: ", beam_hiddens.shape)
            # Update the beams_hidden with the chosen candidates
            beam_hiddens = torch.gather(beam_hiddens_.view(batch_size, -1, hid_size),1,beam_indices.unsqueeze(2).repeat(1, 1, hid_size)).view(-1, hid_size)
            print("The shape of beams hidden after the entire iteration (new hidden layer input): ", beam_hiddens.shape)
            print("The shape of beams inputs after the entire iteration (new input (actions)): ", beam_inputs.shape)
            # print("Current beam_outputs:", beam_outputs)
            # print("Current finish_mask:", finish_mask)
            # print("Current beam_logprobs:", beam_logprobs)
            # print("Current beam_inputs:", beam_inputs)

        # Add the output of this batch, so that at the end of this function all target images their samples can be returned together
        return_outputs.append(beam_outputs)
        return_logprobs.append(beam_logprobs)

    return_outputs = torch.cat(return_outputs, dim=0)
    return_logprobs = torch.cat(return_logprobs, dim=0)

    return (return_outputs, return_logprobs)


class LanguageModel(nn.Module):
    def __init__(self, vocabulary, hidden_size):
        super(LanguageModel, self).__init__()
        self.word_emb = nn.Embedding(vocabulary, hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.output = nn.Sequential(nn.Linear(hidden_size, vocabulary),
                                    nn.LogSoftmax(dim=-1))

    def forward(self, hidden, inputs):
        hid = self.gru(self.word_emb(inputs), hidden)
        return self.output(hid), hid


def generating_random_data(batch_size: int, hidden_size: int, size: int):
    for i in tqdm.tqdm(range(size)):
        yield torch.randn(batch_size, hidden_size).cuda()


if __name__ == "__main__":
    language_model = LanguageModel(200, 128).cuda()
    i2w = torch.load("i2w")
    with torch.no_grad():
        speaker_actions_original, speaker_logp_original = beam_search_top_k(language_model, 10, 15, 3, 2,
                    generating_random_data(1, 128, 1))
        
        print(speaker_actions_original.shape)
        for i in range(speaker_actions_original.size(0)):
            print("Starting the samples for image: ", i)
            sentences_tensor = speaker_actions_original[i]
            generated_sentences = [
                ' '.join([i2w[token_id.item()] for token_id in sentence])
                for sentence in sentences_tensor
            ]
            for temp_sentence in generated_sentences:
                print(temp_sentence)
            print("\n\n")