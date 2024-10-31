import torch
from torch import nn

import tqdm

from typing import Callable, Generator


def nucleus_sampling(model_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                initial_K: int,
                batch_size: int,
                P: float,
                max_len: int,
                eos_id: int,
                bos_id: int,
                dataloader: Generator[torch.Tensor, None, None]):

    final_actions = torch.zeros(batch_size, initial_K, max_len)
    final_logprobs = torch.zeros(batch_size, initial_K, max_len)
    i = 0
    for image in dataloader:
        print("Starting a new image!")
        # Setup for this (target image)
        device = image.get_device()

        # Create the outputs setup
        nucleus_outputs = torch.full((initial_K, 1), bos_id, dtype=torch.long).to(device)

        # Create the inputs setup
        #nucleus_inputs = torch.tensor([bos_id], dtype=torch.long).to(device)
        nucleus_inputs = torch.full((1, ), bos_id, dtype=torch.long).to(device)
        
        # input for the hidden layer of the GRU
        hx = image
        # This is the probability for each finished sequence
        # nucleus_logprobs = torch.tensor([[0.]], device=device)
        nucleus_logprobs = torch.full((initial_K, 1), 1.0, dtype=torch.long).to(device)

        # a Mask to keep track of what sentences have finished. it starts with 0.0 and when its a 1.0 it means that sentence is finished
        final_mask = torch.full((initial_K, 1), 0.0, dtype=torch.long).to(device)

        # print("Startoff nucleus_outputs:", nucleus_outputs)
        # print("Startoff finish_mask:", finish_mask)
        # print("Startoff nucleus_logprobs:", nucleus_logprobs)
        # print("Startoff nucleus_inputs:", nucleus_inputs)

        # Generate the actions (captions, with a max length)

        # First iteration to generate the K- amount of samples
        log_probs, hx = model_func(hx, nucleus_inputs)
        log_probs, most_recent_nucleus_words = torch.topk(log_probs, initial_K)

        # The  transpose is because topk gives back (1, amount) and we need (amount, 1)
        nucleus_outputs = torch.cat((nucleus_outputs, most_recent_nucleus_words.transpose(0, 1)), dim=-1)
        nucleus_logprobs = torch.cat((nucleus_logprobs, log_probs.transpose(0, 1)), dim=-1)

        # Actual nucleus sampking
        for i in range(max_len):
            
            # print(f"Step {i + 1}/{max_len}")

            # This model func, is the step function in coco_speaker.py
            # The logits are to calculate the next word for the sequence (most likely)
            # nucleus_hiddens_ represents the current hiddenstate of the model (GRU)
            log_probs, hx = model_func(hx, most_recent_nucleus_words)

            # Perform nucleus sampling on the logits, to prune the vocabulary and select the new actions
            # So this returns a tensor of actions and their probabilities, the most_recent_words are the actions chosen for each sequence
            log_probs, most_recent_nucleus_words = nucleus_sample_logits(log_probs, P)

            # A mask to figure out what sentences generated the EOS id for this iteration. because if so, then they are done and need padding
            current_final_mask = (most_recent_nucleus_words == eos_id).float()
            # Update the final mask, to contain the currently finished sentences
            final_mask = ((final_mask.bool() | current_final_mask.bool())).float()
            # alter the most recently picked actions, to have a EOS id if that sequence is finished.
            most_recent_nucleus_words[final_mask == 1] = eos_id

            # Add the newest words to the outputs (this creates the sequence, so this adds the new word to the entire sequence)
            nucleus_outputs = torch.cat((nucleus_outputs, most_recent_nucleus_words), dim=-1)
            # Add the log probabilities, this way we can keep track of them
            nucleus_logprobs = torch.cat((nucleus_logprobs, log_probs), dim=-1)

        # Add the output of this batch, so that at the end of this function all target images their samples can be returned together
        print("the nucleus outputs are: ", nucleus_outputs)
        final_actions[i] = nucleus_outputs
        final_logprobs[i] = nucleus_logprobs
        i += 1
    
    return (final_actions, final_logprobs)

def nucleus_sample_logits(logprobs, p):

    # By taking the exponent the values turn into probabilities, which will allow for deciding when P has been met
    probs = torch.exp(logprobs)
    #print("Shape of probs:", probs.shape)
    #print("Current probs:", probs)
    # Sort the probabilities and corresponding indices in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    #print("Current sorted_probs:", sorted_probs)
    
    # Compute the cumulative probabilities
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    # print("Current cumulative_probs:", cumsum_probs)
    
    # Find the target point where cumulative probability exceeds p
    target_index = (cumsum_probs > p).nonzero(as_tuple=True)[1][0].item()
    
    # Select the indices that were picked from the sampling
    sampled_indices = sorted_indices[0][:target_index+1].unsqueeze(0)
    # Use the selected indices to keep the probabilities of the original input
    sampled_probs = torch.gather(logprobs, -1, sampled_indices)

    return sampled_probs, sampled_indices

def nucleus_sampling_testing(logprobs, p):

    # By taking the exponent the values turn into probabilities, which will allow for deciding when P has been met
    probs = torch.exp(logprobs)
    #print("Shape of probs:", probs.shape)
    #print("Current probs:", probs)
    # Sort the probabilities and corresponding indices in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    print("Current sorted_probs:", sorted_probs)
    
    # Compute the cumulative probabilities
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    print("Current cumulative_probs:", cumsum_probs)
    
    # Find the target point where cumulative probability exceeds p
    # First make a tensor of P that matches the shape of logprobs, so for each row (target image, P is applied)
    tensor_p = torch.full((logprobs.shape[0],1), p)
    # Make the target mask, that has 1's for every value that exceeds P, which means the first 1 is the target index
    target_mask = (cumsum_probs > tensor_p).int()
    first_ones = ((target_mask.cumsum(dim=1) == 1) & (target_mask == 1)).int()
    # keep in mind, that this combined mask makes every possible index 1.0 and non possible indexes 0.0 which is perfect for the multinomial
    combined_mask = ((target_mask == 0) | first_ones).float()

    # Get the sampled indices, so this will be one index per row, that will represent the chosen action
    # The nucleus indices will be the randomly chosen indices based on the multinomial (which just randomly picks a number from the possible numbers)
    nucleus_indices = torch.multinomial(combined_mask, 1)
    # These are the actual sampled indices which correlate to the original values
    sampled_indices = torch.gather(sorted_indices, -1, nucleus_indices)

    # Use the selected indices to retrieve the probabilities of the original input
    sampled_probs = torch.gather(logprobs, -1, sampled_indices)

    return sampled_probs, sampled_indices

if __name__ == "__main__":
    logprobs = torch.tensor([
        [-5, -2, -2.67807, -1.1610, -0.75974], 
        [-5, -2, -2.67807, -1.1610, -0.75974],
        [-5, -2, -2.67807, -1.1610, -0.75974], 
        [-5, -2, -2.67807, -1.1610, -0.75974]
        ]) 

    #testprobs = torch.tensor([[1, 8, 5, 10]])
    # sorted_probs, sorted_indices = torch.sort(testprobs, descending=True)
    # print(sorted_indices)
    # print(sorted_probs)
    # print(torch.cumsum(sorted_probs, dim=-1))
    # cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # cutoff_index = (cumulative_probs > 20).nonzero(as_tuple=True)[1][0].item()
    # selected_probs = sorted_probs[0][:cutoff_index+1].unsqueeze(0)
    # selected_indices = sorted_indices[0][:cutoff_index+1].unsqueeze(0)
    # print(cutoff_index)
    # print(selected_probs, selected_indices)
    #print(torch.sum(logprobs))
    # print(logprobs.shape)
    # logprobs = logprobs.unsqueeze(0)
    # print(logprobs.shape)
    # print(logprobs)
    # print(8, 25, 21)

    log_probs, indices = nucleus_sampling_testing(logprobs, 0.5)
    # log_probs, indices = nucleus_sample_logits(logprobs, 0.5)
    print(log_probs.shape)
    print(log_probs)
    print(indices.shape)
    print(indices)