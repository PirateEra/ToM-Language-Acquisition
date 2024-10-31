import torch
def generate_ngrams(sequence, n=1):
    ngrams_set = set()
    sequence = sequence[0].tolist()
    # Generate n-grams
    print(sequence)
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i:i + n])
        ngrams_set.add(ngram)
    return ngrams_set

if __name__ == "__main__":
    initial_K = 10
    bos_id = 2
    nucleus_outputs = torch.full((initial_K, 1), bos_id, dtype=torch.long)
    # nucleus_last_words = torch.full((initial_K, 1), 5, dtype=torch.long)
    # nucleus_outputs = torch.cat((nucleus_outputs, nucleus_last_words), dim=-1)
    # print(nucleus_last_words)
    # final_mask = (nucleus_last_words == 5).float()
    # print(final_mask)

    # test_tensor = torch.tensor([[0.],
    #     [0.],
    #     [1.],
    #     [0.],
    #     [0.],
    #     [0.],
    #     [0.],
    #     [0.],
    #     [0.],
    #     [0.]])
    
    # current_final_mask = torch.tensor([[0.],
    #     [0.],
    #     [0.],
    #     [0.],
    #     [1.],
    #     [0.],
    #     [0.],
    #     [1.],
    #     [0.],
    #     [0.]])
    
    # final_mask = ((test_tensor.bool() | current_final_mask.bool())).float()
    # print(nucleus_outputs.shape)
    # nucleus_outputs[final_mask == 1] = 3
    # batch_size = 8
    # sequence_length = 21
    # final_outputs = torch.zeros(batch_size, initial_K, sequence_length)
    # test_values = torch.full((initial_K, sequence_length), 5, dtype=torch.long)
    # print(final_outputs)
    # final_outputs[5] = test_values
    # print(final_outputs)
    # print(final_outputs.shape)
    # print(test_values.shape)
    B = 10
    EOS = 2
    #print(torch.full((B,), EOS))
    #action = torch.zeros((B,))
    # print(action)
    # beamprobs = torch.randint(low=0, high=10, size=(10,))
    # actions = torch.randint(low=0, high=10, size=(10, 10))
    # total_beamprobs = actions + beamprobs.unsqueeze(1)
    # print("Original tensor\n", actions)
    # print("Summed tensor (grab top k from this)\n", total_beamprobs)
    # top_values, top_indices = torch.topk(total_beamprobs.view(-1), B)
    # print(top_values.shape)
    # top_indices_2d = [(idx.item() // B, idx.item() % B) for idx in top_indices]

    # top_values = torch.tensor([actions[idx[0], idx[1]].item() for idx in top_indices_2d])
    # new_beams = torch.stack([actions[idx[0]] for idx in top_indices_2d])
    # print(new_beams)

    # print("The top indices are", top_indices_2d)
    # print("The top values of the original actions are", top_values.shape)
    # for i in range(2):
    #     print(i)
    # test = torch.randint(low=0, high=10, size=(10,21))
    # G = 2
    # B = 10
    # g = B // 2
    # subgroups = torch.split(test, g)
    # print(subgroups)
    # for group in subgroups:
    #     print(group.shape)
    # test = torch.randint(low=0, high=10, size=(2,21))
    # testarray = [test, test, test]
    # print(torch.cat(testarray).shape)

    # print(test.shape)
    # print(test)
    # extra = torch.randint(low=0, high=10, size=(10,1))
    # test[:, 1] = extra[:, 0]
    # print(test)
    # Example tensor
    #tensor = torch.tensor([1, 2, 3, 4, 5, 5, 7, 5, 9, 10])
    test = torch.tensor([[3, 2, 5],
                     [8, 4, 1],
                     [6, 7, 9],
                     [11, 13, 12],
                     [4, 2, 21],
                     [18, 5, 11]])

    # Define the EOS_ID value
    EOS_ID = 5

    # Find the indices where the tensor values equal EOS_ID
   # finished_sequences = tensor != EOS_ID
    # print(test.shape)
    # print(test[finished_sequences].shape)
    # EOSid = 3
    # actions = torch.randint(low=0, high=10, size=(10,10))
    # all_beamprobs = torch.randint(low=0, high=100, size=(10,10))
    # unfiltered_top_values, top_indices = torch.topk(all_beamprobs.view(-1), all_beamprobs.view(-1).shape[0])
    # # Convert the top_indices to their respective 2d coordinates, so they are actually useable
    # # So the whole // and & is simply to turn a value like 8 in a 10x10 tensor to (0,7) or a value like 12 into (1,1)
    # unfiltered_top_indices_2d = [(idx.item() // all_beamprobs.shape[1], idx.item() % all_beamprobs.shape[1]) for idx in top_indices]
    # # These are the top actions of this iteration
    # counter = 0
    # top_actions = []
    # top_indices_2d = []
    # unfinished_indices = []
    # top_values = []
    # # Loop through the top_indices, until 10 are found (excluding finished sequences)
    # # I also save the finished sequences that were found to include them at the end, since they were above the top 10 unfinished
    # for i, index in enumerate(unfiltered_top_indices_2d):
    #     if counter >= 10: # Exit this for loop if 10 actions were found
    #         break
    #     temp_action = actions[index[0], index[1]] # Get the action
    #     if temp_action != EOSid: # If it is a unfinished sequence, we keep it in its respective lists
    #         counter += 1
    #         top_indices_2d.append(index)
    #         top_actions.append(temp_action)
    #         top_values.append(unfiltered_top_values[i])
    #     else: # store this finished sequence since it was above the top 10 unfinished, so it'll be good to reconsider at the end
    #         print("unfinished")
    #         unfinished_indices.append((index, unfiltered_top_values[i]))
    
    # top_actions = torch.tensor(top_actions) # Turn top actions into a tensor
    # top_values = torch.tensor(top_actions) # Turn top values into a tensor
    # print(test)
    # print(top_values)
    # print(top_indices_2d)
    # print(torch.tensor([1,2,5,8]).shape)
    # testing = torch.tensor([1,2,5,8])
    # # tester = torch.tensor([1,2,5,8])
    # # tester = torch.cat((testing, tester))
    # test = torch.randint(low=0, high=10, size=(10,21))
    # #testing = torch.randint(low=0, high=10, size=(10))
    # # testy = torch.randint(low=0, high=10, size=(10,21))
    # # print(torch.cat((test, testy), dim=0).shape)
    # nice = [test, test, test]
    # nicer = [testing, testing, testing]
    # #print(torch.stack(nice).view(-1, nice[0].shape[-1]).shape)
    # print(torch.cat(nice).shape)
    #print(test[torch.tensor([0,3,5])])
    # x = 5
    # print([[] for _ in range(x)])
    # test = torch.randint(low=0, high=10, size=(10,21))
    # test = torch.split(test, 5)
    # print(len(test))
    # testing = torch.tensor(
    #     [[  2,   4, 171,   7,   4,  67,  71,   6,   8,  26,   5,  47],
    #     [  2,   4, 171,   7,   4,  67,  71,   6,   8,  26,   5, 104],
    #     [  2,  66,  12,  29,   2,   4,  25,  15,   6,  35,   7,   4],
    #     [  2,   4,  63, 119,  15, 161,   7,   4,  24, 190,   5,   3],
    #     [  2,  66,  12,  29,   2,   4,  25,  15,   6,   4, 177,  30],
    #     [  2,   4,  63, 119,  15, 161,   7,   4,  24, 190,   5,   3],
    #     [  2,   9,   4,  60,  10,   4, 186,  18, 124,  47,   5, 104],
    #     [  2,  75,  98,  12,   1, 175,  11,   4,  32,  85,   3,   0],
    #     [  2,  23,  14,   4,  50, 175,   2,   4, 173, 113,  67, 181],
    #     [  2,   4,  63, 119,  15, 161,   7,   4,  24, 190,   5,   3]]
    #     )
    # testing_2 = torch.tensor(
    #     [[  2,   4, 171,   7,   4,  67,  71,   6,   8,  26,   5,  47],
    #     [  2,   4, 171,   7,   4,  67,  71,   6,   8,  26,   5, 104]]
    #     )
    # duplicate_rows = []
    # for i, row in enumerate(testing):
    #     for row_2 in testing_2:
    #         if row.equal(row_2):
    #             print(f"{row} is in both")
    #             duplicate_rows.append(i)
    # testing = testing[~torch.tensor(duplicate_rows).unsqueeze(1).eq(torch.arange(testing.size(0))).any(0)]
    # print(testing)
    # testing_2 = torch.tensor(
    # [[  2,   4, 171,   7,   4,  67,  71,   6,   8,  26,   5,  47],
    # [  2,   4, 171,   7,   4,  67,  71,   6,   8,  26,   5, 104]]
    # )
    # testin_bool = torch.tensor([False, False])
    # # print(torch.cat(testing_2, dim=0))
    # print(len(testing_2[testin_bool]))
    # print(torch.exp(torch.tensor([0,3,5])))
    # print(1 / torch.exp(torch.tensor([0,3,5])))
    # testing_2 = torch.tensor(
    # [[  2,   4, 171,   7,   4,  67,  71,   6,   8,  26,   5,  47],
    # [  2,   4, 171,   7,   4,  67,  71,   6,   8,  26,   5, 104]]
    # , dtype=float)

    # testing_2 = testing_2[0]
    # testing_2 = testing_2[:3+1]
    # testing_2 = torch.mean(testing_2).item()
    # print(testing_2)
    entropies = torch.Tensor(10, 10)
    # test_tensor = torch.tensor([[0.],
    # [0.],
    # [1.],
    # [0.],
    # [0.],
    # [0.],
    # [0.],
    # [0.],
    # [0.],
    # [0.]])
    # print(test_tensor.shape)
    # print(torch.cat((entropies, entropies), dim=0).shape)
    test = torch.tensor([[1, 2, 5, 8 ,9, 8, 3]])
    print(len(generate_ngrams(test, 3)))


    # def get_action_and_probs_diverse_beam_sampling(
    #     self, images: torch.FloatTensor, actions: Optional[torch.LongTensor] = None, BOSid=2, EOSid=3, G=5) -> Tuple[
    #     torch.LongTensor,  # actions (batch_size, max_len)
    #     torch.FloatTensor,  # logprobs (batch_size, max_len)
    #     torch.FloatTensor,  # entropy (batch_size, max_len)
    #     torch.FloatTensor,  # values (batch_size, max_len)
    # ]:
    #     B = images.size(0) # This works because im using batch size 1, so doing this will equal the beam size instead
    #     device = images.get_device()
    #     image_representation = self.encoder(images)
    #     # So this is to feed the GRU with the first layer, which is just a bunch of zeros which will result in BOS id's when fed
    #     ix = torch.zeros_like(image_representation)
    #     hx = image_representation
    #     # Diverse related variables
    #     # G stands for the amount of groups, and g for the amount of beams per group
    #     g = B // G # This is the amount that each group will have, since torch.split is based on the amount per tensor
    #     #######################################################
    #     # INITIAL START (BOS AND THE FIRST B AMOUNT OF WORDS)
    #     #######################################################
    #     # Start off each sentence with the BOS and setup the GRU stuff for the beam search (see it as a do while)
    #     # The first iteration is for BOS and the second to get the actual first B amount of words (the first iteration is not used)
    #     # But i do do it to update hx to make sure its properly setup (i got no clue how its trained)
    #     hx, logits = self.beamsearch_update_logits(hx, ix) # Get the logits (which will result into BOS, since its the first iteration)
    #     _, actions = torch.topk(torch.exp(logits), 1) # Should technically all be BOS, which is the BOS id for the first iteration
    #     #action = action.squeeze(-1)    # So originally its like a tensor [[2], [2], [2]] and it is turned into [2, 2, 2, 2,] so from [B, 1] to [B] shape wise
    #     logprob = torch.gather(logits, dim=-1, index=actions).squeeze(-1) # Get the logprobs that belong to the actions (should be shape B, so a tensor of B values)
    #     beamactions = actions # should be of shape B,1 so for every sequence the first action which is BOS
    #     beamprobs = logprob
    #     ix = self.word_emb(actions.squeeze(-1)) # Update IX (embed the current words, all BOS)

    #     # SECOND ITERATION, which is obtaining the top 10 best words to start off with, this is the starting setup for beam search
    #     hx, logits = self.beamsearch_update_logits(hx, ix) # Get the logits (which will result into B of the same rows, since all rows are based on BOS)
    #     _, actions = torch.topk(torch.exp(logits[0]), B) # This is important, only do top K for the first row (since all the rows are the same), this way u get 10 different start words
    #     logprob = torch.gather(logits, dim=-1, index=actions.unsqueeze(1)).squeeze(-1) # Get the logprobs that belong to the actions (should be shape B, so a tensor of B values)
    #     beamactions = torch.cat((beamactions, actions.unsqueeze(1)), dim=1) # Add the new words to the beam actions which is resulting in the first column being BOSid and the 2nd column being non-identical ids, unsqueeze to turn B shape into B,1
    #     beamprobs = torch.add(logprob, beamprobs) # Sum the probabilities of the BOS and the new words to get the full probability of the beam unsqueeze to add go from B to B,1 to match logprob
    #     ix = self.word_emb(actions.squeeze(-1)) # Update IX (embed the current actions (just added words))
    #     #######################################################
    #     # ACTUAL BEAM SEARCH
    #     ########################################################
    #     completed_sentences = [[] for _ in range(G)]
    #     i2w = torch.load("i2w") # Only needed when debugging
    #     for i in range(self.max_len):
    #         print(f"Step {i + 1}/{self.max_len}")
    #         # Obtain the logits
    #         hx, logits = self.beamsearch_update_logits(hx, ix)
    #         group_beams = []
    #         group_probs = []
    #         group_actions = []
    #         sub_beams = torch.split(beamactions, g)
    #         # Do beam search per group
    #         for j in range(G):
    #             print(f"Group {j + 1}/{G}")
    #             # Expand each sequence with the possible (beam size) options (this is pretty much pruning the vocabulary before actually selecting the top K beams)
    #             logprob, actions = torch.topk(torch.exp(logits), B) # Shape of this should be [B, B], which would be the top B actions for each sequence in B
    #             logprob = torch.gather(logits, dim=-1, index=actions).squeeze(-1) # Grab the logprobs of each of the actions that were decided upon in the line above
    #             # Get the probabilities for each possible beam in the iteration, which is a total of B*B sequences
    #             all_beamprobs = logprob + beamprobs.unsqueeze(1) # so this is again a tensor of shape B,B but with the total probabilities
    #             # Use topK to automatically order what beams are most likely (from this the top B will be grabbed excluding finished sentences)
    #             # The view(-1) pretty much flattens the all_beamprobs, so that it can easily grab the topk out of a 2d tensor
    #             # all_beamprobs.view(-1).shape[0] makes sure every single value is returned from the topk which pretty much causes the full ordering
    #             unfiltered_top_values, top_indices = torch.topk(all_beamprobs.view(-1), all_beamprobs.view(-1).shape[0])

    #             # Convert the top_indices to their respective 2d coordinates, so they are actually useable
    #             # So the whole // and & is simply to turn a value like 8 in a 10x10 tensor to (0,7) or a value like 12 into (1,1)
    #             unfiltered_top_indices_2d = [(idx.item() // all_beamprobs.shape[1], idx.item() % all_beamprobs.shape[1]) for idx in top_indices]
    #             # These are the top actions of this iteration
    #             counter = 0
    #             top_actions = []
    #             top_indices_2d = []
    #             top_values = []
    #             finished_indices = []
    #             finished_values = []
    #             # Loop through the top_indices, until g are found (excluding finished sequences)
    #             # I also save the finished sequences that were found to include them at the end, since they were above the top 10 unfinished
    #             for i, index in enumerate(unfiltered_top_indices_2d):
    #                 if counter >= g: # Exit this for loop if g actions were found
    #                     break
    #                 temp_action = actions[index[0], index[1]] # Get the action
    #                 if temp_action != EOSid: # If it is a unfinished sequence, we keep it in its respective lists
    #                     counter += 1
    #                     top_indices_2d.append(index)
    #                     top_actions.append(temp_action)
    #                     top_values.append(unfiltered_top_values[i])
    #                 else: # store this finished sequence since it was above the top 10 unfinished, so it'll be good to reconsider at the end
    #                     finished_indices.append(index)
    #                     finished_values.append(unfiltered_top_values[i])
                
    #             top_actions = torch.tensor(top_actions).to(device)# Turn top actions into a tensor
    #             top_values = torch.tensor(top_values).to(device)# Turn top values into a tensor

    #             # Deal with the finished sentences by storing them
    #             if len(finished_indices) > 0:
    #                 finished_sequences = torch.stack([sub_beams[j][idx[0]] for idx in finished_indices]) # Get the finished sequences
    #                 finished_sequences = torch.cat((finished_sequences, torch.full((finished_sequences.shape[0], 1), EOSid).to(device)), dim=1) # Add a column of EOS id's to the finished sequences to finish them
    #                 completed_sentences[j].append((finished_sequences, finished_values)) # Store the finished sequences and values as a tuple for later reference
    #             print("The top indices are: ", top_indices_2d)
                
    #             print("The top (current) actions are", top_actions.shape, top_actions)
    #             # Create the sequences from the top actions and the top indices
    #             new_beams = torch.stack([sub_beams[j][idx[0]] for idx in top_indices_2d]) # This keeps the beams from the previous iteration (without the new actions), stack simply turns a list of B tensors into a tensor of B tensors
    #             # print("The shape of the new beams (without the new actions)", new_beams.shape)
    #             # Add the new actions to the beams (the unsqueeze is so they can be concatenated, its a shape 10 thats turned into a shape 10,1)
    #             new_beams = torch.cat((new_beams, top_actions.unsqueeze(1)), dim=1)

    #             # Add this groups values to the group lists to be combined after the group beam searches have been finished
    #             group_beams.append(new_beams)
    #             group_probs.append(top_values)
    #             group_actions.append(top_actions)

    #             # Update the logits (this is what causes diversity, due to lowering the chance of the actions that just got chosen so the other groups wont)
    #             for action in top_actions:
    #                 for row in range(logits.shape[0]):
    #                     # logits[index[0], index[1]] = logits[index[0], index[1]] - self.huge # subtract a huge value from the logit to make it unlikely
    #                     logits[row, action] = logits[row, action] - self.huge
                
    #             # Debugging
    #             generated_sentences = [
    #                 ' '.join([i2w[token_id.item()] for token_id in sentence])
    #                 for sentence in new_beams
    #             ]
    #             for temp_sentence in generated_sentences:
    #                 print(temp_sentence)
            
    #         # Since the final_lists are lists of tensors for each groups chosen values, they have to be combined into 1 tensor
    #         top_actions = torch.cat(group_actions)
    #         # Turn beamprobs into the new beams their logprobs, so this now contains the total probability for each kept beam
    #         beamprobs = torch.cat(group_probs)
    #         beamactions = torch.cat(group_beams)

    #         # This embeds each action (word) so it can be fed into the GRU, essentially it  turns the action into a vector of 512 (layer size)
    #         ix = self.word_emb(top_actions)
    #         print("\n\n")
        
    #     # Normalize the beam probabilities, using length normalization
    #     # This could potentially mess with the reranking, im not sure if it needs the original log probs or not (but im guessing not)
    #     beamprobs = beamprobs / beamactions.shape[1]
    #     # Re evaluate, by grabbing the top K sequences, based on the completed and leftover sequences
    #     max_size = beamactions.size(1) # Get the size the 2nd dimension has to be
    #     # Re evalution per group, so that in the end the sentences dont get mixed up
    #     # This is because the probabilities of the groups that are not group 1 are bound to be lower, 
    #     # which will remove them at the very end if not done in a seperate group
    #     sub_beam_probs = torch.split(beamprobs, g)
    #     sub_beams = torch.split(beamactions, g)
    #     final_beams = []
    #     final_probs = []
    #     for i in range(G):
    #         padded_finished_sequences = []
    #         current_probs = sub_beam_probs[i]
    #         current_beams = sub_beams[i]
    #         for tensor in completed_sentences[i]:
    #             probabilities = torch.stack(tensor[1]).to(device) # Get the probabilities
    #             tensor = tensor[0] # Since its a tuple, grab the actual tensor (be aware im resetting the tensor variable here)
    #             sequence_length = tensor.shape[1] # This will be needed to normalize the score
    #             probabilities = probabilities / sequence_length # Normalize the probabilities, using length normalization
    #             padding = (0, max_size - tensor.size(1))  # get the amount of padding needed for this tensor
    #             padded_tensor = torch.nn.functional.pad(tensor, padding, "constant", 0) # Add the padding (0 is padding id)
    #             padded_finished_sequences.append(padded_tensor) # Add the padded sequence tensor to work with it
    #             current_probs = torch.cat((current_probs, probabilities)) # Add the probabilities to the sequences to later get the top k
            
    #         padded_finished_sequences = torch.cat(padded_finished_sequences, dim=0) # Add all the padded finishes sequences together
    #         current_beams = torch.cat((padded_finished_sequences, current_beams), dim=0) # Add the finished sequences to the other sequences

    #         # Grab the top K from all the gathered sequences
    #         current_probs, indices = torch.topk(current_probs, g) # This will contain the top B logprobs in beamprobs and the top B indices of the top B sequences
    #         current_beams = current_beams[indices] # Extract the needed sequences from beamactions
    #         # Add this groups probs and beams to the final beams/probs
    #         final_probs.append(current_probs)
    #         final_beams.append(current_beams)

        
    #     # Join the beamprobs and beamactions since they are still in groups at the moment
    #     beamprobs = torch.cat(final_probs)
    #     beamactions = torch.cat(final_beams)

                
    #     return beamactions, beamprobs
    