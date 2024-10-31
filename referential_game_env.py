from random import sample
from typing import Dict, List, Tuple
import numpy as np
from listener import Listener
from math import log
from find_image import find_image_multi
from collections import Counter
import torch
import pdb
import pandas as pd
from IPython.display import display, HTML

def H(n):
    #move somewhere else! calculates nth harmonic number
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)

class ReferentialGameEnv(object):
    def __init__(self, *,
                 max_len: int,
                 eos_id: int,
                 noop_penalty: float,
                 length_penalty: float,
                 batch_size: int,
                 n_distr: int,  # this might not be general
                 distribution: str = "uniform",
                 game_file_path: str = "game_file.pt",
                 captions_file: str,
                 hard = False,
                 **kwargs) -> None:
        super().__init__()
        self.hard = hard
        self.max_len = max_len
        self.eos_id = eos_id
        self.noop_penalty = noop_penalty
        self.length_penalty = length_penalty
        self.batch_size = batch_size
        self.n_distr = n_distr
        self.distribution = distribution
        self.game_file_path = game_file_path
        self.image_size = (2048,)
        self.listener = Listener(**kwargs).cuda()
        self.target_ids = None
        self.captions_file_path = captions_file
        self.caption_length = 9
        self._get_game_file()

        # Code for rendering the images
        self.img_df = pd.read_parquet('./data/mscoco.parquet')
        # This filters the original mscoco dataset in a way that allows only the first caption per image to stay with the url of it
        # https://huggingface.co/datasets/cat-state/mscoco-1st-caption This is what i used to make this work
        first_caps = self.img_df.drop_duplicates(subset='URL', keep='first')
        first_caps.to_parquet('mscoco-1st-caption.parquet')
        # This alters all captions in a way that they can later be matched with the captions provided by andys code
        self.img_df['TEXT'] = self.img_df['TEXT'].str.lower().str.strip().str.rstrip(".")
        self.img_df['TEXT'] = self.img_df['TEXT'].str.strip()
        self.img_df['TEXT'][0]

    def _get_game_file(self) -> np.array:
        import torch
        self.game_file = torch.load(self.game_file_path)
        for i in self.game_file:
            try:
                self.game_file[i] = torch.from_numpy(np.array(self.game_file[i]))
            except:
                pass

        filter_ids = []
        for idx, cap in enumerate(self.game_file["captions"]):
            if max(cap) < 199:
                filter_ids.append(idx)
        filter_id_set = set(filter_ids)
        filter_id_to_idx = {filter_ids[i]: i for i in range(len(filter_ids))}
        
        self.game_file["captions"] = self.game_file["captions"][filter_ids]
        self.game_file["images"] = self.game_file["images"][filter_ids]
        if "similarity_rank" not in self.game_file:
            self.game_file["sample_candidates"] = torch.from_numpy(
                np.random.randint(low=0,
                    high=len(filter_ids), size=(len(filter_ids), 100)
                )
            )
        else:
            sample_candidates = []
            import tqdm
            for i in tqdm.tqdm(filter_ids):
                tmp = []
                for j in self.game_file["similarity_rank"][i]:
                    j = j.item()
                    if j in filter_id_set and j != i:
                        tmp.append(filter_id_to_idx[j])
                    if len(tmp) == 1000:  # Select the top 100 most similar images
                        break
                sample_candidates.append(tmp)
            self.game_file["sample_candidates"] = torch.LongTensor(sample_candidates)


        #self.captions_file = torch.load(self.captions_file_path)
        #self.captions_file = torch.from_numpy(np.array([[k[:self.caption_length] for k in j] for j in self.captions_file]))
        #self.game_file["all_captions"] = self.captions_file[filter_ids]

        # self.captions_file = torch.load(self.captions_file_path)

        # # Slice off any captions longer than the allowed length
        # self.captions_file = [[k[:self.caption_length] for k in j] for j in self.captions_file]

        # # Add padding to any captions that are too short
        # for captions in self.captions_file:
        #     for caption in captions:
        #         if len(caption) < self.caption_length:
        #             caption.extend([0] * (self.caption_length - len(caption)))

        # self.game_file["all_captions"] = torch.from_numpy(np.array(self.captions_file))[filter_ids]

    def _find_eos(self, actions: np.array) -> List[int]:
        eos_loc = [-1 for _ in range(len(actions))]
        for idx, i in enumerate(actions):
            for j in range(self.max_len):
                if i[j] == self.eos_id:
                    eos_loc[idx] = j+1
                    break
        return eos_loc

    def _new_game(self) -> np.array:
        if not self.hard:
            return self._new_game_easy()
        else:
            return self._new_game_hard()

    def _new_game_easy(self) -> np.array:
        import torch  # using old code here; change to numpy later
        sample_candidates = self.game_file["sample_candidates"]
        n = sample_candidates.size()[1]
        n_img = sample_candidates.size()[0]
        target_images = torch.randint(n_img, size=(self.batch_size,))
        if self.distribution == 'zipf':
            zipf_weights = np.array([1/(i*H(n)) for i in range(1, n+1)])
            distr_array = np.random.choice(n, (self.batch_size, self.n_distr+1), False, zipf_weights)
            distr_images = torch.from_numpy(distr_array)
        else:
            distr_images = torch.randint(n, size=(
            self.batch_size, self.n_distr + 1
        ))
        target_candidates = torch.index_select(
            sample_candidates, 0, target_images.view(-1)
        ).view(self.batch_size, n)
        distr_images = torch.gather(
            target_candidates, 1, distr_images).view(
                self.batch_size, self.n_distr+1
        )
        target_indices = torch.randint(
            self.n_distr + 1, size=(self.batch_size,))
        distr_images[range(self.batch_size), target_indices] \
            = target_images.view(self.batch_size)
        self.distr_images = distr_images.numpy()
        self.target_ids = target_indices.numpy()
        self.images = torch.index_select(
                self.game_file["images"], 0,
                distr_images.view(-1)
            ).view(self.batch_size, self.n_distr+1, *self.image_size).numpy()
        return dict(
            images=self.images,
            images_ids=distr_images.numpy(),
            goal=target_indices.numpy()
        )

    def _new_game_hard(self) -> np.array:
        import torch
        import numpy as np
        
        sample_candidates = self.game_file["sample_candidates"]
        n_img = sample_candidates.size()[0]
        target_images = torch.randint(n_img, size=(self.batch_size,))
        
        distr_images = torch.zeros((self.batch_size, self.n_distr + 1), dtype=torch.long)
        
        for i in range(self.batch_size):
            # Get the candidates for the target
            candidates = sample_candidates[target_images[i]].tolist()
            # Remove the target if needed, from the candidates
            if target_images[i].item() in candidates:
                candidates.remove(target_images[i].item())

            selected_distractors = candidates[:self.n_distr]
            
            # Ensure we have exactly n_distr distractors by adding random unique distractors
            while len(selected_distractors) < self.n_distr:
                random_distractor = torch.randint(n_img, size=(1,)).item()
                if random_distractor not in selected_distractors and random_distractor != target_images[i].item():
                    selected_distractors.append(random_distractor)

            distr_images[i, :self.n_distr] = torch.tensor(selected_distractors)
            distr_images[i, self.n_distr] = target_images[i]

        # Shuffle to ensure the target is at a random position
        target_indices = torch.randint(self.n_distr, size=(self.batch_size,))
        for i in range(self.batch_size):
            target_position = target_indices[i].item()
            # Save the target value
            target_value = distr_images[i, target_position].item()
            
            # Swap the values
            distr_images[i, target_position] = distr_images[i, self.n_distr]
            distr_images[i, self.n_distr] = target_value

        self.distr_images = distr_images.numpy()
        self.target_ids = target_indices.numpy()
        self.images = torch.index_select(
            self.game_file["images"], 0,
            distr_images.view(-1)
        ).view(self.batch_size, self.n_distr + 1, *self.image_size).numpy()
        return dict(
            images=self.images,
            images_ids=distr_images.numpy(),
            goal=target_indices.numpy()
        )

    # def _new_game(self) -> np.array:
    #     import torch
    #     import numpy as np
        
    #     sample_candidates = self.game_file["sample_candidates"]
    #     n_img = sample_candidates.size()[0]
    #     target_images = torch.randint(n_img, size=(self.batch_size,))
        
    #     distr_images = torch.zeros((self.batch_size, self.n_distr + 1), dtype=torch.long)
        
    #     for i in range(self.batch_size):
    #         target_image_id = target_images[i].item()
            
    #         # Get the captions of the target and candidate distractors
    #         target_caption = self.game_file["captions"][target_image_id]
    #         candidate_ids = sample_candidates[target_image_id].tolist()
            
    #         # Remove the target image itself from candidate list if present
    #         if target_image_id in candidate_ids:
    #             candidate_ids.remove(target_image_id)
            
    #         # Select n_distr unique distractors based on different captions
    #         selected_distractors = []
    #         while len(selected_distractors) < self.n_distr:
    #             candidate_id = torch.tensor(np.random.choice(candidate_ids, 1))
    #             candidate_caption = self.game_file["captions"][candidate_id]
                
    #             # Check if the candidate has a different caption than the target
    #             if not np.array_equal(target_caption, candidate_caption):
    #                 selected_distractors.append(candidate_id.item())
    #                 candidate_ids.remove(candidate_id.item())
            
    #         # Add the target image to the selected distractors
    #         selected_distractors.append(target_image_id)
            
    #         # Shuffle the selected distractors to randomize the target position
    #         np.random.shuffle(selected_distractors)
    #         distr_images[i, :] = torch.tensor(selected_distractors)
        
    #     self.target_ids = torch.arange(self.batch_size).numpy()
    #     self.distr_images = distr_images.numpy()
    #     self.images = torch.index_select(
    #         self.game_file["images"], 0,
    #         distr_images.view(-1)
    #     ).view(self.batch_size, self.n_distr + 1, *self.image_size).numpy()
        
    #     return dict(
    #         images=self.images,
    #         images_ids=distr_images.numpy(),
    #         goal=torch.arange(self.batch_size).numpy()  # assuming the target index is the last position after shuffling
    #     )






    # def _render(self, actions, return_dict, name="None"):
    #     import torch
    #     i2w = torch.load("i2w")
    #     for i in range(self.batch_size):
    #         print(f"Speaker sentence: {' '.join(i2w[j] for j in actions[i])}")
    #         print("goal\tchoice\timages")
    #         goal = [" " for i in range(self.n_distr+1)]
    #         goal[self.target_ids[i]] = "→"
    #         choice = [" " for i in range(self.n_distr+1)]
    #         choice[return_dict["choice"][i].item()] = "→"
    #         captions = torch.index_select(
    #             self.game_file["captions"], 0,
    #             torch.from_numpy(self.distr_images[i])
    #         ).cpu().tolist()
    #         print(captions)
    #         captions = [
    #             ' '.join(i2w[i] for i in j) for j in captions
    #         ]
    #         for j in range(self.n_distr+1):
    #             print(f"{goal[j]}\t{choice[j]}\t{captions[j]}")
    #         print("\n")

    def caption_to_image_url(self, caption):
        # Filter the caption to make it match the dataframe captions
        caption = caption.split('<EOS>')[0]
        caption = caption.lstrip('<BOS> ') 
        caption = caption.rstrip(' . ')
        caption = caption.strip()

        # Select the url
        result = self.img_df[self.img_df['TEXT'].str.contains(caption, case=False)]
        if not result.empty:
            return result["URL"].values[0]
        
        # If there was no url, (probably some issue in my matching the captions, so this can be fixed by debugging the caption) then it tells you no img was found
        return "IMG not found"



    def _render(self, actions, return_dict, name):
        i2w = torch.load("i2w")
        html_content = "<html>\n<body>\n"
        
        for i in range(self.batch_size):
            output = ' '.join(i2w[token] for token in actions[i])
            output = output.split('<EOS>')[0]
            html_content += f"<h3>Output sentence for game {i+1}:</h3>\n<p>{output}</p>\n"

            html_content += "<h3>Image Results</h3>\n"
            html_content += "<table style='width:100%; border-collapse: collapse;'>\n"

            goal = [" " for _ in range(self.n_distr+1)]
            goal[self.target_ids[i]] = "→"
            choice = [" " for _ in range(self.n_distr+1)]
            choice[return_dict["choice"][i].item()] = "→"
            captions = torch.index_select(
                self.game_file["captions"], 0,
                torch.from_numpy(self.distr_images[i])
            ).cpu().tolist()

            captions = [' '.join(i2w[token] for token in caption) for caption in captions]

            # Write captions
            html_content += "<tr>\n"
            for j in range(self.n_distr + 1):
                html_content += f"\t<td style='padding: 10px; text-align: center;'>{captions[j]}</td>\n"
            html_content += "</tr>\n"

            # Write images
            html_content += "<tr>\n"
            for j in range(self.n_distr+1):
                try:
                    img_link = self.caption_to_image_url(captions[j])
                    if choice[j] == "→" and goal[j] == "→":
                        img_caption = "<br>(GOAL) (RESULT)"
                    elif goal[j] == "→":
                        img_caption = "<br>(GOAL)"
                    elif choice[j] == "→":
                        img_caption = "<br>(RESULT)"
                    else:
                        img_caption = ""
                    html_content += f"\t<td style='text-align: center;'><img src='{img_link}'><br>{img_caption}</td>\n"
                except IndexError:
                    html_content += "\t<td>IMAGE NOT FOUND</td>\n"
            html_content += "</tr>\n"

            html_content += "</table>\n"

        html_content += "</body>\n</html>"
        
        display(HTML(html_content))



    def step(self, actions: np.array, render=False, name="None") -> Tuple[
        Dict[np.array, np.array], np.array
    ]:
        B = actions.shape[0]
        # for gold standard runs
        # import torch
        # actions = torch.index_select(self.game_file["captions"], 0, torch.from_numpy(self.distr_images[range(B), self.target_ids])).view(self.batch_size, -1).numpy()
        # actions = actions[:, :self.max_len]
        
        # listener act
        if True:
            action_len = self._find_eos(actions)
            for idx, i in enumerate(action_len):
                if i == -1:
                    action_len[idx] = self.max_len
            return_dict = self.listener.act(self.images, actions, action_len)
            if render:
                self._render(actions, return_dict, name)

        # observation
        if True:
            obs = dict()
            # feedback
            import torch
            # add choices and controls
            obs["choices"] = return_dict["choice"]
            obs["controls"] = return_dict["control"]
            obs["feedback"] = torch.index_select(
                self.game_file["captions"], 0,
                torch.from_numpy(self.distr_images[range(B), return_dict["choice"].cpu()])
            ).view(self.batch_size, -1).numpy()
            obs["feedback"] = obs["feedback"][:, :self.max_len]

            obs["ground_truth"] = torch.index_select(
                self.game_file["captions"], 0,
                torch.from_numpy(self.distr_images[range(B), self.target_ids])
            ).view(self.batch_size, -1).numpy()
            obs["ground_truth"] = obs["ground_truth"][:, :self.max_len]
            # I fixed the accuracy here, but keep in mind this now only works for batch size 1, so it works for my evaluation, but in the future it might not
            if obs["choices"].item() == self.target_ids:
                obs["accuracy"] = 1 # 1 because batch size 1 means that if its right its just 100%
            else:
                obs["accuracy"] = 0 # 0 because batch size 1 and not a match means its just wrong
            # obs["accuracy"] = sum(acc) / len(acc) if len(acc) else 1/(self.n_distr + 1) # This did not work as planned, old code and is weird
            # new game
            obs.update(self._new_game())

        return obs

    def reset(self):
        return self._new_game()

    def close(self):
        pass

    def get_most_frequent_words(self, vocab_size) -> List[int]:
        return [tup[0] for tup in Counter(torch.flatten(self.game_file["captions"]).tolist()).most_common(vocab_size)]
