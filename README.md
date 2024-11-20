# cot_probing


## setup instructions
1. install python3.11
1. setup your private SSH key
   1. put it under in `.ssh/id_[protocol]`
   1. `chmod 600 [key]`
   1. you can debug with `ssh -T -v git@github.com`
1. clone the repo via ssh `git@github.com:jettjaniak/cot-probing.git`
1. make virtual env `python3.11 -m venv .venv`
1. activate virtual env `source .venv/bin/activate`
1. install project in editable state `pip install -e .`
1. install pre-commit hooks `pre-commit install`
1. run `pytest`

## runs in W&B

 - attn-probes project
   - training only on full CoT
   - using biased FSPs 
   - seeds 1-10
     - wrong evaluation
     - using last instead of best model
   - seeds 11-20
     - last \n included
   - seeds 21-40
     - using activations from 676 questions
     - inconsistent with labeled_qs file
   - seeds 41-50