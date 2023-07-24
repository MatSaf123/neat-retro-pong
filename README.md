# Solving ATARI Retro Pong with NEAT RL Algorithm

Using Neuroevolution of Augmented Topologies (NEAT) in Python in order to solve ATARI Pong simulation provided by OpenAI Gym library. More interesting parts of this project in my opinion were the image pre-processing, multiple worker-AIs learning on multiple threads setup and setting up the reinforced-learning loop for the AI.

<p align="center">    
    <img src=https://user-images.githubusercontent.com/56278688/232304505-77fde359-7052-4639-b71b-c3ab592cbfb0.gif width="20%" height="10%">
</p>

# To run
1. Create new venv via your favorite python tool (mine is `python3 -m venv venv`)
2. Run
```
pip install -r requirements.txt
```
3. Run app via the basic CLI

# CLI
Very basic and crude, but somewhat working. Example usages:

`python3 main.py pong train new`

`python3 main.py train checkpoints/checkpoint-1.foo`

`python3 main.py pong test`

# Random notes
- On Linux I also needed `sudo apt install swig`, and I did `pip install box2d-py` manually as well.
