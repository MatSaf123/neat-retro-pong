"""Basic CLI for me to be able to run NEAT in different ways"""

import sys
from neat_pong import pong

if __name__ == "__main__":

    args = sys.argv[1:]
    print(args)

    # Print avaiable stuff if --help was passed, then bail
    if "--help" in args:
        print(
            """
NEAT-AI CLI [v0.0.1]
Basic CLI for me to be able to run NEAT in different ways.

Arg 1: select project (currently available: [pong])
Arg 2: select mode (currently available: [train, test])
Arg 3: use checkpoint? (pass filepath for yes, leave blank or `none` for no)
Arg 4: TODO: render mode?
        """
        )
        sys.exit()

# Attempt at running NEAT stuff otherwise

if args[0] not in ["pong"]:
    print("No such project as {}".format(args[0]))
    sys.exit(1)

if args[1] not in ["train", "test"]:
    print("No such mode as {}".format(args[1]))
    sys.exit(1)

checkpoint_path = None
# Checkpoint param is optional
if len(args) > 2:
    if args[2] == "none":
        pass
    else:
        checkpoint_path = args[2]

# Run neat-pong
if args[0] == "pong":
    pong.run(args[1], checkpoint_path)
