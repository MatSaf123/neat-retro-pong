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

Arg 1: select mode (currently available: [train, test])
Arg 2: use checkpoint? (pass filepath for yes, leave blank or `new` for no)
Arg 3: pong env stochastic (random-ish) or deterministic? (available: [stochastic, deterministic])
        """
        )
        sys.exit()

if args[0] not in ["train", "test", "visualize"]:
    print("No such mode as {}".format(args[1]))
    sys.exit(1)

# Checkpoint param for training is optional, but pickle file for testing is not
filepath = None

if args[0] == "train":
    if len(args) > 2:
        if args[1] == "new":
            pass
        else:
            filepath = args[1]

if args[0] == "test" or args[0] == "visualize":
    if len(args) > 2:
        filepath = args[1]
    else:
        print(
            "Testing/visualization mode requires filepath to pickled model as third arg"
        )
        sys.exit(1)


if args[2] not in ["deterministic", "stochastic"]:
    print("No such mode as {}".format(args[2]))

# Run neat-pong
pong.run(args[0], args[2], filepath)
