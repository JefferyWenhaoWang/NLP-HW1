import tty, sys, termios

def getchar():
    """Read in one character from stdin.
    http://code.activestate.com/recipes/134892/"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def getline(prompt=''):
    """Reads a line from stdin with text prediction and returns it,
    without the trailing newline. If the user presses Ctrl-D at the
    beginning of the line, raises EOFError.
    """

    print(prompt, end='', flush=True)
    q = lm.start()
    q, p = lm.step(q, lm.vocab.numberize('<BOS>'))
    states = [q]
    probs = [p]
    chars = []
    prediction = []

    def erase():
        """Erase the previous prediction without moving the cursor."""
        if len(prediction) > 0:
            print(' '*len(prediction) + f'\x1b[{len(prediction)}D', end='', flush=True)
    
    while True:
        
        # Make prediction
        erase()
        prediction = []
        q = states[-1]
        p = probs[-1]
        for i in range(20):
            c = max(lm.vocab, key=lambda c: p[lm.vocab.numberize(c)])
            if c == '<EOS>':
                break
            prediction.append(c)
            q, p = lm.step(q, lm.vocab.numberize(c))
        
        # Print prediction
        if len(prediction) > 0:
            print('\x1b[7m' + ''.join(prediction) + f'\x1b[0m\x1b[{len(prediction)}D', end='', flush=True)

        # Read and process character
        c = getchar()
        if c == '\x03': # Ctrl-C
            erase()
            raise KeyboardInterrupt()
        elif c == '\x04': # Ctrl-D
            if len(chars) == 0:
                erase()
                print('')
                raise EOFError()
            else:
                print('\a', end='')
        elif c in ['\r', '\n']: # Return
            erase()
            print('', flush=True)
            break
        elif c in ['\x7f', '\b']: # Backspace
            if len(chars) == 0:
                print('\a', end='')
            else:
                erase()
                print(f'\x1b[1D', end='', flush=True)
                states.pop()
                probs.pop()
                chars.pop()
        else:
            print(c, end='', flush=True)
            q = states[-1]
            q, p = lm.step(q, lm.vocab.numberize(c))
            states.append(q)
            probs.append(p)
            chars.append(c)
    return ''.join(chars)

if __name__ == "__main__":
    
    # Process command-line arguments (change as needed)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='train')
    args = parser.parse_args()

    # Replace the following lines with an instantiation of your model
    import models.ngram.unigram
    data = [list(line.rstrip('\n')) for line in open(args.train)]
    lm = unigram.Unigram(data)
    
    # End replace

    while True:
        try:
            line = getline('> ')
        except EOFError:
            break
