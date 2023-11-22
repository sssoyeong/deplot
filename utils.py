import pickle


def save_variables(filename, *variables):
    with open(filename, 'wb') as f:
        for var in variables:
            pickle.dump(var, f)
    print(f'saved file: {filename}')

# def load_variables(filename, )