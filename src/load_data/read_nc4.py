class Loader():
    def __init__(self) -> None:
        pass    
    
    def __len__(self) -> int:
        return 1
    
    def get_batch(self, id):
        pass

    def add_forced_variables(self, dataset):
        pass

surface_var = ['emis', 'frlake', 'frland', 'frlandice', 'frocean', 'frseaice', 'ts']
col_var = ['fcld', 'o3', 'pl', 'q', 'qi', 'ql', 'ri', 'rl', 't']
force_var = []
