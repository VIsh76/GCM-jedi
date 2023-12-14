### Pipeline

- For epoch e in [0,E]:
    Initialize( NN, norm, training, ...)
    data_x, data_y = load_data()
    - For batch in data:
        - For t in in [0, N]:
            data_xd = run_dycore(data_x)
            data_xp = run_physic(data_x)
            data_x = combine(data_xd, data_xp)
        l = Loss(data_x, data_y)
        Back_prop(l)
        Analysis(l)

### In python
if __name__ == '__main__':
    initialize()
    for epoch in range(epochs):
        data = load_data()
        for batch in range(len(data)):
            run_step()
            loss()
            backward()
            analysis()
        analysis()
