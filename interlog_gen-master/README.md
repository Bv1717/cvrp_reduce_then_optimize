# InterLog instance generator

## CLI:

- `preprocess` - preprocess input data
- `generate` - generate instances (including distance matrices)
- `process-only` - run distance matrix generation only

usage info: `python src/main.py --help`

### Examples

- preprocess input data: `python src/main.py preprocess`

- generate instances: `python src/main.py generate --customers 100 --satellites 10 --seed 1`
    - creates an instance with `100` customers, `10` satellites, and using seed `1`

- run distance matrix generation
  only: ``python src/main.py process-only --instance-folder ./resources/instances/100_10_1/ --network-types bike``
    - creates the distance matrices based on the instance located in `./resources/instances/100_10_1/` for `bike`
      network only

### Notes

To calculate the distance matrix, first the OSM data for the problem region is loaded using ``OSMNX``. The loaded
network is cached in the folder ``./resources/cache`` by default. If you experience
issues, try running the generation processes with the ``--no-cache`` option.

### Caveats

For simplicity, ``networkx`` package is used for calculating the distance matrix. This can
be quite slow (1000 customers, 40 satellites takes ~1h per mode). A rudimentary parallelization
using the ``multiprocessing`` package is implemented. Use `--num-processes N` to set the number of
processes `N` spawned. Due to the simplicity, this will provide a small runtime boost, but with
inefficient resource usage (memory requirements might grow linear with `N`). When generating
lots of instances, it might be more efficient to generate them in parallel using only one process
(default) each.  
