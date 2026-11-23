# sensor_firesync

A minimal agent-based model that links one property of a visual sensor, its temporal resolution, to one swarm-level outcome: how tightly a group of flashing agents synchronizes.

The model extends the integrate-and-fire model of single-firefly flash timing in Martin et al. (2026), "Excitation-inhibition interactions mediate firefly flash synchronization", bioRxiv 2026.01.19.700439 (code and data: [peleg-lab/led_firesync](https://github.com/peleg-lab/led_firesync)), to groups of agents whose sensor has a finite temporal resolution.

![Flash-time spread as a function of sensor temporal resolution](figure_sensor_resolution.png)

**Figure.** (A) Flash times of 20 simulated agents whose sensor has the temporal resolution of the firefly eye (40 Hz, sampling period 25 ms). Left: three seconds of one run. Right: the shaded group flash, enlarged; the bar shows one sampling period, and the arrow marks two flashes separated by the flash-time spread (9 ms). (B) Flash-time spread (median and interquartile range of 20 runs) as a function of sensor temporal resolution for groups of 5, 20 and 50 agents. The shaded band marks the flicker fusion frequency of the *Photuris frontalis* eye (35 to 50 Hz, from electroretinography), the dashed line the duration of one flash (25 ms), and the solid gray line the spread expected with no synchrony (one quarter of the flash period).

## Model

- **Agent.** An integrate-and-fire phase oscillator with the natural flash period of *P. frontalis* (0.64 s, 5% variation between agents). The phase rises at a constant rate, the agent flashes when the phase reaches 1, and the phase resets to 0. When the agent perceives a flash, its phase shifts by an amount that depends on its current phase (a phase response curve).
- **Sensor.** The eye samples in windows of length 1/f, where f is the temporal resolution in Hz. All neighbor flashes that begin inside one window are perceived as a single event, reported when the window closes. The agent therefore cannot order or count flashes that are closer together than one sampling period.
- **Outcome.** The flash-time spread: the median time between a flash and the nearest flash of each other agent, over the last 20 s of a 60 s run. Zero means perfect zero-lag synchrony. The script also reports the Kuramoto order parameter and the flash period.

## Run

Requires Python 3 with NumPy and Matplotlib.

```
pip install -r requirements.txt
python firefly_swarm_sensor_sweep.py --report            # sweep + the figure above (about 3 minutes)
python firefly_swarm_sensor_sweep.py --report --from-csv sweep_restoring.csv   # redraw the figure from the saved sweep
python firefly_swarm_sensor_sweep.py --single 40 --n 20  # one condition, prints the metrics
python firefly_swarm_sensor_sweep.py --prc measured      # same sweep with the measured firefly response curve
python firefly_swarm_sensor_sweep.py --k 8               # each agent sees 8 random neighbors
python firefly_swarm_sensor_sweep.py --help              # all options and a full description of the model
```

`--report` writes `sweep_restoring.csv` (one row per group size and temporal resolution) and `figure_sensor_resolution.png`. Both files in this repository were produced by that command with the default settings: 20 runs of 60 s per condition, random initial phases, every agent sees every other agent.

## Results of the first simulations

- At the temporal resolution measured for the firefly eye (35 to 50 Hz), every simulated group synchronized, with a flash-time spread of 8 to 10 ms. This is about one third of the duration of a flash and about 0.4 of the sampling period.
- The spread shrinks as the sensor gets faster: about 25 ms at 10 Hz and 5 ms at 100 Hz.
- Synchrony is lost when the sensor is slowed below about 5, 7 and 10 Hz for groups of 5, 20 and 50 agents. Larger groups need a faster sensor.

## Limits

- The default response rule (`--prc restoring`) is a generic phase-restoring rule, not the measured firefly response. With the measured single-source response curve of Martin et al. (`--prc measured`), mutually coupled agents do not reach zero-lag synchrony at any temporal resolution. The rule for mutual interaction between many flashers is an open modeling step.
- The model does not yet include spatial resolution, sensing range, flash duration and brightness, or movement.

## Funding

This work was supported by Air Force Research Laboratory award FA8651-25-2-0005.
